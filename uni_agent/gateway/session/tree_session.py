"""Tree gateway session and trajectory management with optional terminal snapshots."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import pickle
from dataclasses import dataclass, field, replace
from itertools import repeat
from pathlib import Path
from typing import Any

from fastapi import HTTPException

from uni_agent.gateway.session.codec import MessageCodec
from uni_agent.gateway.session.session import GatewaySession, GenerationOutcome, InflightRequest, SessionPhase
from uni_agent.gateway.session.types import (
    AgentContext,
    InternalGenerationRequest,
    SessionHandle,
    Trajectory,
    UnsupportedMediaError,
)
from uni_agent.rl_insight.adapter import start_generation_span

logger = logging.getLogger(__name__)
_TRAJECTORY_CAPACITY_HINT = (
    "The trajectory limit is prompt_length + response_length. "
    "For this event, Claude Code's output-token-limit error is misleading: increasing "
    "CLAUDE_CODE_MAX_OUTPUT_TOKENS alone will not help. Check the rollout prompt_length/response_length "
    "budget or compact context earlier; the episode may be unfinished."
)


@dataclass(slots=True)
class MessageNode:
    """One backend generation with any unmatched history needed to reach it.

    A node starts with the request's unmatched messages and prompt token delta,
    which may include reconstructed history. It becomes completed in place after
    the model output and decoded assistant message arrive. ``chain_id == 0``
    means a new branch is not committed yet; successful commits allocate the
    next 1-based chain ID, as in ``GatewaySession``.
    """

    node_id: int
    messages: list[dict[str, Any]] = field(default_factory=list)
    parent: MessageNode | None = field(default=None, repr=False)
    children: list[MessageNode] = field(default_factory=list, repr=False)
    status: str = "PENDING"
    prompt_ids: list[int] = field(default_factory=list)
    # Only populated when assistant merging edits the pre-generation token delta.
    pre_assistant_prompt_ids: list[int] | None = None
    response_ids: list[int] = field(default_factory=list)
    response_logprobs: list[float] | None = None
    # Context merge trimming; prompt_ids already includes inserted boundary tokens.
    removed_prefix_token_count: int = 0
    image_data: list[Any] | None = None
    video_data: list[Any] | None = None
    agent_context: AgentContext | None = None
    tools_fingerprint: str | None = None
    min_global_steps: int | None = None
    max_global_steps: int | None = None
    chain_id: int = 0
    message_turn_count: int = 0
    num_coalesced_requests: int = 0
    rollback_count: int = 0
    rollback_dropped_trainable_tokens_total: int = 0

    @property
    def is_root(self) -> bool:
        return self.parent is None and self.node_id == 0


@dataclass(frozen=True, slots=True)
class _MountedGeneration:
    """Request-local tree state kept while its backend generation is running."""

    context_ids: list[int]
    image_data: list[Any] | None
    video_data: list[Any] | None
    target_node: MessageNode
    merge_backup: MessageNode | None


@dataclass(frozen=True, slots=True)
class _TreeActiveChainView:
    """Compatibility view matching the fields used by linear-session callers."""

    chain_id: int
    message_history: list[dict[str, Any]]
    active_tool_schemas: list[dict[str, Any]] | None


class TreeGatewaySession(GatewaySession):
    """Own, synchronize, and optionally persist a shared-prefix trajectory tree."""

    def __init__(
        self,
        handle: SessionHandle,
        codec: MessageCodec,
        *,
        prompt_length: int | None = None,
        response_length: int | None = None,
        sampling_params: dict[str, Any] | None = None,
        enable_last_assistant_rollback: bool = True,
        coalesce_reserved_exact_requests: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            handle=handle,
            codec=codec,
            prompt_length=prompt_length,
            response_length=response_length,
            sampling_params=sampling_params,
            enable_last_assistant_rollback=enable_last_assistant_rollback,
            coalesce_reserved_exact_requests=coalesce_reserved_exact_requests,
            metadata=metadata,
        )
        if type(coalesce_reserved_exact_requests) is not bool:
            raise ValueError("coalesce_reserved_exact_requests must be a bool")
        if type(enable_last_assistant_rollback) is not bool:
            raise ValueError("enable_last_assistant_rollback must be a bool")
        self.root: MessageNode | None = MessageNode(node_id=0, status="COMPLETED")
        self.tool_schemas_by_fingerprint: dict[str, list[dict[str, Any]] | None] = {}
        # Committed chain tips or closed trajectories, ordered by latest commit/materialization.
        self.leaves: dict[int, MessageNode | Trajectory] = {}
        self._routed_experts_by_chain: dict[int, Any] = {}
        self._num_trajectories = 0
        self._next_node_id = 1
        self._tree_lock = asyncio.Lock()
        self._mm_processor_kwargs = dict(codec.mm_processor_kwargs or {})
        log_dir = self._metadata.get("_log_dir")
        self._storage_path = Path(log_dir) / "trajectory_tree.pkl" if log_dir else None

    async def run_generation(self, request: InternalGenerationRequest, backend) -> GenerationOutcome:
        """Coalesce identical requests and overlap backend calls within the session."""
        agent_context = request.get("agent_context")
        request = {
            **request,
            "agent_context": agent_context.copy() if agent_context is not None else None,
            "sampling_params": dict(request["sampling_params"]),
        }
        generation_span = start_generation_span(self._trace_identity)
        owner_request: InflightRequest | None = None
        request_fingerprint: str | None = None
        try:
            async with self.request_lock:
                self._require_active()
            request_fingerprint = (
                self._compute_request_fingerprint(request) if self._coalesce_reserved_exact_requests else None
            )
            # Register ownership before the first tree-lock await, including new chains.
            joined_request = (
                self._inflight_exact_requests.get(request_fingerprint) if request_fingerprint is not None else None
            )
            if joined_request is not None:
                self._record_coalesced_request(joined_request, request_fingerprint)
                succeeded, result, _, _ = await asyncio.shield(joined_request.future)
                if not succeeded:
                    raise result
            else:
                owner_request = InflightRequest(asyncio.get_running_loop().create_future())
                if request_fingerprint is not None:
                    self._inflight_exact_requests[request_fingerprint] = owner_request
                result = await self._run_generation(request, backend, owner_request)
                if request_fingerprint is not None:
                    self._resolve_inflight_exact_request(
                        request_fingerprint, owner_request, (True, result, result[4], result[5])
                    )
            assistant_msg, finish_reason, prompt_tokens, completion_tokens, chain_id, turn, capacity_exhausted = result
            self._touch()
            if capacity_exhausted:
                generation_span.capacity_exhausted(prompt_tokens=prompt_tokens, chain_id=chain_id)
            else:
                generation_span.success(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    chain_id=chain_id,
                    turn=turn,
                    assistant_msg=assistant_msg,
                    finish_reason=finish_reason,
                )
            return GenerationOutcome(
                assistant_msg=assistant_msg,
                finish_reason=finish_reason,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )
        except BaseException as exc:
            if owner_request is not None and request_fingerprint is not None:
                self._resolve_inflight_exact_request(request_fingerprint, owner_request, (False, exc, None, None))
            generation_span.failure(exc)
            raise
        finally:
            generation_span.report()

    async def finalize(self) -> list[Trajectory]:
        async with self.request_lock:
            if self.phase == SessionPhase.ABORTED:
                raise RuntimeError(f"Session {self.handle.session_id} is aborted")
            if self.phase == SessionPhase.FINALIZED:
                raise RuntimeError(f"Session {self.handle.session_id} is finalized")
            async with self._tree_lock:
                trajectories = [
                    leaf if isinstance(leaf, Trajectory) else self._linearize_path(self._path_from_root(leaf))
                    for leaf in self.leaves.values()
                ]
                if self._storage_path is not None:
                    try:
                        await self._write_storage_state_locked(self._storage_path)
                    except Exception:
                        logger.warning(
                            "Failed to persist trajectory tree %s",
                            self._storage_path,
                            exc_info=True,
                        )
                self._num_trajectories = len(trajectories)
                self._clear_tree()
                self.phase = SessionPhase.FINALIZED
            if self._coalesced_request_count or self._rollback_count:
                logger.info(
                    "Session finalized with session=%s trajectories=%s num_coalesced_requests=%s "
                    "rollback_count=%s rollback_dropped_trainable_tokens_total=%s",
                    self.handle.session_id,
                    len(trajectories),
                    self._coalesced_request_count,
                    self._rollback_count,
                    self._rollback_dropped_trainable_tokens_total,
                )
            self._touch()
            return trajectories

    async def abort(self) -> None:
        async with self.request_lock:
            if self.phase == SessionPhase.ABORTED:
                return
            if self.phase == SessionPhase.FINALIZED:
                raise RuntimeError(f"Session {self.handle.session_id} is finalized")
            async with self._tree_lock:
                storage_path = self._storage_path
                try:
                    if storage_path is not None and self.root is not None:
                        storage_path = storage_path.with_name("trajectory_tree_abort.pkl")
                        await self._write_storage_state_locked(storage_path)
                        self._storage_path = storage_path
                except Exception:
                    logger.warning(
                        "Failed to persist aborted trajectory tree %s",
                        storage_path,
                        exc_info=True,
                    )
                self._num_trajectories = 0
                self._clear_tree()
                self.phase = SessionPhase.ABORTED
            self._touch()

    def snapshot_state(self) -> dict[str, Any]:
        """Read in-memory metadata without waiting for locks or performing I/O."""
        chain_ids = sorted(chain_id for chain_id, leaf in self.leaves.items() if isinstance(leaf, MessageNode))
        return {
            "session_id": self.handle.session_id,
            "phase": self.phase.value,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "num_trajectories": self._num_trajectories
            + sum(isinstance(leaf, Trajectory) for leaf in self.leaves.values()),
            "has_active_trajectory": bool(chain_ids),
            "num_active_chains": len(chain_ids),
            "active_chain_ids": chain_ids,
            "rollback_count": self._rollback_count,
            "rollback_dropped_trainable_tokens_total": self._rollback_dropped_trainable_tokens_total,
            "num_coalesced_requests": self._coalesced_request_count,
            # Cumulative node IDs allocated, including failed or removed branches.
            "num_nodes": self._next_node_id - 1,
        }

    @property
    def active_chains(self) -> list[_TreeActiveChainView]:
        """Expose committed tree tips through the linear session's read-only view."""
        return [
            _TreeActiveChainView(
                chain_id=chain_id,
                message_history=self._path_messages(self._path_from_root(leaf)),
                active_tool_schemas=self._copy_tools(self.tool_schemas_by_fingerprint.get(leaf.tools_fingerprint)),
            )
            for chain_id, leaf in sorted(self.leaves.items())
            if isinstance(leaf, MessageNode)
        ]

    @active_chains.setter
    def active_chains(self, value: list[Any]) -> None:
        # Accept GatewaySession.__init__'s assignment; the tree owns the live view.
        del value

    @property
    def storage_path(self) -> Path | None:
        return self._storage_path

    def _storage_state(self) -> dict[str, Any]:
        # Keep topology flat: even a children-only tree can exceed pickle's recursion limit.
        nodes: dict[int, MessageNode] = {}
        stack = [self.root] if self.root is not None else []
        stack.extend(leaf for leaf in self.leaves.values() if isinstance(leaf, MessageNode))
        while stack:
            node = stack.pop()
            if id(node) in nodes:
                continue
            nodes[id(node)] = node
            stack.extend(node.children)
        copies = {key: replace(node, parent=None, children=[]) for key, node in nodes.items()}
        return {
            "root": copies[id(self.root)] if self.root is not None else None,
            "node_children": [
                (copies[key], [copies[id(child)] for child in node.children]) for key, node in nodes.items()
            ],
            "tool_schemas_by_fingerprint": self.tool_schemas_by_fingerprint,
            "chains": {
                chain_id: copies[id(leaf)] if isinstance(leaf, MessageNode) else leaf
                for chain_id, leaf in self.leaves.items()
            },
            "routed_experts_by_chain": self._routed_experts_by_chain,
            "rewrite_merge_count": self._rollback_count,
            "rewrite_dropped_trainable_tokens": self._rollback_dropped_trainable_tokens_total,
            "coalesced_request_count": self._coalesced_request_count,
        }

    @staticmethod
    def _write_storage_state(path: Path, state: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_suffix(path.suffix + ".tmp")
        with temporary_path.open("wb") as output_file:
            pickle.dump(state, output_file, protocol=pickle.HIGHEST_PROTOCOL)
            output_file.flush()
            os.fsync(output_file.fileno())
        os.replace(temporary_path, path)

    def _record_coalesced_request(self, joined_request: InflightRequest, request_fingerprint: str) -> None:
        joined_request.waiter_count += 1
        self._coalesced_request_count += 1
        if self._coalesced_request_count == 1:
            logger.warning(
                "Exact in-flight request coalesced: session=%s fingerprint=%s role=waiter. "
                "The waiter shares the owner's backend generation and trajectory result. "
                "Set coalesce_reserved_exact_requests=false to preserve independent "
                "same-session sampling.",
                self.handle.session_id,
                request_fingerprint[:12],
            )

    async def _run_generation(
        self,
        request: InternalGenerationRequest,
        backend,
        owner_request: InflightRequest,
    ) -> tuple[dict[str, Any], str, int, int, int | None, int, bool]:
        """Run one concurrent tree transaction around an unlocked backend call."""
        messages, tools = request["messages"], request["tools"]
        agent_context = request.get("agent_context")
        async with self._tree_lock:
            self._require_active()
            mounted = await self._mount_generation(
                messages=messages,
                tools=tools,
                agent_context=agent_context,
            )
            node = mounted.target_node
            prompt_tokens = len(mounted.context_ids)
            remaining_capacity = (
                self._trajectory_capacity - prompt_tokens if self._trajectory_capacity is not None else None
            )
            if remaining_capacity is not None and remaining_capacity <= 0:
                try:
                    prompt_tokens = self._close_length_exhausted_chain(mounted)
                finally:
                    self._rollback_mounted_branch(target_node=node, merge_backup=mounted.merge_backup)
                logger.warning(
                    "Trajectory capacity prevents generation: session=%s. %s",
                    self.handle.session_id,
                    _TRAJECTORY_CAPACITY_HINT,
                )
                return (
                    {"role": "assistant", "content": ""},
                    "length",
                    prompt_tokens,
                    0,
                    node.chain_id or None,
                    0,
                    True,
                )

            effective_sampling_params = dict(request["sampling_params"])
            if remaining_capacity is not None:
                effective_sampling_params["max_tokens"] = min(
                    effective_sampling_params.get("max_tokens", remaining_capacity), remaining_capacity
                )

        try:
            try:
                backend_kwargs = {}
                if agent_context is not None and getattr(backend, "supports_agent_context", False) is True:
                    backend_kwargs["agent_context"] = agent_context.copy()
                output = await backend.generate(
                    request_id=self.handle.session_id,
                    prompt_ids=mounted.context_ids,
                    sampling_params=effective_sampling_params,
                    image_data=mounted.image_data,
                    video_data=mounted.video_data,
                    mm_processor_kwargs=self._codec.mm_processor_kwargs or {},
                    **backend_kwargs,
                )
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            except Exception as exc:
                raise HTTPException(status_code=500, detail=f"{exc.__class__.__name__}: {exc}") from exc
            response_ids = list(output.token_ids)
            response_logprobs: list[float] | None = None
            if effective_sampling_params.get("logprobs", False):
                if output.log_probs is None:
                    raise RuntimeError("backend omitted logprobs when requested")
                response_logprobs = list(output.log_probs)
                if len(response_logprobs) != len(response_ids):
                    raise RuntimeError(
                        "backend logprobs must align with token_ids: "
                        f"got {len(response_logprobs)} logprobs for {len(response_ids)} tokens"
                    )
            async with self._tree_lock:
                self._require_active()
                assistant_message, finish_reason = await self._codec.decode_response(
                    response_ids, tools=tools, stop_reason=output.stop_reason
                )
                extra_fields = output.extra_fields or {}
                merge_result = self._codec.merge_assistant_tokens(mounted.context_ids, response_ids)
                if response_logprobs is not None and len(response_logprobs) != merge_result.appended_token_count:
                    raise ValueError(
                        "assistant_logprobs length must match appended assistant token count, "
                        f"got {len(response_logprobs)} and {merge_result.appended_token_count}"
                    )
                if merge_result.removed_prefix_token_count > len(node.prompt_ids):
                    raise RuntimeError("Assistant merge removes tokens from shared ancestors")
                if merge_result.removed_prefix_token_count or merge_result.inserted_token_ids:
                    node.pre_assistant_prompt_ids = list(node.prompt_ids)
                if merge_result.removed_prefix_token_count:
                    del node.prompt_ids[-merge_result.removed_prefix_token_count :]
                node.prompt_ids.extend(merge_result.inserted_token_ids)
                node.messages.append(dict(assistant_message))
                node.response_ids = response_ids
                node.response_logprobs = response_logprobs
                parent = node.parent
                node.message_turn_count = (parent.message_turn_count if parent is not None else 0) + sum(
                    message.get("role") in ("user", "assistant") for message in node.messages
                )
                node.min_global_steps = extra_fields.get("min_global_steps")
                node.max_global_steps = extra_fields.get("max_global_steps")
                routed_experts = getattr(output, "routed_experts", None)
                node.status = "COMPLETED"
                previous_leaf = self.leaves.get(node.chain_id)
                if previous_leaf is not None:
                    assert isinstance(previous_leaf, MessageNode), "Cannot commit to a closed chain"
                    node.num_coalesced_requests = previous_leaf.num_coalesced_requests
                    node.rollback_count = previous_leaf.rollback_count
                    node.rollback_dropped_trainable_tokens_total = previous_leaf.rollback_dropped_trainable_tokens_total
                node.num_coalesced_requests += owner_request.waiter_count
                if mounted.merge_backup is not None:
                    self._rollback_count += 1
                    self._rollback_dropped_trainable_tokens_total += len(mounted.merge_backup.response_ids)
                    node.rollback_count += 1
                    node.rollback_dropped_trainable_tokens_total += len(mounted.merge_backup.response_ids)
                    mounted.merge_backup.image_data = mounted.merge_backup.video_data = None
                if node.chain_id == 0:
                    node.chain_id = self._next_chain_id
                    self._next_chain_id += 1
                chain_id = node.chain_id
                # Registry order tracks the latest commit or capacity materialization.
                self.leaves.pop(chain_id, None)
                self.leaves[chain_id] = node
                if routed_experts is not None:
                    self._routed_experts_by_chain[chain_id] = routed_experts
        except BaseException as error:
            cancellation: asyncio.CancelledError | None = None
            while True:
                try:
                    async with self._tree_lock:
                        # Rollback must stay synchronous: only waiting for the lock may be cancelled.
                        self._rollback_mounted_branch(target_node=node, merge_backup=mounted.merge_backup)
                    break
                except asyncio.CancelledError as exc:
                    cancellation = exc
            if cancellation is not None:
                raise cancellation from error
            raise
        if finish_reason == "length" and remaining_capacity is not None and len(response_ids) >= remaining_capacity:
            logger.warning(
                "Generation reached trajectory capacity: session=%s context_tokens=%s completion_tokens=%s. %s",
                self.handle.session_id,
                prompt_tokens,
                len(response_ids),
                _TRAJECTORY_CAPACITY_HINT,
            )
        return (
            assistant_message,
            finish_reason,
            prompt_tokens,
            len(response_ids),
            node.chain_id,
            node.message_turn_count + 1,
            False,
        )

    async def _mount_generation(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        agent_context: AgentContext | None = None,
    ) -> _MountedGeneration:
        """Match only stable nodes and mount one request-local PENDING branch."""
        merge_candidate: MessageNode | None = None
        serialized_tools = json.dumps(tools, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
        tools_fingerprint = hashlib.sha256(serialized_tools.encode("utf-8")).hexdigest()
        if tools_fingerprint not in self.tool_schemas_by_fingerprint:
            self.tool_schemas_by_fingerprint[tools_fingerprint] = self._copy_tools(tools)

        parent = self.root
        parent_path: list[MessageNode] = []
        offset = 0
        assistant_candidate: MessageNode | None = None
        active_path_ranks: dict[int, tuple[int, bool, int]] = {}
        for commit_order, leaf in enumerate(self.leaves.values()):
            if not isinstance(leaf, MessageNode):
                continue
            if leaf.agent_context != agent_context or leaf.tools_fingerprint != tools_fingerprint:
                continue
            path = self._path_from_root(leaf)
            history = self._path_messages(path)
            prefix_length = len(history) - 1
            if len(messages) <= prefix_length or not self._messages_equal(history[:-1], messages[:prefix_length]):
                continue
            exact = self._messages_equal(history[-1:], messages[prefix_length : prefix_length + 1])
            rank = (len(history) if exact else prefix_length, exact, commit_order)
            for node in path:
                active_path_ranks[id(node)] = max(active_path_ranks.get(id(node), (0, False, -1)), rank)
        while offset < len(messages) and messages[-1].get("role") != "assistant":
            incoming = messages[offset:]
            matching_full: list[MessageNode] = []
            candidate: MessageNode | None = None
            for child in parent.children:
                if child.agent_context != agent_context:
                    continue
                if child.tools_fingerprint != tools_fingerprint:
                    continue
                if child.status != "COMPLETED" or not child.messages:
                    continue
                prefix_length = len(child.messages) - 1
                if len(incoming) < prefix_length or not self._messages_equal(
                    child.messages[:-1], incoming[:prefix_length]
                ):
                    continue
                if len(incoming) == prefix_length and child.messages[-1].get("role") == "assistant":
                    continue  # Same prompt after completion starts an independent sample.
                if len(incoming) > prefix_length and self._messages_equal(
                    [child.messages[-1]], [incoming[prefix_length]]
                ):
                    matching_full.append(child)
                    continue
                if child.messages[-1].get("role") == "assistant" and (
                    candidate is None or self.leaves.get(child.chain_id) is child
                ):
                    candidate = child
            if matching_full:
                # Prefer the best matching active history, then its latest commit, including shared ancestors.
                child = max(matching_full, key=lambda node: active_path_ranks.get(id(node), (0, False, -1)))
                parent = child
                parent_path.append(child)
                offset += len(child.messages)
                continue
            assistant_candidate = candidate
            break

        pending_messages = messages[offset:]
        if assistant_candidate is not None:
            sibling_assistant_nodes = [
                child
                for child in parent.children
                if child.status == "COMPLETED"
                and self.leaves.get(child.chain_id) is child
                and child.agent_context == agent_context
                and child.tools_fingerprint == tools_fingerprint
                and child.messages
                and child.messages[-1].get("role") == "assistant"
                and len(child.messages) == len(assistant_candidate.messages)
                and self._messages_equal(child.messages[:-1], assistant_candidate.messages[:-1])
            ]
            if (
                self._enable_last_assistant_rollback
                and self.leaves.get(assistant_candidate.chain_id) is assistant_candidate
                and len(sibling_assistant_nodes) == 1
                and not assistant_candidate.children
                and not any(
                    child.status == "PENDING" and child.agent_context == agent_context for child in parent.children
                )
            ):
                merge_candidate = assistant_candidate

        context_ids: list[int] = []
        for node in parent_path:
            if node.removed_prefix_token_count:
                del context_ids[-node.removed_prefix_token_count :]
            context_ids.extend(node.prompt_ids)
            context_ids.extend(node.response_ids)
        merge_result = None
        image_data = [image for node in parent_path for image in node.image_data or []] or None
        video_data = [video for node in parent_path for video in node.video_data or []] or None
        if not parent.is_root:
            self._codec.validate_incremental_media(pending_messages)
        new_image_data, new_video_data = await self._codec.extract_multi_modal_data(pending_messages)
        if not parent.is_root and new_video_data:
            raise UnsupportedMediaError(
                "Continuous Token context merging does not currently support incremental video data"
            )
        if new_image_data:
            image_data = [*(image_data or []), *new_image_data]
        if new_video_data:
            video_data = [*(video_data or []), *new_video_data]
        if parent.is_root:
            prompt_ids = self._codec.build_initial_tokens(
                pending_messages, tools=tools, image_data=image_data, video_data=video_data
            )
            context_ids = prompt_ids
        else:
            previous_messages = self._path_messages(parent_path)
            merge_result = self._codec.merge_context_tokens(
                previous_messages,
                previous_messages + pending_messages,
                context_ids,
                tools=tools,
                image_data=image_data,
            )
            prompt_ids = merge_result.token_ids[len(context_ids) - merge_result.removed_prefix_token_count :]
            context_ids = merge_result.token_ids
        chain_id = 0
        if merge_candidate is not None:
            chain_id = merge_candidate.chain_id
        elif not parent.is_root and self.leaves.get(parent.chain_id) is parent and not parent.children:
            chain_id = parent.chain_id
        node_id = merge_candidate.node_id if merge_candidate is not None else self._next_node_id
        if merge_candidate is None:
            self._next_node_id += 1
        target_node = MessageNode(
            node_id=node_id,
            messages=pending_messages,
            parent=parent,
            prompt_ids=prompt_ids,
            removed_prefix_token_count=merge_result.removed_prefix_token_count if merge_result else 0,
            image_data=list(new_image_data) if new_image_data else None,
            video_data=list(new_video_data) if new_video_data else None,
            tools_fingerprint=tools_fingerprint,
            agent_context=agent_context.copy() if agent_context is not None else None,
            chain_id=chain_id,
        )
        try:
            parent.children.append(target_node)
            if merge_candidate is not None:
                parent.children.remove(merge_candidate)
        except BaseException:
            self._rollback_mounted_branch(target_node=target_node, merge_backup=merge_candidate)
            raise
        return _MountedGeneration(
            context_ids=context_ids,
            image_data=image_data,
            video_data=video_data,
            target_node=target_node,
            merge_backup=merge_candidate,
        )

    def _rollback_mounted_branch(
        self,
        target_node: MessageNode | None,
        merge_backup: MessageNode | None,
    ) -> None:
        if target_node is not None:
            target_node.image_data = target_node.video_data = None
        if self.root is None:
            return
        if target_node is not None and target_node.parent is not None:
            parent = target_node.parent
            parent.children = [child for child in parent.children if child is not target_node]
        if merge_backup is not None:
            parent = merge_backup.parent
            if not any(child is merge_backup for child in parent.children):
                parent.children.append(merge_backup)

    def _close_length_exhausted_chain(self, mounted: _MountedGeneration) -> int:
        """Freeze the committed chain, applying an accepted rollback before closing."""
        chain_id = mounted.target_node.chain_id
        leaf = self.leaves.get(chain_id)
        if not isinstance(leaf, MessageNode):
            return len(mounted.context_ids)
        path = self._path_from_root(leaf)
        backup = mounted.merge_backup
        if backup is not None:
            # Keep this turn's original input/media, but omit its generated assistant.
            path[-1] = replace(
                backup,
                prompt_ids=(
                    backup.pre_assistant_prompt_ids
                    if backup.pre_assistant_prompt_ids is not None
                    else backup.prompt_ids
                ),
                messages=backup.messages[:-1],
                response_ids=[],
                response_logprobs=[] if backup.response_logprobs is not None else None,
                min_global_steps=None,
                max_global_steps=None,
                message_turn_count=backup.message_turn_count - 1,
                rollback_count=backup.rollback_count + 1,
                rollback_dropped_trainable_tokens_total=(
                    backup.rollback_dropped_trainable_tokens_total + len(backup.response_ids)
                ),
            )
        trajectory = self._linearize_path(path)
        drop_chain = False
        if backup is not None:
            # Match Chain's pre-assistant response-length check, not sum(mask).
            drop_chain = not trajectory.response_ids
            stored_ids = trajectory.prompt_ids if drop_chain else trajectory.response_ids
            assistant_prefix = self._codec.turn_separator + self._codec.generation_prompt
            if assistant_prefix:
                if stored_ids[-len(assistant_prefix) :] != assistant_prefix:
                    raise ValueError("Stored trajectory does not end with the assistant prefix")
                del stored_ids[-len(assistant_prefix) :]
                if not drop_chain:
                    del trajectory.response_mask[-len(assistant_prefix) :]
                    if trajectory.response_logprobs is not None:
                        del trajectory.response_logprobs[-len(assistant_prefix) :]
            trajectory.response_logprobs = trajectory.response_logprobs or None
        trajectory.extra_fields["materialization_reason"] = "max_trajectory_length"
        # All conversion/validation precedes the state change; this runs under the tree lock.
        self.leaves.pop(chain_id)
        if not drop_chain:
            self.leaves[chain_id] = trajectory
        self._routed_experts_by_chain.pop(chain_id, None)
        if backup is not None:
            self._rollback_count += 1
            self._rollback_dropped_trainable_tokens_total += len(backup.response_ids)
        return len(trajectory.prompt_ids) + len(trajectory.response_ids)

    def _clear_tree(self) -> None:
        """Release the in-memory tree and break MessageNode reference cycles."""
        stack = [leaf for leaf in self.leaves.values() if isinstance(leaf, MessageNode)]
        if self.root is not None:
            stack.append(self.root)
        self.root = None
        self.leaves.clear()
        self.tool_schemas_by_fingerprint.clear()
        self._routed_experts_by_chain.clear()
        while stack:
            node = stack.pop()
            stack.extend(node.children)
            node.parent = None
            node.children.clear()
            node.image_data = node.video_data = None

    async def _write_storage_state_locked(self, storage_path: Path) -> None:
        """Keep the tree locked until its writer exits, even after cancellation."""
        task = asyncio.create_task(asyncio.to_thread(self._write_storage_state, storage_path, self._storage_state()))
        cancellation: asyncio.CancelledError | None = None
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError as exc:
                cancellation = exc
            except Exception:
                if cancellation is None:
                    raise
        if cancellation is not None:
            if not task.cancelled():
                task.exception()  # Retrieve any write failure before propagating cancellation.
            raise cancellation
        task.result()

    def _messages_equal(
        self,
        left: list[dict[str, Any]],
        right: list[dict[str, Any]],
    ) -> bool:
        return len(left) == len(right) and all(
            self._codec.canonicalize_message_for_prefix_comparison(left_message)
            == self._codec.canonicalize_message_for_prefix_comparison(right_message)
            for left_message, right_message in zip(left, right, strict=True)
        )

    def _linearize_path(self, path: list[MessageNode]) -> Trajectory:
        leaf = path[-1]
        chain_start = next(index for index, node in enumerate(path) if node.chain_id == leaf.chain_id)
        prompt_ids = list(path[0].prompt_ids)
        response_ids: list[int] = []
        response_mask: list[int] = []
        response_logprobs: list[float] | None = [] if leaf.response_logprobs is not None else None
        for index, node in enumerate(path):
            if node.removed_prefix_token_count:
                if node.removed_prefix_token_count > len(response_ids):
                    del prompt_ids[-(node.removed_prefix_token_count - len(response_ids)) :]
                del response_ids[-node.removed_prefix_token_count :]
                del response_mask[-node.removed_prefix_token_count :]
                if response_logprobs is not None:
                    del response_logprobs[-node.removed_prefix_token_count :]
            if index:
                response_ids.extend(node.prompt_ids)
                response_mask.extend(repeat(0, len(node.prompt_ids)))
                if response_logprobs is not None:
                    response_logprobs.extend(repeat(0.0, len(node.prompt_ids)))
            if index == chain_start:
                # History owned by another chain is this chain's initial prompt.
                prompt_ids.extend(response_ids)
                response_ids.clear()
                response_mask.clear()
                if response_logprobs is not None:
                    response_logprobs.clear()
            response_ids.extend(node.response_ids)
            response_mask.extend(repeat(1, len(node.response_ids)))
            if response_logprobs is not None:
                response_logprobs.extend(
                    node.response_logprobs
                    if node.response_logprobs is not None
                    else repeat(0.0, len(node.response_ids))
                )
        if len(response_ids) != len(response_mask):
            raise RuntimeError("linearized response_ids and response_mask are not aligned")
        if response_logprobs is not None and len(response_logprobs) != len(response_ids):
            raise RuntimeError("linearized response_ids and response_logprobs are not aligned")
        marks = [
            (node.min_global_steps, node.max_global_steps)
            for node in path[chain_start:]
            if node.min_global_steps is not None
        ]
        extra_fields: dict[str, Any] = {}
        if marks:
            extra_fields["min_global_steps"] = min(mark[0] for mark in marks)
            extra_fields["max_global_steps"] = max(mark[1] for mark in marks)
        if self._mm_processor_kwargs:
            extra_fields["mm_processor_kwargs"] = dict(self._mm_processor_kwargs)
        image_data = [image for node in path for image in node.image_data or []]
        video_data = [video for node in path for video in node.video_data or []]
        multi_modal_data: dict[str, Any] = {}
        if image_data:
            multi_modal_data["images"] = image_data
        if video_data:
            multi_modal_data["videos"] = video_data
        if leaf.agent_context is not None:
            extra_fields["agent_context"] = leaf.agent_context.copy()
        if leaf.num_coalesced_requests:
            extra_fields["num_coalesced_requests"] = leaf.num_coalesced_requests
        if leaf.rollback_count:
            extra_fields["rollback_count"] = leaf.rollback_count
            extra_fields["rollback_dropped_trainable_tokens_total"] = leaf.rollback_dropped_trainable_tokens_total
        return Trajectory(
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            response_mask=response_mask,
            response_logprobs=response_logprobs or None,
            num_turns=leaf.message_turn_count + 1,
            chain_id=leaf.chain_id,
            routed_experts=self._routed_experts_by_chain.get(leaf.chain_id),
            multi_modal_data=multi_modal_data or None,
            extra_fields=extra_fields,
        )

    def _path_from_root(self, node: MessageNode) -> list[MessageNode]:
        path: list[MessageNode] = []
        current: MessageNode | None = node
        while current is not None and not current.is_root:
            path.append(current)
            current = current.parent
        path.reverse()
        return path

    def _path_messages(self, path: list[MessageNode]) -> list[dict[str, Any]]:
        return [dict(message) for node in path for message in node.messages]

    def _copy_tools(self, tools: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
        return [dict(tool) for tool in tools] if tools is not None else None

    def _require_active(self) -> None:
        if self.phase != SessionPhase.ACTIVE:
            raise HTTPException(
                status_code=409,
                detail=f"Session {self.handle.session_id} is {self.phase.value.lower()}",
            )
