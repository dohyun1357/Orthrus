# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Optional, Set
import torch
import torch.nn as nn
from vllm.model_executor.pooling_metadata import PoolingMetadata, PoolingTensors
from vllm.model_executor.layers.pooler import PoolerHead, PoolingType
from vllm.sequence import PoolerOutput, PoolingSequenceGroupOutput
from vllm.model_executor.layers.segment_mean_triton import segment_sum
from typing_extensions import assert_never

class HybridPooler(nn.Module):
    """Chunk-aware pooling, with internal state per request_id."""

    @staticmethod
    def from_pooling_type(
        pooling_type: PoolingType,
        *,
        normalize: bool,
        softmax: bool,
    ) -> "HybridPooler":
        if pooling_type == PoolingType.MEAN:
            # Prefer Triton-accelerated mean pooler
            # return HybridMeanPooler(normalize, softmax)
            return HybridMeanPoolerTriton(normalize, softmax)
        if pooling_type == PoolingType.LAST:
            return HybridLastPooler(normalize, softmax)
        if pooling_type == PoolingType.CLS:
            return HybridCLSPooler(normalize, softmax)
        if pooling_type == PoolingType.ALL:
            return HybridAllPooler(normalize, softmax)
        assert_never(pooling_type)

    def __init__(self, normalize: bool, softmax: bool):
        super().__init__()
        self.head = PoolerHead(normalize=normalize, softmax=softmax)

    def _split_sequences(
        self, hidden_states: torch.Tensor, prompt_lens: torch.Tensor
    ) -> List[torch.Tensor]:
        # hidden_states: [total_tokens, D], prompt_lens: [B]
        offsets = torch.cat([
            torch.tensor([0], device=hidden_states.device),
            torch.cumsum(prompt_lens, dim=0)[:-1]
        ])
        return [
            hidden_states[offsets[i] : offsets[i] + prompt_lens[i].item()]
            for i in range(prompt_lens.shape[0])
        ]

    def get_prompt_lens(
        self,
        hidden_states: torch.Tensor,
        pooling_metadata: PoolingMetadata,
    ) -> torch.Tensor:
        return PoolingTensors.from_pooling_metadata(
            pooling_metadata, hidden_states.device).prompt_lens

    def forward(
        self,
        hidden_states: torch.Tensor,
        pooling_metadata: PoolingMetadata,
        request_ids: List[str],
        is_last_chunk: bool = False,
        **_ignore,
    ) -> PoolerOutput:
        raise NotImplementedError("Implemented in subclasses.")


# MeanPooler where hidden states are accumulated. Safe, easy implementation that uses alot of memory.
# class HybridMeanPooler(HybridPooler):
#     def __init__(self, normalize: bool, softmax: bool):
#         super().__init__(normalize, softmax)
#         self._running_hidden_states: Dict[str, list[torch.Tensor]] = {}

#     def forward(
#         self,
#         hidden_states: torch.Tensor,
#         pooling_metadata: PoolingMetadata,
#         request_ids: List[str],
#         finished_requests_ids: Set[str],
#         **_ignore,
#     ) -> PoolerOutput:
#         # 1. Accumulate chunked sequences per request ID
#         prompt_lens = self.get_prompt_lens(hidden_states, pooling_metadata)
#         seqs = self._split_sequences(hidden_states, prompt_lens)
#         for rid, seq in zip(request_ids, seqs):
#             self._running_hidden_states.setdefault(rid, []).append(seq)

#         # 2. Collect and concatenate finished request sequences
#         finished_rids = [rid for rid in request_ids if rid in finished_requests_ids]
#         full_sequences = {}
#         for rid in finished_rids:
#             full = torch.cat(self._running_hidden_states.pop(rid), dim=0)
#             full_sequences[rid] = full

#         if not full_sequences:
#             return PoolerOutput(outputs=[
#                 PoolingSequenceGroupOutput(data=torch.empty(0, 0))
#                 for _ in request_ids
#             ])

#         # 3. Apply mean pooling to each full sequence
#         pooled_by_rid = {}
#         for rid, full in full_sequences.items():
#             pooled_by_rid[rid] = full.mean(dim=0)

#         # 4. Apply pooler head to stacked pooled outputs
#         pooled_tensor = torch.stack(list(pooled_by_rid.values()), dim=0)
#         pooled_tensor = self.head(pooled_tensor, pooling_metadata)

#         # 5. Remap output tensors back to request order
#         final_outputs = []
#         pooled_iter = iter(pooled_tensor)
#         for rid in request_ids:
#             if rid in pooled_by_rid:
#                 final_outputs.append(PoolingSequenceGroupOutput(data=next(pooled_iter)))
#             else:
#                 final_outputs.append(PoolingSequenceGroupOutput(data=torch.empty(0, 0)))

#         return PoolerOutput(outputs=final_outputs)


# MeanPooler where we keep a running sum. Streaming like implementation. WIP
class HybridMeanPooler(HybridPooler):
    def __init__(self, normalize: bool, softmax: bool):
        super().__init__(normalize, softmax)
        self._running_sums: Dict[str, torch.Tensor] = {}
        self._running_token_counts: Dict[str, int] = {}

    def forward(
        self,
        hidden_states: torch.Tensor,
        pooling_metadata: PoolingMetadata,
        request_ids: List[str],
        finished_requests_ids: Set[str],
        **_ignore,
    ) -> PoolerOutput:
        prompt_lens = self.get_prompt_lens(hidden_states, pooling_metadata)
        seqs = self._split_sequences(hidden_states, prompt_lens)

        for rid, seq in zip(request_ids, seqs):
            # seq shape: [tokens_in_chunk, hidden_size]
            chunk_sum = seq.sum(dim=0)  # [hidden_size]
            if rid in self._running_sums:
                self._running_sums[rid] = self._running_sums[rid] + chunk_sum
                self._running_token_counts[rid] += seq.size(0)
            else:
                self._running_sums[rid] = chunk_sum
                self._running_token_counts[rid] = seq.size(0)

        finished_rids = [rid for rid in request_ids if rid in finished_requests_ids]
        if not finished_rids:
            return PoolerOutput(outputs=[
                PoolingSequenceGroupOutput(data=torch.empty(0, 0))
                for _ in request_ids
            ])

        pooled_list: List[torch.Tensor] = []
        pooled_by_rid: Dict[str, torch.Tensor] = {}

        for rid in finished_rids:
            total_sum = self._running_sums.pop(rid)
            total_count = self._running_token_counts.pop(rid)
            mean_pooled = total_sum / total_count
            pooled_by_rid[rid] = mean_pooled
            pooled_list.append(mean_pooled)

        pooled_tensor = torch.stack(pooled_list, dim=0)
        pooled_tensor = self.head(pooled_tensor, pooling_metadata)

        final_outputs: List[PoolingSequenceGroupOutput] = []
        pooled_iter = iter(pooled_tensor)
        for rid in request_ids:
            if rid in pooled_by_rid:
                final_outputs.append(PoolingSequenceGroupOutput(data=next(pooled_iter)))
            else:
                final_outputs.append(PoolingSequenceGroupOutput(data=torch.empty(0, 0)))

        return PoolerOutput(outputs=final_outputs)


class HybridMeanPoolerTriton(HybridPooler):
    """Streaming mean pooler using Triton segment-sum for each chunk.

    Maintains running sums and counts per request id across chunks and emits
    pooled vectors for those that are finished in the current step.
    """
    def __init__(self, normalize: bool, softmax: bool):
        super().__init__(normalize, softmax)
        self._running_sums: Dict[str, torch.Tensor] = {}
        self._running_token_counts: Dict[str, int] = {}

    def forward(
        self,
        hidden_states: torch.Tensor,
        pooling_metadata: PoolingMetadata,
        request_ids: List[str],
        finished_requests_ids: Set[str],
        **_ignore,
    ) -> PoolerOutput:
        # Compute per-sequence sums for this chunk with Triton.
        prompt_lens = self.get_prompt_lens(hidden_states, pooling_metadata)
        # prompt_lens shape [B], on device
        chunk_sums = segment_sum(hidden_states, prompt_lens)

        # Update running sums and counts
        for i, rid in enumerate(request_ids):
            tok_cnt = int(prompt_lens[i].item())
            if tok_cnt == 0:
                continue
            if rid in self._running_sums:
                self._running_sums[rid] = self._running_sums[rid] + chunk_sums[i]
                self._running_token_counts[rid] += tok_cnt
            else:
                self._running_sums[rid] = chunk_sums[i]
                self._running_token_counts[rid] = tok_cnt

        # If nothing finished, return placeholders
        finished_rids = [rid for rid in request_ids if rid in finished_requests_ids]
        if not finished_rids:
            return PoolerOutput(outputs=[
                PoolingSequenceGroupOutput(data=torch.empty(0, 0))
                for _ in request_ids
            ])

        # Compute final means for finished requests
        pooled_list: List[torch.Tensor] = []
        pooled_by_rid: Dict[str, torch.Tensor] = {}
        for rid in finished_rids:
            total_sum = self._running_sums.pop(rid)
            total_count = self._running_token_counts.pop(rid)
            mean_pooled = total_sum / max(1, total_count)
            pooled_by_rid[rid] = mean_pooled
            pooled_list.append(mean_pooled)

        # Apply head (normalize/softmax if configured)
        pooled_tensor = torch.stack(pooled_list, dim=0)
        pooled_tensor = self.head(pooled_tensor, pooling_metadata)

        # Reorder to match request_ids, fill placeholders for unfinished
        final_outputs: List[PoolingSequenceGroupOutput] = []
        proc_idx = 0
        for rid in request_ids:
            if rid in pooled_by_rid:
                final_outputs.append(PoolingSequenceGroupOutput(data=pooled_tensor[proc_idx]))
                proc_idx += 1
            else:
                final_outputs.append(PoolingSequenceGroupOutput(data=torch.empty(0, 0)))

        return PoolerOutput(outputs=final_outputs)



class HybridLastPooler(HybridPooler):
    def __init__(self, normalize: bool, softmax: bool):
        super().__init__(normalize, softmax)
        self._last_token: Dict[str, torch.Tensor] = {}

    def forward(
        self,
        hidden_states: torch.Tensor,
        pooling_metadata: PoolingMetadata,
        request_ids: List[str],
        finished_requests_ids: Set[str],
        **_ignore,
    ) -> PoolerOutput:
        prompt_lens = self.get_prompt_lens(hidden_states, pooling_metadata)
        seqs = self._split_sequences(hidden_states, prompt_lens)
        for rid, seq in zip(request_ids, seqs):
            self._last_token[rid] = seq[-1]

        finished_indices: List[int] = []
        finished_toks: List[torch.Tensor] = []
        for i, rid in enumerate(request_ids):
            if rid in finished_requests_ids and rid in self._last_token:
                finished_indices.append(i)
                finished_toks.append(self._last_token[rid])

        processed: Optional[torch.Tensor] = None
        if finished_toks:
            stacked = torch.stack(finished_toks, dim=0)
            processed = self.head(stacked, pooling_metadata)
            for rid in finished_requests_ids:
                self._last_token.pop(rid, None)

        outputs: List[PoolingSequenceGroupOutput] = []
        proc_idx = 0
        for i, rid in enumerate(request_ids):
            if rid in finished_requests_ids and processed is not None:
                outputs.append(PoolingSequenceGroupOutput(data=processed[proc_idx]))
                proc_idx += 1
            else:
                outputs.append(PoolingSequenceGroupOutput(data=self._last_token[rid]))

        return PoolerOutput(outputs=outputs)



class HybridCLSPooler(HybridPooler):
    def __init__(self, normalize: bool, softmax: bool):
        super().__init__(normalize, softmax)
        # remember the very first token of the first chunk
        self._cls_token: dict[str, torch.Tensor] = {}

    def forward(
        self,
        hidden_states: torch.Tensor,
        pooling_metadata: PoolingMetadata,
        request_ids: List[str],
        is_last_chunk: bool = False,
        **_ignore,
    ) -> PoolerOutput:
        prompt_lens = self.get_prompt_lens(hidden_states, pooling_metadata)
        seqs = self._split_sequences(hidden_states, prompt_lens)

        # on first appearance, stash seq[0]
        for rid, seq in zip(request_ids, seqs):
            if rid not in self._cls_token:
                self._cls_token[rid] = seq[0]

        # always return the same CLS
        rows = [ self._cls_token[rid] for rid in request_ids ]

        if is_last_chunk:
            # clean up
            for rid in request_ids:
                self._cls_token.pop(rid, None)

        return PoolerOutput([
            PoolingSequenceGroupOutput(data=row) for row in rows
        ])


class HybridAllPooler(HybridPooler):
    def __init__(self, normalize: bool, softmax: bool):
        super().__init__(normalize, softmax)
        # keep every chunk for each request_id
        self._all_chunks: dict[str, List[torch.Tensor]] = {}

    def forward(
        self,
        hidden_states: torch.Tensor,
        pooling_metadata: PoolingMetadata,
        request_ids: List[str],
        is_last_chunk: bool = False,
        **_ignore,
    ) -> PoolerOutput:
        prompt_lens = self.get_prompt_lens(hidden_states, pooling_metadata)
        seqs = self._split_sequences(hidden_states, prompt_lens)

        # accumulate
        for rid, seq in zip(request_ids, seqs):
            self._all_chunks.setdefault(rid, []).append(seq)

        # intermediate: flatten just the chunk you got
        if not is_last_chunk:
            return PoolerOutput([
                PoolingSequenceGroupOutput(data=seq.flatten())
                for seq in seqs
            ])

        # final: concat all chunks for each request, flatten
        packed = []
        for rid in request_ids:
            full = torch.cat(self._all_chunks[rid], dim=0)
            packed.append(full.flatten())
            del self._all_chunks[rid]

        # NOTE: stacking flattened tensors of variable length is not possible.
        # This implementation of HybridAllPooler may require further revision
        # depending on the desired final output format.
        # Assuming for now the consumer can handle a list of tensors.
        return PoolerOutput([
            PoolingSequenceGroupOutput(data=row) for row in packed
        ])
