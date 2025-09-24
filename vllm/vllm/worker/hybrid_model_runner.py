# SPDX-License-Identifier: Apache-2.0

import dataclasses
from dataclasses import replace
from typing import Any, Dict, List, Optional, Set, Tuple, Type, Union

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.attention.backends.abstract import AttentionBackend
from vllm.distributed import get_pp_group
from vllm.distributed.kv_transfer import get_kv_transfer_group
from vllm.forward_context import set_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.model_executor.pooling_metadata import PoolingMetadata
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import IntermediateTensors, PoolerOutput, SequenceData, SequenceGroupMetadata

from vllm.multimodal import (MULTIMODAL_REGISTRY, BatchedTensorInputs,
                             MultiModalKwargs, MultiModalPlaceholderMap,
                             MultiModalRegistry)
from vllm.pooling_params import PoolingParams
from vllm.worker.model_runner import GPUModelRunnerBase, ModelInputForGPUBuilder, ModelInputForGPUWithSamplingMetadata, TModelInputForGPU
from vllm.worker.model_runner_base import (
    ModelRunnerInputBuilderBase,
    _add_sampling_metadata_broadcastable_dict,
    _init_attn_metadata_from_tensor_dict,
    _init_sampling_metadata_from_tensor_dict,
)

logger = init_logger(__name__)


@dataclasses.dataclass(frozen=True)
class HybridModelInputForGPU(ModelInputForGPUWithSamplingMetadata):
    """
    Extends base ModelInputForGPUWithSamplingMetadata to include pooling fields.
    """
    pooling_metadata: Optional[PoolingMetadata] = None
    ordered_pooling_request_ids: Optional[List[str]] = None
    list_of_token_indices_for_gathering_pooling_hs: Optional[List[torch.Tensor]] = None
    ordered_sampling_request_ids: Optional[List[str]] = None
    lora_row_indices: Optional[torch.Tensor] = None

    def as_broadcastable_tensor_dict(self) -> Dict[str, Any]:
        d = super().as_broadcastable_tensor_dict()
        _add_sampling_metadata_broadcastable_dict(d, self.sampling_metadata)
        # pooling metadata is driver-only, skip broadcast
        return d

    @classmethod
    def from_broadcasted_tensor_dict(
        cls,
        tensor_dict: Dict[str, Any],
        attn_backend: Optional[AttentionBackend] = None,
    ) -> "HybridModelInputForGPU":
        # first init sampling and attention
        tensor_dict = _init_sampling_metadata_from_tensor_dict(tensor_dict)
        if attn_backend:
            tensor_dict = _init_attn_metadata_from_tensor_dict(attn_backend, tensor_dict)
        return cls(**tensor_dict)

    
class HybridModelRunner(GPUModelRunnerBase[HybridModelInputForGPU]):
    _model_input_cls: Type[HybridModelInputForGPU] = HybridModelInputForGPU
    _builder_cls: Type[ModelRunnerInputBuilderBase] = ModelInputForGPUBuilder

    @torch.inference_mode()
    def execute_model(
        self,
        model_input: HybridModelInputForGPU,
        kv_caches: List[torch.Tensor],
        intermediate_tensors: Optional[IntermediateTensors] = None,
        num_steps: int = 1,
        **kwargs,
    ) -> Optional[Union[List[Union[SamplerOutput, PoolerOutput]], IntermediateTensors]]:
        """
        Matches upstream ModelRunner.execute_model API, adding hybrid pooling.
        """
        if num_steps > 1:
            raise ValueError("num_steps > 1 is not supported")

        if self.lora_config:
            assert model_input.lora_requests is not None
            assert model_input.lora_mapping is not None
            self.set_active_loras(model_input.lora_requests,
                                  model_input.lora_mapping)

        if self.prompt_adapter_config:
            assert model_input.prompt_adapter_requests is not None
            assert model_input.prompt_adapter_mapping is not None
            self.set_active_prompt_adapters(
                model_input.prompt_adapter_requests,
                model_input.prompt_adapter_mapping)

        self.attn_state.begin_forward(model_input)

        assert model_input.attn_metadata is not None
        prefill_meta = model_input.attn_metadata.prefill_metadata
        decode_meta = model_input.attn_metadata.decode_metadata
        # TODO(andoorve): We can remove this once all
        # virtual engines share the same kv cache.
        virtual_engine = model_input.virtual_engine
        previous_hidden_states = kwargs.get("previous_hidden_states")
        if prefill_meta is None and decode_meta.use_cuda_graph:
            assert model_input.input_tokens is not None
            graph_batch_size = model_input.input_tokens.shape[0]
            model_executable = self.graph_runners[virtual_engine][
                graph_batch_size]
            if previous_hidden_states is not None:
                previous_hidden_states = torch.cat([
                    previous_hidden_states,
                    torch.empty([
                        graph_batch_size - previous_hidden_states.shape[0],
                        *previous_hidden_states.shape[1:]
                    ],
                                dtype=previous_hidden_states.dtype,
                                device=previous_hidden_states.device)
                ])
        else:
            model_executable = self.model
        
        is_prefill_run = prefill_meta is not None

        bypass_model_exec = False
        if self.need_recv_kv(model_input, kv_caches):
            hidden_or_intermediate_states, bypass_model_exec, model_input = \
                get_kv_transfer_group().recv_kv_caches_and_hidden_states(
                    # model is used to know which layer the current worker
                    # is working on, so that we can receive KV for only those
                    # layers.
                    model_executable,
                    model_input,
                    kv_caches=kv_caches
                )

        multi_modal_kwargs = model_input.multi_modal_kwargs or {}
        seqlen_agnostic_kwargs = {
            "finished_requests_ids": model_input.finished_requests_ids,
            "request_ids_to_seq_ids": model_input.request_ids_to_seq_ids,
        } if self.has_inner_state else {}
        model_kwargs = {}
        if previous_hidden_states is not None:
            model_kwargs["previous_hidden_states"] = previous_hidden_states
        if (self.observability_config is not None
                and self.observability_config.collect_model_forward_time):
            model_forward_start = torch.cuda.Event(enable_timing=True)
            model_forward_end = torch.cuda.Event(enable_timing=True)
            model_forward_start.record()

        if not bypass_model_exec:
            with set_forward_context(model_input.attn_metadata,
                                     self.vllm_config, virtual_engine):
                hidden_or_intermediate_states = model_executable(
                    input_ids=model_input.input_tokens,
                    positions=model_input.input_positions,
                    intermediate_tensors=intermediate_tensors,
                    **MultiModalKwargs.as_kwargs(multi_modal_kwargs,
                                                 device=self.device),
                    **seqlen_agnostic_kwargs,
                    **model_kwargs,
                )

        if (self.observability_config is not None
                and self.observability_config.collect_model_forward_time):
            model_forward_end.record()

        if self.need_send_kv(model_input, kv_caches):
            get_kv_transfer_group().send_kv_caches_and_hidden_states(
                # model_executable is used to know which layer the current
                # worker is working on, so that we can send KV for only those
                # layers.
                model_executable,
                model_input,
                kv_caches,
                hidden_or_intermediate_states,
            )

        if not get_pp_group().is_last_rank:
            if (self.is_driver_worker
                    and hidden_or_intermediate_states is not None
                    and isinstance(hidden_or_intermediate_states,
                                   IntermediateTensors)
                    and self.observability_config is not None
                    and self.observability_config.collect_model_forward_time):
                model_forward_end.synchronize()
                model_forward_time = model_forward_start.elapsed_time(
                    model_forward_end)
                orig_model_forward_time = 0.0
                if intermediate_tensors is not None:
                    orig_model_forward_time = intermediate_tensors.tensors.get(
                        "model_forward_time", torch.tensor(0.0)).item()
                hidden_or_intermediate_states.tensors["model_forward_time"] = (
                    torch.tensor(model_forward_time + orig_model_forward_time))
            # print(f"Returning hidden_or_intermediate_states: {hidden_or_intermediate_states.shape}")
            return hidden_or_intermediate_states

        if not self.is_driver_worker:
            return []
        
        # print(f"######################### HYBRID MODEL RUNNER #########################")
        # print(f"{hidden_or_intermediate_states.shape = }")
        # print(f"######################### HYBRID MODEL RUNNER #########################")

        model_outputs_dict: Dict[str, Optional[Union[PoolerOutput, SamplerOutput]]] = {
            "pooling": None,
            "sampling": None,
        }
        # print(f"{is_prefill_run = }")

        # Pooling part
        if model_input.pooling_metadata and model_input.pooling_metadata.seq_groups:
            actual_hidden_states_full_batch = (
                hidden_or_intermediate_states.last_hidden_state
                if isinstance(hidden_or_intermediate_states, IntermediateTensors)
                else hidden_or_intermediate_states
            )

            if model_input.list_of_token_indices_for_gathering_pooling_hs:
                segments = []
                # print(f"[DEBUG] Batch Shape: {actual_hidden_states_full_batch.shape}")
                for indices_tensor in model_input.list_of_token_indices_for_gathering_pooling_hs:
                    # print(f"[DEBUG] Indices Shape: {indices_tensor.shape}")
                    # print(f"[DEBUG] Max Index: {indices_tensor.max().item() if indices_tensor.numel() > 0 else 'N/A'}")
                    # print(f"[DEBUG] Min Index: {indices_tensor.min().item() if indices_tensor.numel() > 0 else 'N/A'}")

                    # This is the condition that MUST be true
                    # if indices_tensor.numel() > 0 and indices_tensor.max().item() >= actual_hidden_states_full_batch.shape[0]:
                        # print("[CRITICAL] Out-of-bounds condition detected!")
                    segments.append(
                        actual_hidden_states_full_batch.index_select(0, indices_tensor)
                    )

                # segments = [
                #     actual_hidden_states_full_batch.index_select(0, indices_tensor)
                #     for indices_tensor in model_input.list_of_token_indices_for_gathering_pooling_hs
                # ]
                compact_hidden_states_for_pooler = torch.cat(segments, dim=0)
            else:
                compact_hidden_states_for_pooler = actual_hidden_states_full_batch

            pooler_output_obj = self.model.pooler(
                hidden_states=compact_hidden_states_for_pooler,
                pooling_metadata=model_input.pooling_metadata,
                request_ids=model_input.ordered_pooling_request_ids,
                finished_requests_ids=model_input.finished_requests_ids
            )
            model_outputs_dict["pooling"] = pooler_output_obj

        # Sampling part
        if model_input.sampling_metadata and model_input.sampling_metadata.seq_groups:
            entire_hidden_states = (
                hidden_or_intermediate_states.last_hidden_state
                if isinstance(hidden_or_intermediate_states, IntermediateTensors)
                else hidden_or_intermediate_states
            )

            if model_input.list_of_token_indices_for_gathering_pooling_hs:
                pooling_indices = torch.cat(
                    model_input.list_of_token_indices_for_gathering_pooling_hs, dim=0
                )
                mask = torch.ones(
                    entire_hidden_states.size(0),
                    dtype=torch.bool,
                    device=entire_hidden_states.device,
                )
                mask[pooling_indices] = False
                sampling_indices = torch.nonzero(mask, as_tuple=True)[0]
                hidden_states_for_sampler = entire_hidden_states.index_select(
                    0, sampling_indices
                )
            else:
                hidden_states_for_sampler = entire_hidden_states

            # print(f"Sampler will see {hidden_states_for_sampler.shape[0]} rows (vs {entire_hidden_states.shape[0]})")

            # print(f"actual_hidden_states_for_sampler shape: {hidden_states_for_sampler.shape}")
            logits = self.model.compute_logits(
                hidden_states_for_sampler,
                model_input.sampling_metadata
            )
            
            sampler_output_obj = self.sampler(
                logits=logits,
                sampling_metadata=model_input.sampling_metadata,
            )

            if model_input.async_callback is not None:
                model_input.async_callback()

            if (self.observability_config is not None and
                    self.observability_config.collect_model_forward_time and
                    sampler_output_obj is not None and model_forward_start and model_forward_end):
                model_forward_end.synchronize()
                model_execution_time = model_forward_start.elapsed_time(model_forward_end)

                orig_model_forward_time = 0.0
                if intermediate_tensors is not None and hasattr(intermediate_tensors, 'tensors') and isinstance(intermediate_tensors.tensors, dict):
                    orig_model_forward_time = intermediate_tensors.tensors.get(
                        "model_forward_time", torch.tensor(0.0)).item()
                
                sampler_output_obj.model_forward_time = (orig_model_forward_time + model_execution_time)


            if self.return_hidden_states and sampler_output_obj:
                assert model_input.sampling_metadata is not None
                indices = model_input.sampling_metadata.selected_token_indices
                if model_input.is_prompt:
                    hidden_states = hidden_states_for_sampler.index_select(
                        0, indices)
                    sampler_output_obj.prefill_hidden_states = hidden_states_for_sampler
                elif decode_meta.use_cuda_graph:
                    hidden_states = hidden_states_for_sampler[:len(indices)]
                else:
                    hidden_states = hidden_states_for_sampler
                sampler_output_obj.hidden_states = hidden_states
            
            model_outputs_dict["sampling"] = sampler_output_obj
            
        outputs_list: List[Union[PoolerOutput, SamplerOutput]] = []
        if model_outputs_dict["pooling"] is not None:
            outputs_list.append(model_outputs_dict["pooling"])
        if model_outputs_dict["sampling"] is not None:
            outputs_list.append(model_outputs_dict["sampling"])

        # print(f"Returning {len(outputs_list)} outputs: " + 
        #       ", ".join(type(o).__name__ for o in outputs_list))
        # print("-----------------------------------------------------------------")
        return outputs_list
    

    def make_model_input_from_broadcasted_tensor_dict(
        self,
        tensor_dict: Dict[str, Any]
    ) -> HybridModelInputForGPU:
        return HybridModelInputForGPU.from_broadcasted_tensor_dict(
            tensor_dict,
            attn_backend=self.attn_backend,
        )


    def _prepare_pooling_and_ids(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        all_seq_lens_in_batch: List[int],
        device: torch.device,
    ) -> Tuple[Optional[PoolingMetadata], Optional[List[str]], Optional[List[torch.Tensor]]]:
        pooling_sgs = [sg for sg in seq_group_metadata_list if sg.pooling_params is not None]
        if not pooling_sgs:
            return None, None, None

        indices_per_req: Dict[str, List[int]] = {}
        seq_data_per_req: Dict[str, Dict[int, SequenceData]] = {}
        params_per_req: Dict[str, PoolingParams] = {}

        offset = 0
        for length, sg in zip(all_seq_lens_in_batch, seq_group_metadata_list):
            # print(f"sequence group: {sg.request_id}, {sg.token_chunk_size = }")
            seq_len = sg.token_chunk_size if getattr(sg, "token_chunk_size", None) else length
            first_seq = next(iter(sg.seq_data.values()))
            # print(f"Processing sequence group: {sg.request_id}, length: {length}, seq_len: {seq_len}, cached_tokens: {first_seq.get_num_cached_tokens()}")
            if sg.pooling_params is not None:
                rid = sg.request_id
                indices_per_req.setdefault(rid, []).extend(range(offset, offset + seq_len - first_seq.get_num_cached_tokens()))
                seq_data_per_req.setdefault(rid, {}).update(sg.seq_data)
                params_per_req[rid] = sg.pooling_params

            offset += seq_len - first_seq.get_num_cached_tokens()

        ordered_pooling_request_ids: List[str] = []
        list_of_indices_to_gather: List[torch.Tensor] = []
        compact_pm_prompt_lens: List[int] = []
        compact_pm_seq_groups: List[Tuple[List[int], PoolingParams]] = []
        compact_pm_seq_data: Dict[int, SequenceData] = {}

        for rid, idx_list in indices_per_req.items():
            idxs = torch.tensor(idx_list, dtype=torch.long, device=device)
            ordered_pooling_request_ids.append(rid)
            list_of_indices_to_gather.append(idxs)
            compact_pm_prompt_lens.append(idxs.numel())

            compact_pm_seq_groups.append((idx_list, params_per_req[rid]))
            compact_pm_seq_data.update(seq_data_per_req[rid])

        # print(f"Number of pooling sequence groups: {len(compact_pm_seq_groups)}")
        # print(f"Pooling prompt lengths: {compact_pm_prompt_lens}")
        # print(f"Pooling request IDs: {ordered_pooling_request_ids}")

        pooling_metadata = PoolingMetadata(
            seq_groups=compact_pm_seq_groups,
            seq_data=compact_pm_seq_data,
            prompt_lens=compact_pm_prompt_lens,
        )

        return pooling_metadata, ordered_pooling_request_ids, list_of_indices_to_gather


    def prepare_model_input(
        self,
        seq_group_metadata_list: Optional[List[SequenceGroupMetadata]],
        virtual_engine: int = 0,
        finished_requests_ids: Optional[List[str]] = None
    ) -> HybridModelInputForGPU:
        assert seq_group_metadata_list is not None
        base_model_input_args = self._prepare_model_input_tensors(
            seq_group_metadata_list, finished_requests_ids
        )
        assert base_model_input_args.seq_lens is not None
        current_seq_lens = base_model_input_args.seq_lens

        # print("################# PREPARE MODEL INPUT ####################")
        # for sg_metadata_item in seq_group_metadata_list:
        #     print(f"request_id: {sg_metadata_item.request_id}, "
        #           f"is_prompt: {sg_metadata_item.is_prompt}, "
        #           f"sampling_params: {'Yes' if sg_metadata_item.sampling_params else 'No'}, "
        #           f"pooling_params: {'Yes' if sg_metadata_item.pooling_params else 'No'}")
        # print("############################################################")

        pooling_metadata_for_compact_view, ordered_pooling_request_ids, list_of_indices_to_gather = \
            self._prepare_pooling_and_ids(
                seq_group_metadata_list,
                current_seq_lens, 
                self.device
            )

        lora_row_indices: Optional[torch.Tensor] = None
        if list_of_indices_to_gather:
            lora_row_indices = torch.cat(list_of_indices_to_gather, dim=0)

        ordered_sampling_request_ids: List[str] = []
        sampling_metadata: Optional[SamplingMetadata] = None
        if get_pp_group().is_last_rank:
            sampling_seq_groups = [
                sg for sg in seq_group_metadata_list
                if sg.sampling_params is not None
            ]
            # print(f"Number of sampling sequence groups: {len(sampling_seq_groups)}")
            if sampling_seq_groups:
                sampling_positions = [
                    idx for idx, sg in enumerate(seq_group_metadata_list)
                    if sg.sampling_params is not None
                ]
                seq_lens_for_sampling = [
                    current_seq_lens[i] for i in sampling_positions
                ]
                if base_model_input_args.query_lens is not None:
                    all_q_lens = base_model_input_args.query_lens
                    query_lens_for_sampling = [
                        all_q_lens[i] for i in sampling_positions
                    ]
                else:
                    query_lens_for_sampling = [
                        len(sg.prompt_token_ids) for sg in sampling_seq_groups
                    ]
                generators = self.get_generators(finished_requests_ids)
                # print(
                #     f"{len(sampling_seq_groups)} sampling groups, "
                #     f"seq_lens: {seq_lens_for_sampling}, "
                #     f"query_lens: {query_lens_for_sampling}"
                # )
                sampling_metadata = SamplingMetadata.prepare(
                    sampling_seq_groups,
                    seq_lens_for_sampling,
                    query_lens_for_sampling,
                    self.device,
                    self.pin_memory,
                    generators,
                    self.sampling_metadata_cache
                )
            else:
                sampling_metadata = None
        
        is_prompt_for_batch = seq_group_metadata_list[0].is_prompt if seq_group_metadata_list else None
        pooling_groups = [
            sg for sg in seq_group_metadata_list
            if sg.pooling_params is not None
        ]
        finished_request_ids_set = set()
        if pooling_groups:
            for sg in pooling_groups:
                first_seq = next(iter(sg.seq_data.values()))
                if sg.token_chunk_size == first_seq.get_num_uncomputed_tokens():
                    finished_request_ids_set.add(sg.request_id)

        return dataclasses.replace(
            base_model_input_args,
            pooling_metadata=pooling_metadata_for_compact_view,
            sampling_metadata=sampling_metadata,
            ordered_pooling_request_ids=ordered_pooling_request_ids,
            ordered_sampling_request_ids=ordered_sampling_request_ids,
            list_of_token_indices_for_gathering_pooling_hs=list_of_indices_to_gather,
            lora_row_indices=lora_row_indices,
            is_prompt=is_prompt_for_batch,
            virtual_engine=virtual_engine,
            finished_requests_ids=finished_request_ids_set,
        )

    def need_recv_kv(self, model_input, kv_caches) -> bool:
        """Check if this runner should receive KV cache from another worker."""
        if self.vllm_config.kv_transfer_config is None:
            return False
        prefill_meta = model_input.attn_metadata.prefill_metadata
        is_profile_run = (kv_caches and kv_caches[0].numel() == 0)
        is_prefill_run = prefill_meta is not None
        return (
            self.vllm_config.kv_transfer_config.is_kv_consumer
            and (not is_profile_run)
            and is_prefill_run
        )

    def need_send_kv(self, model_input, kv_caches) -> bool:
        """Check if this runner should send KV cache to another worker."""
        if self.vllm_config.kv_transfer_config is None:
            return False
        prefill_meta = model_input.attn_metadata.prefill_metadata
        is_profile_run = (kv_caches and kv_caches[0].numel() == 0)
        is_prefill_run = prefill_meta is not None
        return (
            self.vllm_config.kv_transfer_config.is_kv_producer
            and (not is_profile_run)
            and is_prefill_run
        )
