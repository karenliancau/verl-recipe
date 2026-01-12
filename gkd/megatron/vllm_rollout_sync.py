# Copyright 2025 Individual Contributor
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
vLLM Sync Rollout for GKD (Generalized Knowledge Distillation).

This module provides a synchronous vLLM rollout implementation for GKD training.
Unlike vLLMAsyncRollout which uses server mode, this implementation uses the
vLLM LLM class directly for synchronous batch generation.
"""

import logging
import os
from typing import Generator

import torch
import torch.distributed
from omegaconf import ListConfig
from tensordict import TensorDict
from torch.distributed.device_mesh import DeviceMesh
from vllm import LLM, SamplingParams
from vllm.config import CompilationConfig, CompilationLevel

from verl import DataProto
from verl.third_party.vllm import VLLM_SLEEP_LEVEL
from verl.utils.device import get_device_name, get_torch_device
from verl.utils.torch_functional import get_response_mask
from verl.workers.config import HFModelConfig, RolloutConfig
from verl.workers.rollout.base import BaseRollout

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class vLLMSyncRollout(BaseRollout):
    """Synchronous vLLM rollout using the LLM class for batch generation.

    This rollout implementation uses vLLM's LLM class directly for synchronous
    batch generation, which is suitable for GKD training where we need:
    1. Synchronous generation (not async server mode)
    2. Direct weight updates via load_weights
    3. Efficient batch processing
    """

    def __init__(
        self,
        config: RolloutConfig,
        model_config: HFModelConfig,
        device_mesh: DeviceMesh,
    ):
        super().__init__(config, model_config, device_mesh)
        self.config = config
        self.model_config = model_config
        self.device_mesh = device_mesh

        if config.layered_summon:
            self.sleep_level = 1
        else:
            self.sleep_level = VLLM_SLEEP_LEVEL

        model_path = model_config.local_path
        tokenizer = model_config.tokenizer
        model_hf_config = model_config.hf_config
        trust_remote_code = model_config.trust_remote_code

        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id

        tensor_parallel_size = self.config.get("tensor_model_parallel_size", 1)
        assert tensor_parallel_size <= torch.distributed.get_world_size(), (
            "tensor parallel size should be less than or equal to the world size"
        )
        max_num_batched_tokens = self.config.get("max_num_batched_tokens", 8192)

        # Handle rope scaling configuration
        rope_scaling_config = getattr(model_hf_config, "rope_scaling", None)
        if not rope_scaling_config:
            max_position_embeddings = None
            if hasattr(model_hf_config, "max_position_embeddings"):
                max_position_embeddings = model_hf_config.max_position_embeddings
            elif hasattr(model_hf_config, "llm_config") and hasattr(
                model_hf_config.llm_config, "max_position_embeddings"
            ):
                max_position_embeddings = model_hf_config.llm_config.max_position_embeddings
            elif hasattr(model_hf_config, "text_config") and hasattr(
                model_hf_config.text_config, "max_position_embeddings"
            ):
                max_position_embeddings = model_hf_config.text_config.max_position_embeddings
            if max_position_embeddings is None:
                raise ValueError("max_position_embeddings not found in model_hf_config")
            assert max_position_embeddings >= config.prompt_length + config.response_length, (
                "model context length should be greater than total sequence length"
            )
        else:
            rope_scaling_factor = rope_scaling_config.get("factor", 1.0)
            assert (
                model_hf_config.max_position_embeddings * rope_scaling_factor
                >= config.prompt_length + config.response_length
            ), (
                "model context length should be greater than total sequence length, "
                + f"got rope_scaling_factor={rope_scaling_factor} and "
                + f"max_position_embeddings={model_hf_config.max_position_embeddings}"
            )

        max_model_len = int(config.max_model_len or config.prompt_length + config.response_length)

        # Use dummy load format for Megatron weight sync
        load_format = "dummy" if config.load_format.startswith("dummy") else config.load_format

        # Copy engine kwargs to avoid secretly modifying the engine config
        engine_kwargs = config.get("engine_kwargs", {}).get("vllm", {}) or {}
        engine_kwargs = {key: val for key, val in engine_kwargs.items() if val is not None}

        if config.get("limit_images", None):
            engine_kwargs["limit_mm_per_prompt"] = {"image": config.get("limit_images")}

        compilation_config = {}
        cudagraph_capture_sizes = config.get("cudagraph_capture_sizes")
        if not config.enforce_eager and cudagraph_capture_sizes:
            if isinstance(cudagraph_capture_sizes, ListConfig):
                compilation_config["compilation_config"] = CompilationConfig(
                    level=CompilationLevel.PIECEWISE, cudagraph_capture_sizes=cudagraph_capture_sizes
                )
            else:
                logger.warning(f"cudagraph_capture_sizes must be a list, but got {cudagraph_capture_sizes}")

        # Initialize vLLM LLM engine
        self.inference_engine = LLM(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            distributed_executor_backend="external_launcher",
            dtype=config.dtype,
            enforce_eager=config.enforce_eager,
            gpu_memory_utilization=config.gpu_memory_utilization,
            disable_custom_all_reduce=True,
            skip_tokenizer_init=False,
            max_model_len=max_model_len,
            max_num_seqs=config.max_num_seqs,
            load_format=load_format,
            disable_log_stats=config.disable_log_stats,
            max_num_batched_tokens=max_num_batched_tokens,
            enable_chunked_prefill=config.enable_chunked_prefill,
            enable_prefix_caching=False,
            trust_remote_code=trust_remote_code,
            seed=config.get("seed", 0),
            **compilation_config,
            **engine_kwargs,
        )

        # Build default sampling params
        kwargs = dict(
            n=1,
            logprobs=0,
            max_tokens=config.response_length,
            repetition_penalty=config.get("repetition_penalty", 1.0),
            detokenize=False,
        )

        # Add sampling params from config
        for k in config.keys():
            if hasattr(SamplingParams(), str(k)) and k != "seed":
                kwargs[k] = config.get(k)
        kwargs["n"] = 1  # already repeat in ray_trainer

        logger.info(f"vLLM sync rollout sampling kwargs: {kwargs}")
        self.sampling_params = SamplingParams(**kwargs)

        # Store model reference for weight updates
        self.model = self.inference_engine.llm_engine.model_executor.driver_worker.worker.model_runner.model

    def generate_sequences(self, prompts: DataProto) -> DataProto:
        """Generate sequences synchronously using vLLM.

        Args:
            prompts: DataProto containing input_ids, attention_mask, position_ids

        Returns:
            DataProto containing generated sequences with prompts, responses, etc.
        """
        # Extract prompt token ids from batch
        input_ids = prompts.batch["input_ids"]  # (batch_size, prompt_length)
        attention_mask = prompts.batch["attention_mask"]
        position_ids = prompts.batch["position_ids"]

        batch_size = input_ids.size(0)
        prompt_length = input_ids.size(1)

        # Get EOS and PAD token ids
        eos_token_id = prompts.meta_info.get("eos_token_id", self.tokenizer.eos_token_id)
        pad_token_id = prompts.meta_info.get("pad_token_id", self.tokenizer.pad_token_id)

        # Get sampling parameters from meta_info or use defaults
        temperature = prompts.meta_info.get("temperature", self.config.temperature)
        response_length = prompts.meta_info.get("response_length", self.config.response_length)
        top_p = prompts.meta_info.get("top_p", self.config.get("top_p", 1.0))
        top_k = prompts.meta_info.get("top_k", self.config.get("top_k", -1))

        # Create sampling params
        sampling_params = SamplingParams(
            n=1,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k if top_k > 0 else -1,
            max_tokens=response_length,
            detokenize=False,
        )

        # Convert input_ids to list of lists for vLLM
        # vLLM expects list of token id lists, not padded tensors
        prompt_token_ids = []
        for i in range(batch_size):
            # Get actual tokens (non-padded)
            mask = attention_mask[i].bool()
            tokens = input_ids[i][mask].tolist()
            prompt_token_ids.append(tokens)

        # Generate using vLLM
        outputs = self.inference_engine.generate(
            prompt_token_ids=prompt_token_ids,
            sampling_params=sampling_params,
        )

        # Process outputs
        all_responses = []
        for output in outputs:
            response_tokens = list(output.outputs[0].token_ids)
            all_responses.append(response_tokens)

        # Pad responses to response_length
        padded_responses = []
        for resp in all_responses:
            if len(resp) < response_length:
                resp = resp + [pad_token_id] * (response_length - len(resp))
            else:
                resp = resp[:response_length]
            padded_responses.append(resp)

        # Convert to tensors
        responses = torch.tensor(padded_responses, dtype=input_ids.dtype, device=input_ids.device)

        # Concatenate prompt and response
        seq = torch.cat([input_ids, responses], dim=1)

        # Build attention mask for full sequence
        response_attention_mask = get_response_mask(
            response_id=responses, eos_token=eos_token_id, dtype=attention_mask.dtype
        )
        full_attention_mask = torch.cat([attention_mask, response_attention_mask], dim=-1)

        # Build position ids for full sequence
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).repeat(batch_size, 1)
        response_position_ids = position_ids[:, -1:] + delta_position_id
        full_position_ids = torch.cat([position_ids, response_position_ids], dim=-1)

        # Create output batch
        batch = TensorDict(
            {
                "prompts": input_ids,
                "responses": responses,
                "input_ids": seq,
                "attention_mask": full_attention_mask,
                "position_ids": full_position_ids,
            },
            batch_size=batch_size,
        )

        get_torch_device().empty_cache()
        return DataProto(batch=batch)

    async def update_weights(self, weights: Generator[tuple[str, torch.Tensor], None, None], **kwargs):
        """Update the weights of the rollout model.

        Args:
            weights: A generator that yields the name of the weight tensor and the tensor itself.
        """
        from verl.utils.vllm.patch import patch_vllm_moe_model_weight_loader

        patch_vllm_moe_model_weight_loader(self.model)
        self.model.load_weights(weights)

    async def resume(self, tags: list[str]):
        """Resume rollout weights or kv cache in GPU memory."""
        pass  # Not needed for sync mode

    async def release(self):
        """Release weights and kv cache in GPU memory."""
        pass  # Not needed for sync mode


# Register the sync rollout
from verl.workers.rollout.base import _ROLLOUT_REGISTRY
_ROLLOUT_REGISTRY[("vllm", "sync")] = "recipe.gkd.megatron.vllm_rollout_sync.vLLMSyncRollout"
