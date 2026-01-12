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
vLLM Rollout for GKD (Generalized Knowledge Distillation).

This module provides a unified vLLM rollout implementation for GKD training.
It uses the vLLM LLM class directly for batch generation with support for
HCCL/NCCL weight synchronization from Megatron actors.

Supports two generation modes controlled by config.mode:
1. mode: "sync" (or other) - Direct batch generation via LLM.generate() only
2. mode: "async" - Also starts ZeroMQ server for async requests

Key features:
- Compatible with dummy_megatron load format for weight sync
- Direct weight updates via model.load_weights()
- Efficient batch processing with vLLM's continuous batching
- Optional ZeroMQ async server (when mode="async") for non-blocking requests

Note: Both modes always support synchronous generate_sequences() calls.
The "async" mode additionally provides a ZeroMQ server for remote async requests.
Weight synchronization always uses HCCL/NCCL broadcast regardless of mode.
"""

import asyncio
import getpass
import logging
import os
from typing import Any, Generator, Optional

import cloudpickle as pickle
import torch
import torch.distributed
import zmq
import zmq.asyncio
from filelock import FileLock
from omegaconf import ListConfig
from tensordict import TensorDict
from torch.distributed.device_mesh import DeviceMesh
from vllm import LLM, SamplingParams
from vllm.config import CompilationConfig, CompilationLevel

from verl import DataProto
from verl.third_party.vllm import VLLM_SLEEP_LEVEL
from verl.utils.device import get_device_name, get_torch_device
from verl.utils.ray_utils import get_event_loop
from verl.utils.torch_functional import get_response_mask
from verl.workers.config import HFModelConfig, RolloutConfig
from verl.workers.rollout.base import BaseRollout
from verl.workers.rollout.utils import get_free_port, is_valid_ipv6_address

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _get_config_value(config, key, default=None):
    """Helper to get config value from either OmegaConf or dataclass."""
    if hasattr(config, "get"):
        return config.get(key, default)
    return getattr(config, key, default)


class vLLMRollout(BaseRollout):
    """Unified vLLM rollout for GKD training with weight synchronization support.

    This rollout implementation uses vLLM's LLM class directly for batch generation,
    which is suitable for GKD training where we need:
    1. Direct weight updates via load_weights() for Megatron weight sync
    2. Efficient batch processing with vLLM's continuous batching
    3. Optional async server mode via ZeroMQ for non-blocking requests

    The class supports two modes:
    - Sync mode (default): Direct batch generation via generate_sequences()
    - Async server mode: ZeroMQ server for async requests (enable_async_server=True)

    Both modes use HCCL/NCCL for weight synchronization, only the generation
    request handling differs.
    """

    def __init__(
        self,
        config,  # Can be RolloutConfig dataclass or OmegaConf DictConfig
        model_config: HFModelConfig,
        device_mesh: DeviceMesh,
    ):
        # Don't call super().__init__ with strict type checking since config might be OmegaConf
        self.config = config
        self.model_config = model_config
        self.device_mesh = device_mesh

        if _get_config_value(config, "layered_summon", False):
            self.sleep_level = 1
        else:
            self.sleep_level = VLLM_SLEEP_LEVEL

        model_path = model_config.local_path
        tokenizer = model_config.tokenizer
        model_hf_config = model_config.hf_config
        trust_remote_code = model_config.trust_remote_code

        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id

        tensor_parallel_size = _get_config_value(config, "tensor_model_parallel_size", 1)
        assert tensor_parallel_size <= torch.distributed.get_world_size(), (
            "tensor parallel size should be less than or equal to the world size"
        )
        max_num_batched_tokens = _get_config_value(config, "max_num_batched_tokens", 8192)

        # Get config values using helper function
        prompt_length = _get_config_value(config, "prompt_length", 512)
        response_length = _get_config_value(config, "response_length", 512)

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
            assert max_position_embeddings >= prompt_length + response_length, (
                "model context length should be greater than total sequence length"
            )
        else:
            rope_scaling_factor = rope_scaling_config.get("factor", 1.0)
            assert (
                model_hf_config.max_position_embeddings * rope_scaling_factor
                >= prompt_length + response_length
            ), (
                "model context length should be greater than total sequence length, "
                + f"got rope_scaling_factor={rope_scaling_factor} and "
                + f"max_position_embeddings={model_hf_config.max_position_embeddings}"
            )

        max_model_len = int(_get_config_value(config, "max_model_len") or prompt_length + response_length)

        # Use dummy load format for Megatron weight sync
        load_format_raw = _get_config_value(config, "load_format", "dummy")
        load_format = "dummy" if load_format_raw.startswith("dummy") else load_format_raw

        # Copy engine kwargs to avoid secretly modifying the engine config
        engine_kwargs_raw = _get_config_value(config, "engine_kwargs", {})
        if engine_kwargs_raw:
            engine_kwargs = _get_config_value(engine_kwargs_raw, "vllm", {}) or {}
            engine_kwargs = {key: val for key, val in engine_kwargs.items() if val is not None}
        else:
            engine_kwargs = {}

        limit_images = _get_config_value(config, "limit_images")
        if limit_images:
            engine_kwargs["limit_mm_per_prompt"] = {"image": limit_images}

        compilation_config = {}
        cudagraph_capture_sizes = _get_config_value(config, "cudagraph_capture_sizes")
        enforce_eager = _get_config_value(config, "enforce_eager", True)
        if not enforce_eager and cudagraph_capture_sizes:
            if isinstance(cudagraph_capture_sizes, (list, ListConfig)):
                compilation_config["compilation_config"] = CompilationConfig(
                    level=CompilationLevel.PIECEWISE, cudagraph_capture_sizes=list(cudagraph_capture_sizes)
                )
            else:
                logger.warning(f"cudagraph_capture_sizes must be a list, but got {cudagraph_capture_sizes}")

        # Get remaining config values
        dtype = _get_config_value(config, "dtype", "bfloat16")
        gpu_memory_utilization = _get_config_value(config, "gpu_memory_utilization", 0.5)
        max_num_seqs = _get_config_value(config, "max_num_seqs", 1024)
        disable_log_stats = _get_config_value(config, "disable_log_stats", True)
        enable_chunked_prefill = _get_config_value(config, "enable_chunked_prefill", False)
        seed = _get_config_value(config, "seed", 0)

        # Initialize vLLM LLM engine
        self.inference_engine = LLM(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            distributed_executor_backend="external_launcher",
            dtype=dtype,
            enforce_eager=enforce_eager,
            gpu_memory_utilization=gpu_memory_utilization,
            disable_custom_all_reduce=True,
            skip_tokenizer_init=False,
            max_model_len=max_model_len,
            max_num_seqs=max_num_seqs,
            load_format=load_format,
            disable_log_stats=disable_log_stats,
            max_num_batched_tokens=max_num_batched_tokens,
            enable_chunked_prefill=enable_chunked_prefill,
            enable_prefix_caching=False,
            trust_remote_code=trust_remote_code,
            seed=seed,
            **compilation_config,
            **engine_kwargs,
        )

        # Build default sampling params
        temperature = _get_config_value(config, "temperature", 1.0)
        top_p = _get_config_value(config, "top_p", 1.0)
        top_k = _get_config_value(config, "top_k", -1)
        repetition_penalty = _get_config_value(config, "repetition_penalty", 1.0)

        kwargs = dict(
            n=1,
            logprobs=0,
            max_tokens=response_length,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k if top_k > 0 else -1,
            repetition_penalty=repetition_penalty,
            detokenize=False,
        )

        logger.info(f"vLLM sync rollout sampling kwargs: {kwargs}")
        self.sampling_params = SamplingParams(**kwargs)

        # Store config values for generate_sequences
        self.response_length = response_length
        self.max_model_len = max_model_len

        # Store model reference for weight updates
        self.model = self.inference_engine.llm_engine.model_executor.driver_worker.worker.model_runner.model

        # Determine async server mode from config.mode
        # mode: "async" -> enable ZeroMQ server for async requests
        # mode: "sync" or any other value -> sync-only mode (no ZeroMQ server)
        mode = _get_config_value(config, "mode", "sync")
        self._async_server_enabled = (mode == "async")
        self._zmq_address: Optional[str] = None
        self._zmq_socket = None
        self._zmq_context = None
        self._zmq_loop_task = None

        if self._async_server_enabled:
            self._init_zeromq_server()
            logger.info(f"vLLMRollout async server started at {self._zmq_address}")

    # ==================== ZeroMQ Async Server ====================

    def _init_zeromq_server(self) -> str:
        """Initialize ZeroMQ REP socket for async requests.

        Returns:
            str: The ZeroMQ address (ipc:// or tcp://)
        """
        try:
            import ray
            local_world_size = int(os.environ.get("RAY_LOCAL_WORLD_SIZE", "1"))
            ip = ray.util.get_node_ip_address().strip("[]")
        except Exception:
            local_world_size = 1
            ip = "127.0.0.1"

        tensor_parallel_size = _get_config_value(self.config, "tensor_model_parallel_size", 1)
        socket_type = "ipc" if tensor_parallel_size <= local_world_size else "tcp"

        # File lock to prevent multiple workers listen to same port
        with FileLock(f"/tmp/verl_gkd_vllm_zmq_{getpass.getuser()}.lock"):
            self._zmq_context = zmq.asyncio.Context()
            self._zmq_socket = self._zmq_context.socket(zmq.REP)

            if socket_type == "ipc":
                pid = os.getpid()
                address = f"ipc:///tmp/verl_gkd_vllm_zmq_{pid}_{getpass.getuser()}.ipc"
            else:
                port, _ = get_free_port(ip)
                if is_valid_ipv6_address(ip):
                    address = f"tcp://[{ip}]:{port}"
                    self._zmq_socket.setsockopt(zmq.IPV6, 1)
                else:
                    address = f"tcp://{ip}:{port}"

            self._zmq_socket.bind(address)
            self._zmq_address = address

        # Start async event loop for handling requests
        loop = get_event_loop()
        self._zmq_loop_task = loop.create_task(self._zmq_server_loop())

        return address

    async def _zmq_server_loop(self):
        """Main loop for handling ZeroMQ async requests."""
        while True:
            try:
                message = await self._zmq_socket.recv()
                method, args, kwargs = pickle.loads(message)

                # Execute the requested method
                result = await self._execute_async_method(method, *args, **kwargs)

                await self._zmq_socket.send(pickle.dumps(result))
            except asyncio.CancelledError:
                logger.info("ZeroMQ server loop cancelled")
                break
            except Exception as e:
                logger.exception(f"vLLMRollout ZeroMQ server error: {e}")
                try:
                    await self._zmq_socket.send(pickle.dumps(e))
                except Exception:
                    pass
                break

    async def _execute_async_method(self, method: str, *args, **kwargs) -> Any:
        """Execute a method requested via ZeroMQ.

        Args:
            method: Method name to execute
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            The result of the method execution
        """
        if method == "generate":
            return await self._async_generate(*args, **kwargs)
        elif method == "generate_sequences":
            # Handle DataProto generation request
            prompts_dict = args[0] if args else kwargs.get("prompts")
            prompts = DataProto.from_dict(prompts_dict)
            result = self.generate_sequences(prompts)
            return result.to_dict()
        elif method == "health_check":
            return {"status": "ok", "address": self._zmq_address}
        else:
            raise ValueError(f"Unknown method: {method}")

    async def _async_generate(
        self,
        prompt_token_ids: list[list[int]],
        sampling_params: dict[str, Any],
        **kwargs,
    ) -> dict[str, Any]:
        """Async generation for individual requests.

        Args:
            prompt_token_ids: List of prompt token id lists
            sampling_params: Sampling parameters dict

        Returns:
            dict containing generated token_ids and optional log_probs
        """
        # Create SamplingParams from dict
        max_tokens = sampling_params.pop("max_tokens", self.response_length)
        temperature = sampling_params.pop("temperature", 1.0)
        top_p = sampling_params.pop("top_p", 1.0)
        top_k = sampling_params.pop("top_k", -1)

        params = SamplingParams(
            n=1,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k if top_k > 0 else -1,
            max_tokens=max_tokens,
            detokenize=False,
            **sampling_params,
        )

        # Run synchronous generation in executor to avoid blocking
        loop = asyncio.get_event_loop()
        outputs = await loop.run_in_executor(
            None,
            lambda: self.inference_engine.generate(
                prompt_token_ids=prompt_token_ids,
                sampling_params=params,
            )
        )

        # Process outputs
        results = []
        for output in outputs:
            token_ids = list(output.outputs[0].token_ids)
            log_probs = None
            if output.outputs[0].logprobs:
                log_probs = [
                    lp[token_ids[i]].logprob
                    for i, lp in enumerate(output.outputs[0].logprobs)
                ]
            results.append({
                "token_ids": token_ids,
                "log_probs": log_probs,
            })

        return {"outputs": results}

    def get_zeromq_address(self) -> Optional[str]:
        """Get the ZeroMQ server address if async server is enabled.

        Returns:
            str or None: The ZeroMQ address or None if not enabled
        """
        return self._zmq_address

    def is_async_server_enabled(self) -> bool:
        """Check if async server mode is enabled."""
        return self._async_server_enabled

    # ==================== Sync Generation ====================

    def _tokenize_prompts(self, raw_prompts: list) -> tuple[list[list[int]], int]:
        """Tokenize raw prompts using chat template.

        Args:
            raw_prompts: List of raw prompt messages (conversation format)

        Returns:
            Tuple of (prompt_token_ids, max_prompt_len)
        """
        prompt_token_ids = []
        max_prompt_len = 0

        for raw_prompt in raw_prompts:
            # Apply chat template to get token ids
            # raw_prompt is typically a list of message dicts like [{"role": "user", "content": "..."}]
            token_ids = self.tokenizer.apply_chat_template(
                raw_prompt,
                add_generation_prompt=True,
                tokenize=True,
            )
            prompt_token_ids.append(token_ids)
            max_prompt_len = max(max_prompt_len, len(token_ids))

        return prompt_token_ids, max_prompt_len

    def _pad_prompts(
        self,
        prompt_token_ids: list[list[int]],
        max_len: int,
        pad_token_id: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pad prompts to the same length (left padding).

        Args:
            prompt_token_ids: List of token id lists
            max_len: Maximum length to pad to
            pad_token_id: Padding token id

        Returns:
            Tuple of (input_ids, attention_mask, position_ids) tensors
        """
        batch_size = len(prompt_token_ids)
        input_ids = torch.full((batch_size, max_len), pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
        position_ids = torch.zeros((batch_size, max_len), dtype=torch.long)

        for i, tokens in enumerate(prompt_token_ids):
            seq_len = len(tokens)
            # Left padding: place tokens at the end
            start_pos = max_len - seq_len
            input_ids[i, start_pos:] = torch.tensor(tokens, dtype=torch.long)
            attention_mask[i, start_pos:] = 1
            position_ids[i, start_pos:] = torch.arange(seq_len, dtype=torch.long)

        return input_ids, attention_mask, position_ids

    def generate_sequences(self, prompts: DataProto) -> DataProto:
        """Generate sequences synchronously using vLLM.

        This method handles tokenization from raw_prompt in non_tensor_batch,
        similar to how AgentLoop processes prompts.

        Args:
            prompts: DataProto containing raw_prompt in non_tensor_batch

        Returns:
            DataProto containing generated sequences with prompts, responses, etc.
        """
        # Get raw prompts from non_tensor_batch
        raw_prompts = prompts.non_tensor_batch.get("raw_prompt")
        if raw_prompts is None:
            raise ValueError(
                "raw_prompt not found in non_tensor_batch. "
                "GKD rollout expects raw_prompt for tokenization."
            )

        batch_size = len(raw_prompts)

        # Tokenize prompts using chat template
        prompt_token_ids, max_prompt_len = self._tokenize_prompts(raw_prompts)

        # Get config values
        prompt_length = _get_config_value(self.config, "prompt_length", 512)
        response_length = prompts.meta_info.get("response_length", self.response_length)

        # Truncate prompts if needed
        for i in range(len(prompt_token_ids)):
            if len(prompt_token_ids[i]) > prompt_length:
                # Truncate from the left (keep the most recent context)
                prompt_token_ids[i] = prompt_token_ids[i][-prompt_length:]

        # Update max_prompt_len after truncation
        max_prompt_len = min(max_prompt_len, prompt_length)

        # Get EOS and PAD token ids
        eos_token_id = prompts.meta_info.get("eos_token_id", self.tokenizer.eos_token_id)
        pad_token_id = prompts.meta_info.get("pad_token_id", self.tokenizer.pad_token_id)
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id

        # Pad prompts to prompt_length (left padding)
        input_ids, attention_mask, position_ids = self._pad_prompts(
            prompt_token_ids, prompt_length, pad_token_id
        )

        # Get sampling parameters from meta_info or use defaults
        temperature = prompts.meta_info.get("temperature", _get_config_value(self.config, "temperature", 1.0))
        top_p = prompts.meta_info.get("top_p", _get_config_value(self.config, "top_p", 1.0))
        top_k = prompts.meta_info.get("top_k", _get_config_value(self.config, "top_k", -1))

        # Create sampling params
        sampling_params = SamplingParams(
            n=1,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k if top_k > 0 else -1,
            max_tokens=response_length,
            detokenize=False,
        )

        # Convert prompt_token_ids to TokensPrompt format for vLLM
        # vLLM expects prompts as list of dicts with "prompt_token_ids" key
        token_prompts = [{"prompt_token_ids": ids} for ids in prompt_token_ids]

        # Generate using vLLM
        outputs = self.inference_engine.generate(
            prompts=token_prompts,
            sampling_params=sampling_params,
        )

        # Process outputs
        all_responses = []
        for output in outputs:
            response_tokens = list(output.outputs[0].token_ids)
            all_responses.append(response_tokens)

        # Pad responses to response_length (right padding)
        padded_responses = []
        for resp in all_responses:
            if len(resp) < response_length:
                resp = resp + [pad_token_id] * (response_length - len(resp))
            else:
                resp = resp[:response_length]
            padded_responses.append(resp)

        # Convert to tensors
        device = get_torch_device().current_device()
        responses = torch.tensor(padded_responses, dtype=torch.long, device=device)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        position_ids = position_ids.to(device)

        # Concatenate prompt and response
        seq = torch.cat([input_ids, responses], dim=1)

        # Build attention mask for full sequence
        response_attention_mask = get_response_mask(
            response_id=responses, eos_token=eos_token_id, dtype=attention_mask.dtype
        )
        full_attention_mask = torch.cat([attention_mask, response_attention_mask], dim=-1)

        # Build position ids for full sequence
        delta_position_id = torch.arange(1, response_length + 1, device=device)
        delta_position_id = delta_position_id.unsqueeze(0).repeat(batch_size, 1)
        # Get the last valid position for each sample
        last_positions = position_ids.gather(1, (attention_mask.sum(dim=1) - 1).unsqueeze(1))
        response_position_ids = last_positions + delta_position_id
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

    # ==================== Weight Update ====================

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

    def shutdown(self):
        """Shutdown the rollout, cleanup resources."""
        if self._zmq_loop_task:
            self._zmq_loop_task.cancel()
        if self._zmq_socket:
            self._zmq_socket.close()
        if self._zmq_context:
            self._zmq_context.term()
        logger.info("vLLMRollout shutdown complete")


# ==================== Async Client for ZeroMQ Server ====================

class vLLMRolloutClient:
    """Client for communicating with vLLMRollout via ZeroMQ.

    Use this client to send async generation requests to vLLMRollout
    when enable_async_server=True.
    """

    def __init__(self, address: str, timeout: float = 60.0):
        """Initialize the client.

        Args:
            address: ZeroMQ address (e.g., "tcp://127.0.0.1:5555")
            timeout: Request timeout in seconds
        """
        self.address = address
        self.timeout = timeout
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REQ)
        self._socket.setsockopt(zmq.RCVTIMEO, int(timeout * 1000))
        self._socket.setsockopt(zmq.SNDTIMEO, int(timeout * 1000))

        if address.startswith("tcp://["):
            self._socket.setsockopt(zmq.IPV6, 1)

        self._socket.connect(address)

    def generate(
        self,
        prompt_token_ids: list[list[int]],
        sampling_params: dict[str, Any],
    ) -> dict[str, Any]:
        """Send a generation request to the server.

        Args:
            prompt_token_ids: List of prompt token id lists
            sampling_params: Sampling parameters dict

        Returns:
            dict containing generated outputs
        """
        message = pickle.dumps(("generate", (prompt_token_ids, sampling_params), {}))
        self._socket.send(message)
        result = pickle.loads(self._socket.recv())

        if isinstance(result, Exception):
            raise result

        return result

    def generate_sequences(self, prompts: DataProto) -> DataProto:
        """Send a batch generation request.

        Args:
            prompts: DataProto containing batch data

        Returns:
            DataProto with generated sequences
        """
        message = pickle.dumps(("generate_sequences", (prompts.to_dict(),), {}))
        self._socket.send(message)
        result = pickle.loads(self._socket.recv())

        if isinstance(result, Exception):
            raise result

        return DataProto.from_dict(result)

    def health_check(self) -> dict[str, Any]:
        """Check server health.

        Returns:
            dict with status information
        """
        message = pickle.dumps(("health_check", (), {}))
        self._socket.send(message)
        result = pickle.loads(self._socket.recv())

        if isinstance(result, Exception):
            raise result

        return result

    def close(self):
        """Close the client connection."""
        self._socket.close()
        self._context.term()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
