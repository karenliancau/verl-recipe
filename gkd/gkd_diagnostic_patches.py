"""
GKD 诊断日志补丁
添加到 recipe/gkd/megatron_workers.py 中以诊断 GPU 分配和模型加载

应用方法:
1. 在 MegatronOnPolicyDistillActorWorker.init_model() 开头添加:
   logger.info(f"[ACTOR] init_model() START | rank={self.local_rank} | CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'NOT SET')}")

2. 在 _build_model_optimizer() 后添加:
   logger.info(f"[ACTOR] model loaded | gpu_memory={torch.cuda.memory_allocated()/1024**3:.1f}GB")

3. 在 MegatronOnPolicyDistillRolloutWorker.init_model() 开头添加:
   logger.info(f"[ROLLOUT] init_model() START | rank={self.local_rank} | CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'NOT SET')}")

4. 在 _build_rollout() 后添加:
   logger.info(f"[ROLLOUT] model loaded | gpu_memory={torch.cuda.memory_allocated()/1024**3:.1f}GB")
"""

import os
import torch
import time
import logging

logger = logging.getLogger(__name__)


def log_gpu_info(stage: str, rank: int = 0):
    """记录 GPU 分配信息"""
    cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', 'NOT SET')
    local_rank = os.environ.get('LOCAL_RANK', 'NOT SET')
    
    if torch.cuda.is_available():
        mem_allocated = torch.cuda.memory_allocated() / 1024**3
        mem_reserved = torch.cuda.memory_reserved() / 1024**3
        
        logger.info(
            f"[{stage}] GPU Info | "
            f"rank={rank} | local_rank={local_rank} | "
            f"CUDA_VISIBLE_DEVICES={cuda_visible} | "
            f"allocated={mem_allocated:.1f}GB | reserved={mem_reserved:.1f}GB"
        )
    else:
        logger.warning(f"[{stage}] CUDA not available!")


# ===== 补丁 1: ActorWorker init_model =====
# 在 recipe/gkd/megatron_workers.py 中找到 MegatronOnPolicyDistillActorWorker.init_model()
# 替换为:

def init_model_actor_with_logging(self):
    """ActorWorker init_model with logging"""
    import time
    start_time = time.time()
    
    # 记录开始
    log_gpu_info("ACTOR_INIT_START", rank=self.local_rank)
    logger.info(f"[ACTOR] Starting model initialization (rank={self.local_rank})")
    
    # 构建模型和优化器
    try:
        self._build_model_optimizer()
        elapsed = time.time() - start_time
        log_gpu_info("ACTOR_MODEL_LOADED", rank=self.local_rank)
        logger.info(f"[ACTOR] Model loaded successfully in {elapsed:.1f}s")
    except Exception as e:
        logger.error(f"[ACTOR] Model loading failed: {e}", exc_info=True)
        raise
    
    # 初始化分布式
    try:
        self._init_distributed()
        logger.info(f"[ACTOR] Distributed initialized (rank={self.local_rank})")
    except Exception as e:
        logger.error(f"[ACTOR] Distributed init failed: {e}", exc_info=True)
        raise
    
    elapsed = time.time() - start_time
    logger.info(f"[ACTOR] init_model() completed in {elapsed:.1f}s")


# ===== 补丁 2: RolloutWorker init_model =====
# 在 recipe/gkd/megatron_workers.py 中找到 MegatronOnPolicyDistillRolloutWorker.init_model()
# 替换为:

def init_model_rollout_with_logging(self):
    """RolloutWorker init_model with logging"""
    import time
    start_time = time.time()
    
    # 记录开始
    log_gpu_info("ROLLOUT_INIT_START", rank=self.local_rank)
    logger.info(f"[ROLLOUT] Starting model initialization (rank={self.local_rank})")
    
    # 构建推理引擎
    try:
        logger.info(f"[ROLLOUT] Building rollout with config: {self.config.rollout}")
        self._build_rollout()
        elapsed = time.time() - start_time
        log_gpu_info("ROLLOUT_MODEL_LOADED", rank=self.local_rank)
        logger.info(f"[ROLLOUT] Rollout engine built successfully in {elapsed:.1f}s")
    except Exception as e:
        logger.error(f"[ROLLOUT] Rollout build failed: {e}", exc_info=True)
        raise
    
    elapsed = time.time() - start_time
    logger.info(f"[ROLLOUT] init_model() completed in {elapsed:.1f}s")


# ===== 补丁 3: sync_rollout_weights with timeout logging =====
# 在 ray_trainer.py 中找到 sync_rollout_weights() 调用
# 替换为带日志的版本:

def sync_rollout_weights_with_logging(rollout_wg, actor_ref, timeout=300):
    """Weight synchronization with detailed logging"""
    import time
    
    logger.info("[SYNC] Starting rollout weight synchronization")
    logger.info(f"[SYNC] Timeout set to {timeout}s")
    
    start_time = time.time()
    try:
        # 调用原始同步函数，但加上超时
        future = rollout_wg.sync_rollout_weights.remote(actor_ref)
        result = ray.get(future, timeout=timeout)
        
        elapsed = time.time() - start_time
        logger.info(f"[SYNC] Weight synchronization completed in {elapsed:.1f}s")
        return result
    
    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"[SYNC] Weight sync failed after {elapsed:.1f}s: {e}", exc_info=True)
        
        # 额外诊断信息
        logger.error(f"[SYNC] Actor ref status: {ray.get(actor_ref.get_status.remote())}")
        logger.error(f"[SYNC] Rollout wg status: {ray.get(rollout_wg.get_status.remote())}")
        
        raise


# ===== 补丁 4: main_gkd.py 中的配置日志 =====

def log_resource_pool_config(resource_pool_spec, config):
    """记录资源池配置"""
    logger.info("="*70)
    logger.info("ResourcePool Configuration:")
    logger.info(f"  Trainer nodes: {config.trainer.nnodes} x {config.trainer.n_gpus_per_node} GPUs")
    logger.info(f"  Rollout nodes: {config.rollout.nnodes} x {config.rollout.n_gpus_per_node} GPUs")
    logger.info(f"  Teacher GPUs: (external process)")
    logger.info(f"  Resource Pool Spec: {resource_pool_spec}")
    logger.info("="*70)


# ===== 使用示例 =====
"""
# 在 main_gkd.py 中，在 create_resource_pool() 前添加:

from diagnose_gpu_allocation import log_resource_pool_config

# ... 创建 resource_pool_spec ...
log_resource_pool_config(resource_pool_spec, config)

# ... 创建资源池 ...

# 在训练循环中添加:
from diagnose_gpu_allocation import log_gpu_info

# 每隔 100 步记录一次
if global_step % 100 == 0:
    log_gpu_info(f"STEP_{global_step}")
"""
