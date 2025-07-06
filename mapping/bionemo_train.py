# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2024 Arc Institute. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2024 Michael Poli. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2024 Stanford University. All rights reserved
# SPDX-License-Identifier: LicenseRef-Apache2
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

# 🔧 在任何导入之前设置NCCL超时环境变量，确保分布式训练有足够时间
import os
os.environ.setdefault("NCCL_TIMEOUT", "7200")  # 2小时
os.environ.setdefault("NCCL_TIMEOUT_S", "7200")  # 某些版本使用这个
os.environ.setdefault("NCCL_ASYNC_ERROR_HANDLING", "1")  # 异步错误处理
os.environ.setdefault("TORCH_NCCL_TRACE_BUFFER_SIZE", "1024")
os.environ.setdefault("TORCH_DISTRIBUTED_TIMEOUT", "7200")
os.environ.setdefault("NCCL_BLOCKING_WAIT", "1")  # 启用阻塞等待，更好的错误报告
os.environ.setdefault("NCCL_DEBUG", "INFO")  # 启用调试信息
# 设置更大的NCCL缓冲区以处理大数据集
os.environ.setdefault("NCCL_BUFFSIZE", "8388608")  # 8MB
os.environ.setdefault("NCCL_NTHREADS", "32")  # 必须是32的倍数

# 尝试设置PyTorch的默认超时
try:
    import torch.distributed as dist
    # 设置默认超时时间（7200秒 = 2小时）
    default_timeout = 7200
    os.environ.setdefault("TORCH_DIST_INIT_BARRIER_TIMEOUT", str(default_timeout))
    print(f"🔧 设置PyTorch分布式超时: {default_timeout}秒")
except ImportError:
    pass

import argparse
from pathlib import Path
from typing import List, Optional
import datetime
# TODO add back support for slurm resilience.
# import nvidia_resiliency_ext.ptl_resiliency as res_module
import torch
from lightning.pytorch.callbacks import LearningRateMonitor, RichModelSummary, Callback
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.optimizer import OptimizerConfig
from nemo import lightning as nl
from nemo.collections import llm
from nemo.collections.llm.gpt.data import MockDataModule, PreTrainingDataModule
from nemo.collections.llm.gpt.data.megatron.hyena.config import parse_dataset_config
from nemo.collections.llm.gpt.data.megatron.hyena.evo2_dataset import Evo2Dataset, Evo2DatasetPadEodLossMask
from nemo.collections.llm.gpt.model.hyena import HYENA_MODEL_OPTIONS
from nemo.collections.llm.recipes.tp_overlap_configs.userbuffers import (
    userbuffers_bf16_h100_h8192_tp4_mbs1_seqlen8192,
    userbuffers_fp8_h100_h8192_tp4_mbs1_seqlen8192,
)
from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer
from nemo.lightning.pytorch import callbacks as nl_callbacks
from nemo.lightning.pytorch.callbacks import ModelCheckpoint
from nemo.lightning.pytorch.callbacks.flops_callback import FLOPsMeasurementCallback
from nemo.lightning.pytorch.callbacks.megatron_comm_overlap import MegatronCommOverlapCallback
from nemo.lightning.pytorch.optim import CosineAnnealingScheduler
from nemo.lightning.pytorch.optim.megatron import MegatronOptimizerModule
from nemo.lightning.pytorch.strategies.utils import RestoreConfig
from nemo.utils.exp_manager import TimingCallback

# LoRA imports
# 注释掉暂时不使用的导入，使用自定义 LoRA 实现
# from nemo.collections.nlp.modules.common.megatron.adapters.parallel_adapters import (
#     ParallelLinearAdapter,
#     AdapterName,
# )
# from nemo.collections.nlp.parts.utils_funcs import get_last_rank

from bionemo.llm.utils.datamodule_utils import infer_global_batch_size
from bionemo.llm.utils.logger_utils import WandbConfig, setup_nemo_lightning_logger


torch._dynamo.config.suppress_errors = True


class LoRACallback(Callback):
    """在训练开始时应用 LoRA 的回调"""
    
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.applied = False
    
    def on_train_start(self, trainer, pl_module):
        """训练开始时应用 LoRA"""
        # 使用 pl_module 获取完全初始化的模型
        actual_model = pl_module if pl_module is not None else self.model
        
        if hasattr(self.model, '_lora_config') and not self.applied:
            print("\n🔧 开始在训练时应用 LoRA...")
            lora_config = self.model._lora_config
            
            # 重新尝试应用 LoRA，这次使用完全初始化的模型
            try:
                print("检查训练时的模型结构...")
                
                # 打印训练时的模型结构
                all_modules = list(actual_model.named_modules())
                print(f"训练开始时模型有 {len(all_modules)} 个模块")
                
                linear_layers = []
                for name, module in actual_model.named_modules():
                    if hasattr(module, 'weight') and module.weight is not None and len(module.weight.shape) == 2:
                        linear_layers.append((name, module))
                
                print(f"找到 {len(linear_layers)} 个线性层:")
                for i, (name, module) in enumerate(linear_layers[:5]):  # 显示前5个
                    print(f"  {i+1}. {name} (形状: {module.weight.shape})")
                
                if linear_layers:
                    # 现在应用 LoRA
                    result = apply_lora_to_model(actual_model, lora_config)
                    self.applied = True
                    print("✅ LoRA 在训练开始时成功应用")
                    
                    # 重新计算参数统计
                    total_params = sum(p.numel() for p in actual_model.parameters())
                    trainable_params = sum(p.numel() for p in actual_model.parameters() if p.requires_grad)
                    
                    print(f"📊 更新后的参数统计:")
                    print(f"总参数数量: {total_params:,}")
                    print(f"可训练参数数量: {trainable_params:,}")
                    
                    if total_params > 0:
                        print(f"LoRA 可训练参数占比: {100 * trainable_params / total_params:.2f}%")
                else:
                    print("⚠️  训练开始时仍然没有找到线性层")
                    
            except Exception as e:
                print(f"❌ 在训练开始时应用 LoRA 失败: {e}")
                import traceback
                traceback.print_exc()


def apply_lora_to_model(model, lora_config):
    """为模型应用 LoRA 适配器"""
    import re
    import torch.nn as nn
    
    class LoRALayer(nn.Module):
        """简单的 LoRA 层实现"""
        def __init__(self, original_layer, rank=16, alpha=32, dropout=0.1):
            super().__init__()
            self.original_layer = original_layer
            self.rank = rank
            self.alpha = alpha
            self.scaling = alpha / rank
            
            # 获取原始层的维度
            if hasattr(original_layer, 'weight') and original_layer.weight is not None:
                out_features, in_features = original_layer.weight.shape
                # 获取原始层的设备和数据类型
                device = original_layer.weight.device
                dtype = original_layer.weight.dtype
            else:
                raise ValueError("原始层必须有有效的 weight 属性")
            
            # LoRA 矩阵 A 和 B，确保在正确的设备上且有正确的数据类型
            self.lora_A = nn.Parameter(
                torch.randn(rank, in_features, device=device, dtype=dtype) * (1 / rank)
            )
            self.lora_B = nn.Parameter(
                torch.zeros(out_features, rank, device=device, dtype=dtype)
            )
            self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
            
            # 冻结原始参数
            for param in original_layer.parameters():
                param.requires_grad = False
                
        def forward(self, x):
            # 原始前向传播
            result = self.original_layer(x)
            
            # LoRA 前向传播 - 确保所有操作在同一设备上
            x_dropped = self.dropout(x)
            # 使用更高效的矩阵乘法顺序：(x @ A.T) @ B.T
            lora_result = (x_dropped @ self.lora_A.T) @ self.lora_B.T * self.scaling
            
            # 检查原始层的返回值格式
            if isinstance(result, tuple):
                # 如果原始层返回 tuple（通常是 (output, bias)），则只修改第一个元素
                output_tensor = result[0] + lora_result
                return (output_tensor,) + result[1:]  # 保持其他元素不变
            else:
                # 如果返回单个 tensor
                return result + lora_result
    
    # 打印模型结构以便调试
    print("模型结构调试信息:")
    print("-" * 50)
    all_modules = list(model.named_modules())
    print(f"模型总共有 {len(all_modules)} 个模块")
    
    # 对于 HyenaModel，我们需要访问其内部的 module 属性
    actual_model = model
    if hasattr(model, 'module'):
        print("检测到 HyenaModel，访问内部 module...")
        actual_model = model.module
        all_modules = list(actual_model.named_modules())
        print(f"内部模型总共有 {len(all_modules)} 个模块")
    
    # 如果仍然只有1个模块，尝试其他属性
    if len(all_modules) <= 1:
        print("尝试访问模型的其他属性...")
        for attr_name in ['model', 'transformer', 'decoder', 'encoder', 'backbone']:
            if hasattr(actual_model, attr_name):
                attr_obj = getattr(actual_model, attr_name)
                if hasattr(attr_obj, 'named_modules'):
                    print(f"找到属性 {attr_name}，检查其模块...")
                    attr_modules = list(attr_obj.named_modules())
                    if len(attr_modules) > len(all_modules):
                        actual_model = attr_obj
                        all_modules = attr_modules
                        print(f"使用 {attr_name}，模块数量: {len(all_modules)}")
                        break
    
    # 打印前20个模块名称以便调试
    print("模块结构:")
    for i, (name, module) in enumerate(all_modules[:20]):
        module_type = type(module).__name__
        if hasattr(module, 'weight') and module.weight is not None and len(module.weight.shape) == 2:
            print(f"  {i+1:2d}. {name} ({module_type}) - 线性层: {module.weight.shape}")
        else:
            print(f"  {i+1:2d}. {name} ({module_type})")
    
    if len(all_modules) > 20:
        print(f"  ... 还有 {len(all_modules) - 20} 个模块")
    
    # LoRA 目标层模式
    target_patterns = lora_config.get('target_modules', [])
    if not target_patterns:
        # 默认目标层：注意力层的 qkv 投影
        target_patterns = [r'.*self_attention\.linear_qkv$']
    
    print(f"\n目标模式: {target_patterns}")
    
    # 查找包含 "attention" 和 "linear" 的层
    attention_layers = []
    linear_layers = []
    qkv_layers = []
    all_linear_layers = []
    
    for name, module in actual_model.named_modules():
        if hasattr(module, 'weight') and module.weight is not None and len(module.weight.shape) == 2:
            all_linear_layers.append((name, module))
            if 'attention' in name.lower():
                attention_layers.append(name)
            if 'linear' in name.lower():
                linear_layers.append(name)
            if 'qkv' in name.lower():
                qkv_layers.append(name)
    
    print(f"\n所有线性层 ({len(all_linear_layers)}):")
    for name, module in all_linear_layers[:10]:  # 只显示前10个
        print(f"  - {name} (形状: {module.weight.shape})")
    
    print(f"\n找到的注意力相关层 ({len(attention_layers)}):")
    for layer in attention_layers[:10]:  # 只显示前10个
        print(f"  - {layer}")
    
    print(f"\n找到的线性层 ({len(linear_layers)}):")
    for layer in linear_layers[:10]:  # 只显示前10个
        print(f"  - {layer}")
        
    print(f"\n找到的QKV层 ({len(qkv_layers)}):")
    for layer in qkv_layers:
        print(f"  - {layer}")
    
    lora_rank = lora_config.get('rank', 16)
    lora_alpha = lora_config.get('alpha', 32)
    lora_dropout = lora_config.get('dropout', 0.1)
    
    modified_layers = []
    
    # 如果没有找到任何线性层，返回警告
    if not all_linear_layers:
        print("⚠️  没有找到任何线性层！模型可能还没有完全初始化。")
        print("这在 NeMo/Megatron 框架中是正常的，模型结构会在训练开始时初始化。")
        # 在这种情况下，我们先保存配置，等待模型完全初始化后再应用
        model._lora_config = lora_config  # 保存配置到模型上
        print("LoRA 配置已保存，将在模型完全初始化后应用。")
        return model
    
    # 如果没有找到匹配的特定层，尝试匹配第一个线性层作为测试
    if not qkv_layers and not attention_layers:
        print("\n⚠️  没有找到预期的注意力层，尝试匹配第一个线性层...")
        if all_linear_layers:
            # 尝试匹配包含特定关键词的层
            for name, module in all_linear_layers:
                if any(keyword in name.lower() for keyword in ['qkv', 'query', 'key', 'value', 'attention', 'attn']):
                    target_patterns = [re.escape(name)]
                    print(f"找到可能的注意力层: {name}")
                    break
            else:
                # 如果没有找到明显的注意力层，使用第一个大的线性层
                target_name, target_module = all_linear_layers[0]
                target_patterns = [re.escape(target_name)]
                print(f"使用第一个线性层作为测试: {target_name} (形状: {target_module.weight.shape})")
    
    # 遍历模型的所有命名模块
    for name, module in actual_model.named_modules():
        # 检查是否匹配目标层模式
        for pattern in target_patterns:
            if re.search(pattern, name):  # 使用 search 而不是 match
                # 为匹配的线性层添加 LoRA 适配器
                if hasattr(module, 'weight') and module.weight is not None and len(module.weight.shape) == 2:
                    try:
                        print(f"正在为层 {name} 添加 LoRA (形状: {module.weight.shape})")
                        
                        # 创建 LoRA 层
                        lora_layer = LoRALayer(
                            module,
                            rank=lora_rank,
                            alpha=lora_alpha,
                            dropout=lora_dropout
                        )
                        
                        # 将原始模块替换为 LoRA 层
                        parent_name = '.'.join(name.split('.')[:-1])
                        child_name = name.split('.')[-1]
                        
                        if parent_name:
                            parent_module = actual_model.get_submodule(parent_name)
                            setattr(parent_module, child_name, lora_layer)
                        else:
                            setattr(actual_model, child_name, lora_layer)
                            
                        modified_layers.append(name)
                        print(f"✓ 已为层 {name} 添加 LoRA 适配器 (rank={lora_rank}, alpha={lora_alpha})")
                        
                        # 只修改第一个匹配的层作为测试
                        if len(modified_layers) >= 1:
                            break
                        
                    except Exception as e:
                        print(f"❌ 无法为层 {name} 添加 LoRA 适配器: {e}")
                break
        
        # 如果已经修改了一些层，就停止
        if len(modified_layers) >= 1:
            break
    
    # 确保只有 LoRA 参数被训练
    lora_param_count = 0
    total_param_count = 0
    
    for name, param in model.named_parameters():  # 使用原始 model
        total_param_count += param.numel()
        if 'lora' in name.lower():
            param.requires_grad = True
            lora_param_count += param.numel()
            print(f"LoRA 参数 {name} 将被训练 (大小: {param.numel()})")
        else:
            param.requires_grad = False
    
    print(f"\n总共修改了 {len(modified_layers)} 个层: {modified_layers}")
    print(f"LoRA 参数数量: {lora_param_count}")
    print(f"总参数数量: {total_param_count}")
    
    return model


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse arguments for Evo2 model training."""
    parser = argparse.ArgumentParser(
        description="Train a Hyena model using NeMo 2.0.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    data_group = parser.add_mutually_exclusive_group(required=True)

    data_group.add_argument(
        "-d",
        "--dataset-config",
        type=str,
        help="Path to the blended / weighted training dataset configuration YAML.",
    )
    data_group.add_argument(
        "--mock-data",
        action="store_true",
        help="Train with Mock data (for testing/debugging), either set this or provide a dataset config.",
    )

    parser.add_argument(
        "--dataset-dir",
        type=str,
        help="Absolute path to the dataset directory. Defaults to using the absolute or relative paths (dataset_prefix) specified in the dataset config YAML.",
    )

    parser.add_argument("--num-nodes", type=int, default=1, help="Number of nodes to use for training, defaults to 1.")
    parser.add_argument("--devices", type=int, default=1, help="Number of devices to use for training, defaults to 1.")
    parser.add_argument("--seq-length", type=int, default=8192, help="Training sequence length")
    parser.add_argument(
        "--tensor-parallel-size", type=int, default=1, help="Order of tensor parallelism. Defaults to 1."
    )
    parser.add_argument(
        "--pipeline-model-parallel-size", type=int, default=1, help="Order of pipeline parallelism. Defaults to 1."
    )
    parser.add_argument(
        "--context-parallel-size", type=int, default=1, help="Order of context parallelism. Defaults to 1."
    )
    parser.add_argument(
        "--create-tensorboard-logger", action="store_true", default=False, help="Create a tensorboard logger."
    )
    parser.add_argument("--wandb-entity", type=str, default=None, help="The team posting this run")
    parser.add_argument("--wandb-project", type=str, default=None, help="Wandb project name ")
    parser.add_argument("--wandb-tags", nargs="+", type=str, default=None, help="Tags associated with this run")
    parser.add_argument(
        "--wandb-group", type=str, default=None, help="A unique string shared by all runs in a given group"
    )
    parser.add_argument(
        "--wandb-job-type",
        type=str,
        default=None,
        help="A unique string representing a type of run, which is useful when you're grouping runs together into larger experiments using group.",
    )
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default=None,
        help="A unique string representing the name of the wandb run. If not provided, the name will be generated from the model and training specifications.",
    )

    parser.add_argument(
        "--wandb-id", type=str, default=None, help="Sets the version, mainly used to resume a previous run"
    )
    parser.add_argument(
        "--wandb-anonymous", action="store_true", help="Enable or explicitly disable anonymous logging"
    )
    parser.add_argument(
        "--wandb-log-model", action="store_true", help="Save checkpoints in wandb dir to upload on W&B servers"
    )
    parser.add_argument("--wandb-offline", action="store_true", help="Use wandb in offline mode")
    parser.add_argument("--sequence-parallel", action="store_true", help="Set to enable sequence parallelism.")
    parser.add_argument("--fp8", action="store_true", help="Set to enable FP8")
    parser.add_argument("--micro-batch-size", type=int, default=1, help="Micro-batch size for data-parallel training.")
    parser.add_argument(
        "--global-batch-size",
        type=int,
        default=None,
        help="Global batch size for training. If set to None, infer it from the TP, CP, and PP parameters.",
    )
    parser.add_argument(
        "--grad-acc-batches", type=int, default=1, help="Number of batches to accumulate gradients over."
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        help="Number of training optimizer update steps. This controls the total number of steps as well as the "
        "shape of the learning rate curve.",
        default=500000,
    )
    parser.add_argument(
        "--early-stop-on-step",
        type=int,
        help="Stop training on this step, if set. This may be useful for testing or debugging purposes.",
    )
    parser.add_argument(
        "--val-check-interval", type=int, help="Number of steps between validation measurements and model checkpoints."
    )
    parser.add_argument("--grad-reduce-in-fp32", action="store_true", default=False, help="Gradient reduce in FP32.")
    parser.add_argument(
        "--fp8-wgrad",
        action="store_true",
        default=False,
        help="Faster option that is maybe less accurate (TBD) when using fp8.",
    )
    parser.add_argument("--use-megatron-comm-overlap-llama3-8k", action="store_true", default=False)
    parser.add_argument(
        "--tp-comm-overlap-backend",
        type=str,
        choices=["nccl", "mpi", "gloo"],
        default="nccl",
        help="TP communication backend to use. Defaults to 'nccl'.",
    )
    parser.add_argument("--align-param-gather", action="store_true", default=False)
    # parser.add_argument("--straggler-detection", action="store_true", default=False)
    parser.add_argument(
        "--model-size",
        type=str,
        choices=sorted(HYENA_MODEL_OPTIONS.keys()),
        default="7b",
        help="Model architecture to use, choose between 7b, 40b, or test (a sub-model of 4 layers, less than 1B "
        "parameters). '_arc_1m' models have GLU / FFN dimensions that support 1M context length when trained "
        "with TP<=8.",
    )
    parser.add_argument(
        "--add-bias-output",
        action="store_true",
        default=False,
        help="Add bias to the output layer to enable learning a simple prior.",
    )
    parser.add_argument(
        "--result-dir", type=Path, required=False, default=Path("./results"), help="Path to the result directory."
    )
    parser.add_argument("--experiment-name", type=str, required=False, default="evo2", help="Name of the experiment.")

    parser.add_argument(
        "--limit-val-batches",
        type=int,
        default=20,
        help="Number of validation steps",
    )
    parser.add_argument(
        "--log-every-n-steps",
        type=int,
        default=1,
        required=False,
        help="Number of steps between logging.",
    )
    parser.add_argument(
        "--ckpt-dir",
        type=str,
        default=None,
        help="Directory to restore an initial checkpoint from. Use this for supervised fine-tuning.",
    )
    parser.add_argument("--wd", type=float, default=0.01, help="Weight decay for optimizer.")
    parser.add_argument(
        "--restore-optimizer-from-ckpt",
        action="store_true",
        help="Restore optimizer state from initial checkpoint. Defaults to False.",
    )
    parser.add_argument(
        "--no-average-in-collective",
        action="store_true",
        default=False,
        help="Avaerage optimizer state in collective rather than dividing by dp size and summing.",
    )
    parser.add_argument("--seed", type=int, default=1234, help="Set random seed for training.")
    parser.add_argument("--workers", type=int, default=8, help="Number of workers to use for data loading.")
    parser.add_argument(
        "--gc-interval",
        type=int,
        default=0,
        help="Set to a value > 0 if you want to synchronize garbage collection, will do gc every gc-interval steps.",
    )
    parser.add_argument(
        "--enable-preemption",
        action="store_true",
        default=True,
        help="Enable preemption hooks. If enabled this will save a checkpoint whenever slurm exits.",
    )
    parser.add_argument(
        "--ckpt-async-save",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--ckpt-format",
        type=str,
        choices=["torch_dist", "zarr"],
        default="torch_dist",
        help="Specify checkpoint format to use. Defaults to 'torch_dist', as 'zarr' is deprecated. Only use if "
        "resuming training from a zarr checkpoint.",
    )
    parser.add_argument(
        "--eod-pad-in-loss-mask",
        action="store_true",
        default=False,
        help="Do not predict EOD/Pad tokens (typical default, but not default in original evo2).",
    )
    parser.add_argument(
        "--cross-entropy-loss-fusion",
        action="store_true",
        default=False,
        help="Use the faster, but maybe less accurate fused form of cross entropy, "
        "which also has bf16 grads internally.",
    )
    parser.add_argument(
        "--no-fp32-residual-connection",
        action="store_true",
        default=False,
        help="If set, turn off fp32 residual connections which may be faster but may impact accuracy.",
    )
    parser.add_argument(
        "--debug-ddp-parity-freq",
        type=int,
        default=0,
        help="Set to value > 0 to debug DDP weight parity between ranks.",
    )
    parser.add_argument(
        "--hybrid-override-pattern",
        type=str,
        help="Override the hybrid override pattern in the config (specifies hyena layer ordering and type).",
    )
    parser.add_argument(
        "--num-layers", type=int, help="If set, override the number of layers specified in the requested config."
    )
    parser.add_argument(
        "--create-tflops-callback",
        action="store_true",
        default=False,
        help="Enable tflops calculation callback for Hyena / Evo2. Defaults to False.",
    )
    parser.add_argument(
        "--log-parameters-and-shapes",
        action="store_true",
        default=False,
        help="Log training parameters shapes and dtypes for debugging.",
    )
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate.")
    parser.add_argument("--min-lr", type=float, default=3e-5, help="Min learning rate in cosine annealing.")
    parser.add_argument("--warmup-steps", type=int, default=2500, help="Number of warmup steps in cosine annealing")
    # NSYS profiling/tooling arguments
    parser.add_argument(
        "--nsys-profiling",
        action="store_true",
        default=False,
        help="Enable targeted `nsys` profiling on the training loop for a defined step range. To actually get profiling"
        " output you must run the whole program with `nsys`. For example: "
        " `nsys profile -s none -o output_report_name -t cuda,nvtx --force-overwrite true "
        "--capture-range=cudaProfilerApi --capture-range-end=stop  [regular python command here]`",
    )
    # start, end, rank
    parser.add_argument(
        "--nsys-start-step",
        type=int,
        required=False,
        default=0,
        help="Start nsys profiling after this step.",
    )
    parser.add_argument(
        "--nsys-end-step",
        type=int,
        required=False,
        help="End nsys profiling after this step.",
    )
    parser.add_argument(
        "--no-renormalize-loss",
        action="store_true",
        default=False,
        help="Do not renormalize the loss weights.",
    )
    # rank as list of integers
    parser.add_argument(
        "--nsys-ranks",
        type=int,
        nargs="+",
        required=False,
        default=[0],
        help="Enable nsys profiling for these ranks.",
    )
    parser.add_argument(
        "--activation-checkpoint-recompute-num-layers",
        type=int,
        help="If set, override the default value set in the config.",
    )
    parser.add_argument(
        "--disable-checkpointing",
        action="store_false",
        default=True,
        dest="create_checkpoint_callback",
        help="Disable creating a ModelCheckpoint callback.",
    )
    parser.add_argument(
        "--clip-grad",
        type=float,
        default=1.0,
        help="Grad clip value. Note that when using DDP this may need to be inflated.",
    )
    parser.add_argument(
        "--seq-len-interpolation-factor",
        type=float,
        help="Adjusts the linear scaling of ROPE (Rotary Position Embedding) for context extension. "
        "Set this factor relative to your base context length e.g., for an original context length of 8192 and "
        "an extended context length of 524288, use 524288/8192 = 64.",
    )
    parser.add_argument(
        "--overlap-param-gather",
        action="store_true",
        default=False,
        help="Overlap the parameter gather with the optimizer step. This is currently disabled due to a NeMo bug "
        "when using DDP. Making this an option defaulting to False is a temporary solution until the bug is fixed.",
    )
    parser.add_argument(
        "--overlap-grad-reduce",
        action="store_true",
        default=False,
        help="Overlap the gradient reduce with the optimizer step.",
    )
    parser.add_argument(
        "--hidden-dropout",
        type=float,
        default=0.0,
        help="Dropout probability for the hyena layers",
    )
    parser.add_argument(
        "--attention-dropout",
        type=float,
        default=0.0,
        help="Dropout probability for the attention layers.",
    )
    recompute_group = parser.add_mutually_exclusive_group(required=False)
    recompute_group.add_argument("--no-activation-checkpointing", action="store_true", default=False)
    recompute_group.add_argument("--selective-activation-checkpointing", action="store_true", default=False)

    # LoRA 微调参数
    parser.add_argument(
        "--enable-lora",
        action="store_true",
        default=True,
        help="启用 LoRA 微调"
    )
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=16,
        help="LoRA 适配器的秩 (rank)"
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=32,
        help="LoRA 缩放参数 alpha"
    )
    parser.add_argument(
        "--lora-dropout",
        type=float,
        default=0.1,
        help="LoRA dropout 概率"
    )
    parser.add_argument(
        "--lora-target-modules",
        nargs="+",
        type=str,
        default=["module.decoder.layers.17.self_attention.linear_qkv"],
        help="LoRA 目标模块的正则表达式模式列表"
    )

    # 分布式训练策略选项
    parser.add_argument(
        "--use-data-parallel",
        action="store_true",
        default=False,
        help="使用数据并行而不是张量并行（可能更稳定，避免NCCL复杂通信）"
    )

    return parser.parse_args(args=args)


def train(args: argparse.Namespace) -> nl.Trainer:
    """Main function to run Evo2 training."""
    # 设置NCCL超时时间为2小时，确保数据集构建有足够时间
    import os
    os.environ["NCCL_TIMEOUT"] = "7200"  # 2小时
    os.environ["TORCH_NCCL_TRACE_BUFFER_SIZE"] = "1024"  # 启用调试信息
    
    # 如果使用分布式训练，显式设置PyTorch分布式超时
    if args.devices > 1:
        import torch.distributed as dist
        # 这个需要在init_process_group之前设置
        os.environ["TORCH_DISTRIBUTED_TIMEOUT"] = "7200"
        
        # 如果启用数据并行模式，调整并行配置
        if args.use_data_parallel:
            print(f"🔄 使用数据并行模式，设备数量: {args.devices}")
            # 强制使用数据并行，禁用张量并行和管道并行
            args.tensor_parallel_size = 1
            args.pipeline_model_parallel_size = 1
            args.context_parallel_size = 1
            print(f"调整并行配置: TP={args.tensor_parallel_size}, PP={args.pipeline_model_parallel_size}, CP={args.context_parallel_size}")
    
    # Instantiate tokenizer.
    tokenizer = get_nmt_tokenizer(
        "byte-level",
    )

    # Infer global batch size.
    global_batch_size = args.global_batch_size
    if global_batch_size is None:
        global_batch_size = infer_global_batch_size(
            micro_batch_size=args.micro_batch_size,
            num_nodes=args.num_nodes,
            devices=args.devices,
            accumulate_grad_batches=args.grad_acc_batches,
            tensor_model_parallel_size=args.tensor_parallel_size,
            pipeline_model_parallel_size=args.pipeline_model_parallel_size,
            context_model_parallel_size=args.context_parallel_size,
        )
    if args.mock_data:
        data_module = MockDataModule(
            seq_length=args.seq_length,
            micro_batch_size=args.micro_batch_size,
            global_batch_size=global_batch_size,
            num_workers=args.workers,
            tokenizer=tokenizer,
        )
    else:
        blended_dataset_config = parse_dataset_config(
            dataset_config_path=args.dataset_config, dataset_path=args.dataset_dir
        )
        dataset_cls = Evo2DatasetPadEodLossMask if args.eod_pad_in_loss_mask else Evo2Dataset
        # Instantiate pre-training module.
        data_module = PreTrainingDataModule(
            paths=blended_dataset_config,
            dataset_cls=dataset_cls,
            seq_length=args.seq_length,
            micro_batch_size=args.micro_batch_size,
            global_batch_size=global_batch_size,
            seed=args.seed,
            num_workers=args.workers,
            tokenizer=tokenizer,
            eod_mask_loss=args.eod_pad_in_loss_mask,
        )

    if args.no_activation_checkpointing:
        activation_checkpointing_args = {
            "recompute_granularity": None,
            "recompute_method": None,
            "recompute_num_layers": None,
        }
    elif args.selective_activation_checkpointing:
        activation_checkpointing_args = {
            "recompute_granularity": "selective",
            "recompute_method": None,
            "recompute_num_layers": None,
        }
    else:
        if args.activation_checkpoint_recompute_num_layers is not None:
            activation_checkpointing_args = {
                "recompute_num_layers": args.activation_checkpoint_recompute_num_layers,
            }
        else:
            activation_checkpointing_args = {}

    # Retrieve model config.
    config_modifiers_init = {
        "tp_comm_overlap": args.use_megatron_comm_overlap_llama3_8k,
        "seq_length": args.seq_length,
        "hidden_dropout": args.hidden_dropout,
        "attention_dropout": args.attention_dropout,
        "to_upper": "weighted" if args.no_renormalize_loss else "normalized_weighted",
        "distribute_saved_activations": False if args.sequence_parallel else True,
        "cross_entropy_loss_fusion": args.cross_entropy_loss_fusion,
        "fp32_residual_connection": not args.no_fp32_residual_connection,
        "add_bias_output": args.add_bias_output,
        **activation_checkpointing_args,
    }
    if args.hybrid_override_pattern:
        config_modifiers_init["hybrid_override_pattern"] = args.hybrid_override_pattern
    if args.num_layers:
        config_modifiers_init["num_layers"] = args.num_layers

    if args.model_size not in HYENA_MODEL_OPTIONS:
        raise ValueError(f"Invalid model size: {args.model_size}")
    evo2_config = HYENA_MODEL_OPTIONS[args.model_size](**config_modifiers_init)

    # Instantiate model.
    model = llm.HyenaModel(evo2_config, tokenizer=data_module.tokenizer)

    # 应用 LoRA 微调（如果启用）
    if args.enable_lora:
        print("启用 LoRA 微调...")
        lora_config = {
            'target_modules': args.lora_target_modules,
            'rank': args.lora_rank,
            'alpha': args.lora_alpha,
            'dropout': args.lora_dropout,
        }
        model = apply_lora_to_model(model, lora_config)
        
        # 打印可训练参数数量（添加安全检查）
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"总参数数量: {total_params:,}")
        print(f"可训练参数数量: {trainable_params:,}")
        
        if total_params > 0:
            print(f"LoRA 可训练参数占比: {100 * trainable_params / total_params:.2f}%")
        else:
            print("⚠️  警告: 模型总参数数量为0，可能模型初始化有问题")
        
        if trainable_params == 0:
            print("⚠️  警告: 没有可训练的参数，LoRA 可能没有正确应用")
            # 如果模型保存了 LoRA 配置，说明需要延迟应用
            if hasattr(model, '_lora_config'):
                print("LoRA 将在训练开始时自动应用")

    # Setup callbacks.
    callbacks = [
        RichModelSummary(max_depth=4),
        LearningRateMonitor(),
        TimingCallback(),
    ]

    # 如果启用了 LoRA 且需要延迟应用，添加 LoRA 回调
    if args.enable_lora and hasattr(model, '_lora_config'):
        lora_callback = LoRACallback(model)
        callbacks.append(lora_callback)
        print("已添加 LoRA 回调，将在训练开始时应用 LoRA")

    if args.enable_preemption:
        callbacks.append(nl_callbacks.PreemptionCallback())
    if args.debug_ddp_parity_freq > 0:
        callbacks.append(nl_callbacks.DdpParityChecker(interval=args.debug_ddp_parity_freq))
    if args.log_parameters_and_shapes:
        callbacks.append(nl_callbacks.ParameterDebugger())
    if args.create_tflops_callback:
        # Add callback that logs the tera-FLOPS per second per GPU during training.
        flop_meas_callback = FLOPsMeasurementCallback(
            evo2_config,
            data_module,
            "hyena",
        )
        callbacks.append(flop_meas_callback)

    # TODO(@cye): Add this back when it works with 24.12.
    # if args.straggler_detection:
    #     callbacks.append(
    #         res_module.StragglerDetectionCallback(
    #             report_time_interval=300,
    #             calc_relative_gpu_perf=True,
    #             calc_individual_gpu_perf=True,
    #             num_gpu_perf_scores_to_print=5,
    #             gpu_relative_perf_threshold=0.7,
    #             gpu_individual_perf_threshold=0.7,
    #             stop_if_detected=True,
    #             enable_ptl_logging=True,
    #         )
    #     )
    if args.use_megatron_comm_overlap_llama3_8k:
        # Pick the floating point appropriate config.
        if args.fp8:
            tp_comm_overlap_cfg = userbuffers_fp8_h100_h8192_tp4_mbs1_seqlen8192
        else:
            tp_comm_overlap_cfg = userbuffers_bf16_h100_h8192_tp4_mbs1_seqlen8192
        callbacks.append(
            MegatronCommOverlapCallback(
                tp_comm_overlap=evo2_config.tp_comm_overlap,
                tp_comm_overlap_cfg=tp_comm_overlap_cfg,
                tp_comm_bootstrap_backend=args.tp_comm_overlap_backend,
                wgrad_deferral_limit=22,  # default from NeMo
                overlap_param_gather_with_optimizer_step=False,  # Currently disabled due to an issue with checkpointing.
                align_param_gather=args.align_param_gather,
            )
        )

    if args.gc_interval > 0:
        callbacks.append(
            nl_callbacks.GarbageCollectionCallback(
                gc_interval_train=args.gc_interval, gc_interval_val=args.gc_interval
            )
        )
    if args.nsys_profiling:
        if args.nsys_end_step is None:
            nsys_end_step = args.max_steps
        else:
            nsys_end_step = args.nsys_end_step
        callbacks.append(
            nl_callbacks.NsysCallback(
                start_step=args.nsys_start_step, end_step=nsys_end_step, ranks=args.nsys_ranks, gen_shape=True
            )
        )

    wandb_run_name = (
        f"evo2-size-{args.model_size}-TP{args.tensor_parallel_size}-"
        f"PP{args.pipeline_model_parallel_size}-CP{args.context_parallel_size}"
        f"-GBS{global_batch_size}-MBS{args.micro_batch_size}-SkipLossRenorm{args.no_renormalize_loss}"
        f"-NOAC{args.no_activation_checkpointing}-SELAC{args.selective_activation_checkpointing}"
        f"-ACRNL{evo2_config.recompute_num_layers}"
        f"-PAT{evo2_config.hybrid_override_pattern}"
        f"-F32R{evo2_config.fp32_residual_connection}"
        f"-FCE{evo2_config.cross_entropy_loss_fusion}"
        f"-AIC{not args.no_average_in_collective}"
        f"-PEOD{args.eod_pad_in_loss_mask}"
        f"-BO{args.add_bias_output}"
        f"-GCLP{args.clip_grad}"
        f"-HDO{args.hidden_dropout}"
        f"-ADO{args.attention_dropout}"
        f"-LR{args.lr}-MINLR{args.min_lr}-WUSTEPS{args.warmup_steps}-WD{args.wd}"
        f"-GRFP32{args.grad_reduce_in_fp32}-FP8WG{args.fp8_wgrad and args.fp8}"
        f"-OGR{args.overlap_grad_reduce}-OPG{args.overlap_param_gather}"
        f"-NODES{args.num_nodes}-FP8{args.fp8}"
        f"-LORA{args.enable_lora}"
        f"{f'-LORA_R{args.lora_rank}-LORA_A{args.lora_alpha}' if args.enable_lora else ''}"
    )

    wandb_config: Optional[WandbConfig] = (
        None
        if args.wandb_project is None
        else WandbConfig(
            offline=args.wandb_offline,
            project=args.wandb_project,
            name=args.wandb_run_name if args.wandb_run_name is not None else wandb_run_name,
            entity=args.wandb_entity,
            tags=args.wandb_tags,
            group=args.wandb_group,
            job_type=args.wandb_job_type,
            id=args.wandb_id,
            anonymous=args.wandb_anonymous,
            log_model=args.wandb_log_model,
        )
    )
    nemo_logger = setup_nemo_lightning_logger(
        root_dir=args.result_dir,
        name=args.experiment_name,
        initialize_tensorboard_logger=args.create_tensorboard_logger,
        wandb_config=wandb_config,
    )

    if args.create_checkpoint_callback:
        checkpoint_path = str(Path(nemo_logger.save_dir) / "checkpoints")
        checkpoint_callback = ModelCheckpoint(
            every_n_train_steps=args.val_check_interval,
            dirpath=checkpoint_path,
            save_top_k=5,
            always_save_context=True,
            save_optim_on_train_end=True,
            save_context_on_train_end=True,
        )
        callbacks.append(checkpoint_callback)

        auto_resume = nl.AutoResume(
            resume_if_exists=True,
            resume_ignore_no_checkpoint=True,
            resume_past_end=False,
            resume_from_directory=checkpoint_path,
            restore_config=(
                RestoreConfig(
                    path=args.ckpt_dir,
                    load_model_state=True,
                    load_optim_state=args.restore_optimizer_from_ckpt,
                )
                if args.ckpt_dir
                else None
            ),
        )
    else:
        auto_resume = None

    ddp: DistributedDataParallelConfig = DistributedDataParallelConfig(
        check_for_nan_in_grad=True,
        overlap_grad_reduce=args.overlap_grad_reduce,
        overlap_param_gather=args.overlap_param_gather,  # Verify that this works using
        grad_reduce_in_fp32=args.grad_reduce_in_fp32,
        align_param_gather=args.align_param_gather,
        average_in_collective=not args.no_average_in_collective,
    )
    # Initialize Megatron Strategy and Trainer.
    strategy = nl.MegatronStrategy(
        ddp=ddp,
        tensor_model_parallel_size=args.tensor_parallel_size,
        pipeline_model_parallel_size=args.pipeline_model_parallel_size,
        context_parallel_size=args.context_parallel_size,
        pipeline_dtype=torch.bfloat16,
        sequence_parallel=args.sequence_parallel,
        ckpt_load_optimizer=True,
        ckpt_save_optimizer=True,
        ckpt_async_save=args.ckpt_async_save,
        save_ckpt_format=args.ckpt_format,
        ckpt_load_strictness="log_all",  # or rebasing to https://github.com/NVIDIA/NeMo/pull/11988/files#diff-7667eae242a8ef776bff78cd08e79bc81df4896a450f0a781f6ed317a3dfb7ffR139
        timeout=datetime.timedelta(hours=4),  # 增加到4小时，确保数据集构建完成
        # 尝试设置分布式后端初始化参数
        process_group_backend="nccl",
    )
    trainer = nl.Trainer(
        devices=args.devices,
        num_nodes=args.num_nodes,
        max_steps=args.max_steps if args.early_stop_on_step is None else args.early_stop_on_step,
        accelerator="gpu",
        strategy=strategy,
        callbacks=callbacks,
        log_every_n_steps=args.log_every_n_steps,
        limit_val_batches=args.limit_val_batches,
        num_sanity_val_steps=0,
        use_distributed_sampler=False,
        plugins=nl.MegatronMixedPrecision(
            precision="bf16-mixed",
            params_dtype=torch.bfloat16,
            grad_reduce_in_fp32=args.grad_reduce_in_fp32,
            fp8="hybrid" if args.fp8 else None,
            fp8_amax_history_len=16 if args.fp8 else 1,
            fp8_amax_compute_algo="max" if args.fp8 else "most_recent",
            fp8_wgrad=args.fp8
            and (
                args.fp8_wgrad or args.use_megatron_comm_overlap_llama3_8k
            ),  # faster and less accurate when set to True, and MUST be True if using TP communication overlap
        ),
        val_check_interval=args.val_check_interval,
        enable_checkpointing=args.create_checkpoint_callback,
    )

    # Logger setup
    nemo_logger.setup(
        trainer,
        resume_if_exists=True,
    )

    if auto_resume is not None:
        auto_resume.setup(trainer, model)

    # Optimizer and scheduler setup
    opt_config = OptimizerConfig(
        optimizer="adam",
        lr=args.lr,
        adam_beta1=0.9,
        adam_beta2=0.95,
        weight_decay=args.wd,
        clip_grad=args.clip_grad,
        use_distributed_optimizer=True,
        bf16=True,
    )

    sched = CosineAnnealingScheduler(
        max_steps=trainer.max_steps,
        warmup_steps=args.warmup_steps,
        min_lr=args.min_lr,
    )

    opt = MegatronOptimizerModule(opt_config, sched, no_weight_decay_cond=evo2_config.hyena_no_weight_decay_cond_fn)
    opt.connect(model)

    # Start training
    trainer.fit(model, data_module)
    return trainer


def main():
    """Parsing args and running evo2 training."""
    args = parse_args()
    train(args=args)


if __name__ == "__main__":
    main()