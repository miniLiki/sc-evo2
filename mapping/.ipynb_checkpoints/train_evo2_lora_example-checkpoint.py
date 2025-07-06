#!/usr/bin/env python3
"""
Evo2 LoRA 微调示例脚本

使用方法:
python train_evo2_lora_example.py

此脚本展示了如何使用修改后的 bionemo_train.py 进行 LoRA 微调
"""

import subprocess
import sys
from pathlib import Path

def run_lora_training():
    """运行 LoRA 微调训练"""
    
    # 基本训练参数
    cmd = [
        sys.executable, "bionemo_train.py",
        "--mock-data",  # 使用模拟数据进行测试
        "--enable-lora",  # 启用 LoRA 微调
        "--lora-rank", "16",  # LoRA 秩
        "--lora-alpha", "32",  # LoRA alpha 参数
        "--lora-dropout", "0.1",  # LoRA dropout
        "--lora-target-modules", "module.decoder.layers.17.self_attention.linear_qkv",  # 目标层
        "--model-size", "test",  # 使用测试模型大小
        "--devices", "1",
        "--num-nodes", "1",
        "--micro-batch-size", "1",
        "--global-batch-size", "8",
        "--max-steps", "100",  # 少量步骤用于测试
        "--val-check-interval", "50",
        "--lr", "5e-4",  # LoRA 通常使用较高的学习率
        "--min-lr", "5e-5",
        "--warmup-steps", "10",
        "--seq-length", "1024",  # 较短的序列长度用于测试
        "--result-dir", "./results_lora",
        "--experiment-name", "evo2_lora_test",
        "--log-every-n-steps", "10",
        "--workers", "4",
    ]
    
    print("开始 LoRA 微调训练...")
    print("命令:", " ".join(cmd))
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("训练成功完成!")
        print("输出:", result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"训练失败: {e}")
        print("错误输出:", e.stderr)
        return False
    
    return True

def main():
    """主函数"""
    print("Evo2 LoRA 微调示例")
    print("=" * 50)
    
    # 检查 bionemo_train.py 是否存在
    if not Path("bionemo_train.py").exists():
        print("错误: 找不到 bionemo_train.py 文件")
        print("请确保在正确的目录中运行此脚本")
        return
    
    # 运行 LoRA 训练
    success = run_lora_training()
    
    if success:
        print("\n" + "=" * 50)
        print("LoRA 微调示例运行成功!")
        print("检查 ./results_lora 目录查看训练结果")
    else:
        print("\n" + "=" * 50)
        print("LoRA 微调示例运行失败，请查看错误信息")

if __name__ == "__main__":
    main() 