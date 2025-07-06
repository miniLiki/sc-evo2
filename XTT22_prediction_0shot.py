#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
甘蔗基因组预测结果分析可视化脚本

分析 prediction_results_06082149 实验结果，生成对数概率分布图表
包括直方图和小提琴图，展示模型对序列的整体可能性评估
"""

import torch
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

# 设置字体为支持英文的字体
plt.rcParams['font.family'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_data():
    """加载预测结果和序列索引映射数据"""
    # 切换到结果目录
    results_dir = "mapping/prediction_results_06082149"
    
    # 加载PyTorch预测结果
    predictions = torch.load(os.path.join(results_dir, 'predictions__rank_0.pt'), map_location='cpu')
    
    # 加载序列索引映射
    with open(os.path.join(results_dir, 'seq_idx_map.json'), 'r') as f:
        seq_idx_map = json.load(f)
    
    # 提取对数概率和序列索引
    log_probs = predictions['log_probs_seqs'].numpy()
    seq_indices = predictions['seq_idx'].numpy()
    
    print(f"加载数据完成:")
    print(f"  对数概率数据点数: {len(log_probs)}")
    print(f"  序列索引映射数: {len(seq_idx_map)}")
    print(f"  对数概率范围: [{log_probs.min():.4f}, {log_probs.max():.4f}]")
    print(f"  对数概率均值: {log_probs.mean():.4f}")
    print(f"  对数概率标准差: {log_probs.std():.4f}")
    
    return log_probs, seq_indices, seq_idx_map

def create_visualization(log_probs, seq_indices, seq_idx_map):
    """创建预测结果可视化图表"""
    
    # 创建图形，包含两个子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # === 子图1: 对数概率直方图 ===
    ax1.hist(log_probs, bins=50, alpha=0.7, color='skyblue', edgecolor='navy', linewidth=0.8)
    ax1.axvline(log_probs.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {log_probs.mean():.3f}')
    ax1.axvline(np.median(log_probs), color='orange', linestyle='--', linewidth=2, label=f'Median: {np.median(log_probs):.3f}')
    
    # 添加统计信息
    ax1.set_xlabel('Log Probability', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Number of Sequences', fontsize=12, fontweight='bold')
    ax1.set_title('Sugarcane Gene Sequence Log Probability Distribution\n(Model Predicted Sequence Fitness)', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 添加统计标注
    textstr = f'Total Sequences: {len(log_probs)}\nStd Dev: {log_probs.std():.3f}\nMin: {log_probs.min():.3f}\nMax: {log_probs.max():.3f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax1.text(0.02, 0.98, textstr, transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', bbox=props)
    
    # === 子图2: 对数概率小提琴图 ===
    # 为小提琴图准备数据
    violin_data = [log_probs]
    parts = ax2.violinplot(violin_data, positions=[1], widths=0.6, showmeans=True, showmedians=True)
    
    # 美化小提琴图
    for pc in parts['bodies']:
        pc.set_facecolor('lightblue')
        pc.set_alpha(0.7)
        pc.set_edgecolor('navy')
    
    # 设置小提琴图样式
    parts['cmeans'].set_color('red')
    parts['cmedians'].set_color('orange')
    parts['cmaxes'].set_color('black')
    parts['cmins'].set_color('black')
    parts['cbars'].set_color('black')
    
    ax2.set_ylabel('Log Probability', fontsize=12, fontweight='bold')
    ax2.set_title('Sugarcane Gene Sequence Log Probability Density\n(Distribution Shape Analysis)', fontsize=14, fontweight='bold')
    ax2.set_xticks([1])
    ax2.set_xticklabels(['Sequence Prediction Score'])
    ax2.grid(True, alpha=0.3)
    
    # 添加四分位数标注
    q25, q50, q75 = np.percentile(log_probs, [25, 50, 75])
    ax2.axhline(y=q25, color='gray', linestyle=':', alpha=0.7, label=f'Q1: {q25:.3f}')
    ax2.axhline(y=q50, color='orange', linestyle='--', alpha=0.8, label=f'Q2 (Median): {q50:.3f}')
    ax2.axhline(y=q75, color='gray', linestyle=':', alpha=0.7, label=f'Q3: {q75:.3f}')
    ax2.legend(loc='upper right')
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    output_filename = "Sugarcane_Gene_Sequence_Prediction_Analysis.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n图表已保存为: {output_filename}")
    print(f"图片分辨率: 300 DPI")
    
    return output_filename

def analyze_distribution(log_probs):
    """分析对数概率分布的统计特征"""
    print("\n=== 详细统计分析 ===")
    
    # 基础统计
    print(f"样本数量: {len(log_probs)}")
    print(f"均值: {np.mean(log_probs):.6f}")
    print(f"标准差: {np.std(log_probs):.6f}")
    print(f"中位数: {np.median(log_probs):.6f}")
    print(f"最小值: {np.min(log_probs):.6f}")
    print(f"最大值: {np.max(log_probs):.6f}")
    
    # 四分位数
    q25, q50, q75 = np.percentile(log_probs, [25, 50, 75])
    iqr = q75 - q25
    print(f"\n四分位数分析:")
    print(f"Q1 (25%): {q25:.6f}")
    print(f"Q2 (50%): {q50:.6f}")
    print(f"Q3 (75%): {q75:.6f}")
    print(f"四分位距(IQR): {iqr:.6f}")
    
    # 偏度和峰度
    skewness = stats.skew(log_probs)
    kurtosis = stats.kurtosis(log_probs)
    print(f"\n分布形状分析:")
    print(f"偏度 (Skewness): {skewness:.4f}")
    if skewness > 0:
        print("  → 右偏分布 (有少数高分序列)")
    elif skewness < 0:
        print("  → 左偏分布 (有少数低分序列)")
    else:
        print("  → 近似对称分布")
    
    print(f"峰度 (Kurtosis): {kurtosis:.4f}")
    if kurtosis > 0:
        print("  → 尖峰分布 (相对正态分布更集中)")
    elif kurtosis < 0:
        print("  → 平峰分布 (相对正态分布更分散)")
    else:
        print("  → 近似正态分布")
    
    # 异常值检测
    lower_bound = q25 - 1.5 * iqr
    upper_bound = q75 + 1.5 * iqr
    outliers = log_probs[(log_probs < lower_bound) | (log_probs > upper_bound)]
    print(f"\n异常值分析:")
    print(f"异常值数量: {len(outliers)} ({len(outliers)/len(log_probs)*100:.2f}%)")
    print(f"异常值范围: 低于 {lower_bound:.4f} 或高于 {upper_bound:.4f}")
    
    # 适应度景观解释
    print(f"\n=== 生物学解释 ===")
    high_fitness = log_probs[log_probs > q75]
    low_fitness = log_probs[log_probs < q25]
    
    print(f"高适应度序列 (>Q3): {len(high_fitness)} 个 ({len(high_fitness)/len(log_probs)*100:.1f}%)")
    print(f"低适应度序列 (<Q1): {len(low_fitness)} 个 ({len(low_fitness)/len(log_probs)*100:.1f}%)")
    print(f"中等适应度序列: {len(log_probs) - len(high_fitness) - len(low_fitness)} 个")
    
    if skewness < -0.5:
        print("模型预测显示: 大多数突变有害，少数突变有益")
    elif skewness > 0.5:
        print("模型预测显示: 大多数序列具有较高适应度")
    else:
        print("模型预测显示: 序列适应度呈相对均匀分布")

def main():
    """主函数"""
    print("开始分析甘蔗基因组预测结果...")
    
    try:
        # 加载数据
        log_probs, seq_indices, seq_idx_map = load_data()
        
        # 生成可视化
        output_file = create_visualization(log_probs, seq_indices, seq_idx_map)
        
        # 详细统计分析
        analyze_distribution(log_probs)
        
        print(f"\n分析完成！生成的图表文件: {output_file}")
        
    except Exception as e:
        print(f"分析过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 