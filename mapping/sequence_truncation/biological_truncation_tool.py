#!/usr/bin/env python3
"""
基于GFF注释的生物学智能截断工具

根据基因注释进行符合生物学规范的序列截断，确保截断后的序列具有明确的生物学意义
"""

import gzip
import argparse
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Set
import re

@dataclass
class AnnotationFeature:
    """GFF注释特征数据结构"""
    seqid: str
    source: str
    type: str
    start: int  # 1-based
    end: int    # 1-based
    score: Optional[float]
    strand: str
    phase: Optional[int]
    attributes: Dict[str, str]
    
    @property
    def length(self) -> int:
        return self.end - self.start + 1
    
    @property
    def gene_id(self) -> Optional[str]:
        """提取基因ID"""
        for key in ['gene_id', 'ID', 'Name', 'gene']:
            if key in self.attributes:
                return self.attributes[key]
        return None

@dataclass
class BiologicalUnit:
    """生物学功能单元"""
    unit_id: str
    chromosome: str
    start: int
    end: int
    strand: str
    unit_type: str  # 'gene_with_regulatory', 'gene_cluster', 'regulatory_region', etc.
    features: List[AnnotationFeature]
    description: str
    biological_significance: str
    
    @property
    def length(self) -> int:
        return self.end - self.start + 1

class GFFParser:
    """GFF3文件解析器"""
    
    def __init__(self, gff_file: str):
        self.gff_file = gff_file
        self.features = []
        self.chromosomes = set()
        
    def parse(self) -> List[AnnotationFeature]:
        """解析GFF3文件"""
        print(f"正在解析GFF文件: {self.gff_file}")
        
        opener = gzip.open if self.gff_file.endswith('.gz') else open
        mode = 'rt' if self.gff_file.endswith('.gz') else 'r'
        
        with opener(self.gff_file, mode) as f:
            for line_num, line in enumerate(f, 1):
                if line.startswith('#') or not line.strip():
                    continue
                
                try:
                    feature = self._parse_gff_line(line.strip())
                    if feature:
                        self.features.append(feature)
                        self.chromosomes.add(feature.seqid)
                        
                        if len(self.features) % 10000 == 0:
                            print(f"  已解析 {len(self.features)} 个特征...")
                            
                except Exception as e:
                    print(f"警告: 第{line_num}行解析失败: {e}")
                    continue
        
        print(f"GFF解析完成: 共{len(self.features)}个特征，涵盖{len(self.chromosomes)}条染色体")
        return self.features
    
    def _parse_gff_line(self, line: str) -> Optional[AnnotationFeature]:
        """解析单行GFF"""
        fields = line.split('\t')
        if len(fields) != 9:
            return None
        
        seqid, source, type_, start, end, score, strand, phase, attributes = fields
        
        # 解析属性
        attr_dict = {}
        for attr in attributes.split(';'):
            if '=' in attr:
                key, value = attr.split('=', 1)
                attr_dict[key.strip()] = value.strip()
        
        return AnnotationFeature(
            seqid=seqid,
            source=source,
            type=type_,
            start=int(start),
            end=int(end),
            score=float(score) if score != '.' else None,
            strand=strand,
            phase=int(phase) if phase != '.' else None,
            attributes=attr_dict
        )

class BiologicalUnitExtractor:
    """生物学功能单元提取器"""
    
    def __init__(self, max_unit_length: int = 20000):
        self.max_unit_length = max_unit_length
        self.feature_hierarchy = {
            'gene': 1,
            'mRNA': 2,
            'exon': 3,
            'CDS': 3,
            'UTR': 3,
            'five_prime_UTR': 3,
            'three_prime_UTR': 3,
            'promoter': 1,
            'enhancer': 1,
            'regulatory_region': 1,
            'repeat_region': 4
        }
    
    def extract_units(self, features: List[AnnotationFeature], chromosome: str) -> List[BiologicalUnit]:
        """从注释特征中提取生物学功能单元"""
        
        chrom_features = [f for f in features if f.seqid == chromosome]
        print(f"处理染色体 {chromosome}: {len(chrom_features)} 个特征")
        
        # 按类型分组特征
        features_by_type = {}
        for feature in chrom_features:
            if feature.type not in features_by_type:
                features_by_type[feature.type] = []
            features_by_type[feature.type].append(feature)
        
        units = []
        
        # 1. 提取基因单元（基因 + 调控区域）
        if 'gene' in features_by_type:
            gene_units = self._extract_gene_units(features_by_type, chrom_features)
            units.extend(gene_units)
        
        # 2. 提取基因间调控区域
        regulatory_units = self._extract_regulatory_units(features_by_type, units)
        units.extend(regulatory_units)
        
        # 3. 合并相近的单元形成基因簇
        cluster_units = self._merge_nearby_units(units)
        
        return cluster_units
    
    def _extract_gene_units(self, features_by_type: Dict, all_features: List[AnnotationFeature]) -> List[BiologicalUnit]:
        """提取基因功能单元"""
        units = []
        genes = features_by_type.get('gene', [])
        
        for gene in genes:
            # 获取与该基因相关的所有特征
            gene_features = [gene]
            
            # 查找该基因的mRNA、exon、CDS等
            for feature in all_features:
                if self._is_related_to_gene(feature, gene):
                    gene_features.append(feature)
            
            # 确定基因单元的边界（包含调控区域）
            gene_start = gene.start
            gene_end = gene.end
            
            # 添加启动子区域（上游2kb）和终止子区域（下游1kb）
            if gene.strand == '+':
                unit_start = max(1, gene_start - 2000)  # 上游启动子
                unit_end = gene_end + 1000  # 下游终止子
            else:
                unit_start = max(1, gene_start - 1000)  # 下游终止子（反向）
                unit_end = gene_end + 2000  # 上游启动子（反向）
            
            # 检查长度限制
            unit_length = unit_end - unit_start + 1
            if unit_length > self.max_unit_length:
                # 如果太长，只保留基因本体
                unit_start = gene_start
                unit_end = gene_end
                unit_type = "gene_only"
                description = f"基因 {gene.gene_id or 'unknown'} (仅编码区，原长度超限)"
                significance = "基因编码序列分析，可用于蛋白功能预测"
            else:
                unit_type = "gene_with_regulatory"
                description = f"基因 {gene.gene_id or 'unknown'} 及其调控区域"
                significance = "完整基因单元，可分析转录调控、启动子活性、基因表达"
            
            unit = BiologicalUnit(
                unit_id=f"{gene.gene_id or f'gene_{gene.start}'}",
                chromosome=gene.seqid,
                start=unit_start,
                end=unit_end,
                strand=gene.strand,
                unit_type=unit_type,
                features=gene_features,
                description=description,
                biological_significance=significance
            )
            
            units.append(unit)
        
        return units
    
    def _extract_regulatory_units(self, features_by_type: Dict, existing_units: List[BiologicalUnit]) -> List[BiologicalUnit]:
        """提取基因间调控区域"""
        units = []
        
        regulatory_types = ['promoter', 'enhancer', 'regulatory_region']
        all_regulatory = []
        
        for reg_type in regulatory_types:
            if reg_type in features_by_type:
                all_regulatory.extend(features_by_type[reg_type])
        
        # 找出未被基因单元覆盖的调控区域
        for reg_feature in all_regulatory:
            if not self._is_covered_by_units(reg_feature, existing_units):
                # 独立调控区域
                unit = BiologicalUnit(
                    unit_id=f"{reg_feature.type}_{reg_feature.start}",
                    chromosome=reg_feature.seqid,
                    start=reg_feature.start,
                    end=reg_feature.end,
                    strand=reg_feature.strand,
                    unit_type="regulatory_region",
                    features=[reg_feature],
                    description=f"独立{reg_feature.type}区域",
                    biological_significance=f"{reg_feature.type}功能分析，可研究远程调控作用"
                )
                
                if unit.length <= self.max_unit_length:
                    units.append(unit)
        
        return units
    
    def _merge_nearby_units(self, units: List[BiologicalUnit]) -> List[BiologicalUnit]:
        """合并相近的功能单元形成基因簇"""
        if not units:
            return []
        
        # 按染色体和位置排序
        sorted_units = sorted(units, key=lambda u: (u.chromosome, u.start))
        merged_units = []
        
        current_cluster = [sorted_units[0]]
        
        for unit in sorted_units[1:]:
            last_unit = current_cluster[-1]
            
            # 如果在同一染色体且距离较近（<5kb），考虑合并
            if (unit.chromosome == last_unit.chromosome and 
                unit.start - last_unit.end < 5000):
                
                # 计算合并后的长度
                cluster_start = min(current_cluster[0].start, unit.start)
                cluster_end = max(current_cluster[-1].end, unit.end)
                merged_length = cluster_end - cluster_start + 1
                
                if merged_length <= self.max_unit_length:
                    current_cluster.append(unit)
                    continue
            
            # 完成当前簇
            if len(current_cluster) > 1:
                # 创建基因簇单元
                cluster_unit = self._create_cluster_unit(current_cluster)
                merged_units.append(cluster_unit)
            else:
                # 单个单元直接添加
                merged_units.append(current_cluster[0])
            
            # 开始新簇
            current_cluster = [unit]
        
        # 处理最后一个簇
        if len(current_cluster) > 1:
            cluster_unit = self._create_cluster_unit(current_cluster)
            merged_units.append(cluster_unit)
        else:
            merged_units.append(current_cluster[0])
        
        return merged_units
    
    def _create_cluster_unit(self, cluster_units: List[BiologicalUnit]) -> BiologicalUnit:
        """创建基因簇单元"""
        cluster_start = min(u.start for u in cluster_units)
        cluster_end = max(u.end for u in cluster_units)
        
        gene_ids = [u.unit_id for u in cluster_units]
        all_features = []
        for unit in cluster_units:
            all_features.extend(unit.features)
        
        return BiologicalUnit(
            unit_id=f"cluster_{'_'.join(gene_ids[:3])}",  # 限制ID长度
            chromosome=cluster_units[0].chromosome,
            start=cluster_start,
            end=cluster_end,
            strand='+',  # 簇没有固定方向
            unit_type="gene_cluster",
            features=all_features,
            description=f"基因簇包含{len(cluster_units)}个功能单元: {', '.join(gene_ids)}",
            biological_significance="基因簇共表达分析，可研究基因共调控、协同进化、功能相关性"
        )
    
    def _is_related_to_gene(self, feature: AnnotationFeature, gene: AnnotationFeature) -> bool:
        """判断特征是否与基因相关"""
        # 基于位置重叠判断
        if feature.seqid != gene.seqid:
            return False
        
        # 如果特征在基因内部或紧邻，认为相关
        return (feature.start <= gene.end + 1000 and feature.end >= gene.start - 1000)
    
    def _is_covered_by_units(self, feature: AnnotationFeature, units: List[BiologicalUnit]) -> bool:
        """判断特征是否被现有单元覆盖"""
        for unit in units:
            if (unit.chromosome == feature.seqid and
                unit.start <= feature.start and
                unit.end >= feature.end):
                return True
        return False

class BiologicalTruncator:
    """生物学截断器"""
    
    def __init__(self, fasta_file: str, gff_file: str, max_length: int = 20000):
        self.fasta_file = fasta_file
        self.gff_file = gff_file
        self.max_length = max_length
        self.sequences = {}
        self.gff_parser = GFFParser(gff_file)
        self.unit_extractor = BiologicalUnitExtractor(max_length)
        
    def load_sequences(self):
        """加载FASTA序列"""
        print(f"正在加载序列文件: {self.fasta_file}")
        
        sequence_count = 0
        total_length = 0
        
        with open(self.fasta_file, 'r') as f:
            for record in SeqIO.parse(f, 'fasta'):
                self.sequences[record.id] = str(record.seq)
                sequence_count += 1
                total_length += len(record.seq)
                
                if sequence_count % 10 == 0:
                    print(f"  已加载 {sequence_count} 条序列...")
        
        print(f"序列加载完成: {sequence_count} 条序列，总长度 {total_length:,} bp")
        
    def analyze_biological_significance(self, units: List[BiologicalUnit]) -> Dict[str, int]:
        """分析生物学意义统计"""
        significance_stats = {}
        
        for unit in units:
            unit_type = unit.unit_type
            if unit_type not in significance_stats:
                significance_stats[unit_type] = 0
            significance_stats[unit_type] += 1
        
        return significance_stats
    
    def truncate_sequences(self, output_file: str) -> Tuple[List[BiologicalUnit], Dict]:
        """执行生物学截断"""
        
        # 加载序列
        self.load_sequences()
        
        # 解析GFF
        features = self.gff_parser.parse()
        
        # 提取生物学单元
        all_units = []
        available_chromosomes = set(self.sequences.keys()) & self.gff_parser.chromosomes
        
        print(f"\n开始提取生物学功能单元...")
        print(f"可用染色体: {sorted(available_chromosomes)}")
        
        for chromosome in sorted(available_chromosomes):
            if chromosome in self.sequences:
                units = self.unit_extractor.extract_units(features, chromosome)
                all_units.extend(units)
                print(f"  {chromosome}: 提取了 {len(units)} 个功能单元")
        
        # 分析意义
        significance_stats = self.analyze_biological_significance(all_units)
        
        # 生成截断序列
        truncated_records = []
        valid_units = []
        
        print(f"\n生成截断序列...")
        for i, unit in enumerate(all_units):
            if unit.chromosome in self.sequences:
                # 提取序列
                chrom_seq = self.sequences[unit.chromosome]
                unit_seq = chrom_seq[unit.start-1:unit.end]  # 转换为0-based
                
                if len(unit_seq) > 0:
                    # 创建序列记录
                    record = SeqRecord(
                        Seq(unit_seq),
                        id=f"{unit.unit_id}",
                        description=f"{unit.description} | {unit.chromosome}:{unit.start}-{unit.end}({unit.strand}) | {unit.biological_significance}"
                    )
                    
                    truncated_records.append(record)
                    valid_units.append(unit)
                    
                    if len(truncated_records) % 100 == 0:
                        print(f"  已生成 {len(truncated_records)} 个截断序列...")
        
        # 保存FASTA文件
        with open(output_file, 'w') as f:
            SeqIO.write(truncated_records, f, 'fasta')
        
        print(f"\n截断完成!")
        print(f"输出文件: {output_file}")
        print(f"总功能单元: {len(valid_units)}")
        
        return valid_units, significance_stats
    
    def generate_report(self, units: List[BiologicalUnit], stats: Dict, output_dir: str):
        """生成分析报告"""
        
        report_file = Path(output_dir) / "truncation_report.txt"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("=== 基于GFF注释的生物学截断报告 ===\n\n")
            
            f.write(f"输入文件:\n")
            f.write(f"  序列文件: {self.fasta_file}\n")
            f.write(f"  注释文件: {self.gff_file}\n")
            f.write(f"  最大长度限制: {self.max_length:,} bp\n\n")
            
            f.write(f"截断结果统计:\n")
            f.write(f"  总功能单元: {len(units)}\n")
            
            for unit_type, count in stats.items():
                f.write(f"  {unit_type}: {count} 个\n")
            
            f.write(f"\n生物学意义说明:\n")
            
            # 详细说明每种单元类型的意义
            significance_explanations = {
                'gene_with_regulatory': """
    ✅ 完整基因单元 (基因+调控区域):
       - 生物学意义: 包含启动子、基因编码区、终止子的完整功能单元
       - Predict任务: 可分析基因表达调控、启动子强度、转录效率
       - Inference任务: 可生成具有调控特征的功能基因序列
       - 应用价值: 基因治疗设计、合成生物学、转录调控研究
                """,
                'gene_only': """
    ⚠️ 仅基因编码区:
       - 生物学意义: 仅包含蛋白编码序列，缺少调控元件
       - Predict任务: 适合蛋白功能预测、编码序列优化分析
       - Inference任务: 可生成编码蛋白的DNA序列
       - 限制: 无法分析转录调控，启动子预测意义有限
                """,
                'gene_cluster': """
    ✅ 基因簇:
       - 生物学意义: 功能相关基因的协调表达单元
       - Predict任务: 可分析基因共调控、代谢通路完整性
       - Inference任务: 可生成功能相关的基因组合
       - 应用价值: 代谢工程、基因簇转移、进化分析
                """,
                'regulatory_region': """
    ⚠️ 独立调控区域:
       - 生物学意义: 增强子、启动子等调控元件
       - Predict任务: 可分析调控强度，但缺乏靶基因上下文
       - Inference任务: 可生成调控序列，但功能验证需要额外实验
       - 建议: 结合相关基因进行完整分析
                """
            }
            
            for unit_type, explanation in significance_explanations.items():
                if unit_type in stats:
                    f.write(explanation)
                    f.write("\n")
            
            f.write(f"\n任务建议:\n")
            
            # 基于统计结果给出建议
            total_complete_units = stats.get('gene_with_regulatory', 0) + stats.get('gene_cluster', 0)
            total_incomplete_units = stats.get('gene_only', 0) + stats.get('regulatory_region', 0)
            
            if total_complete_units > total_incomplete_units:
                f.write("✅ 建议执行 Predict 和 Inference 任务:\n")
                f.write("   - 大部分序列保持了生物学完整性\n")
                f.write("   - 适合研究基因调控和功能预测\n")
                f.write("   - 生成的序列具有明确的生物学背景\n")
            else:
                f.write("⚠️ 谨慎执行任务，建议预处理:\n")
                f.write("   - 较多序列缺少完整的功能上下文\n")
                f.write("   - 建议组合相关基因进行分析\n")
                f.write("   - 或针对特定研究目的选择合适的序列子集\n")
            
            f.write(f"\n详细单元列表:\n")
            f.write("-" * 100 + "\n")
            f.write(f"{'单元ID':<20} {'类型':<20} {'染色体':<10} {'长度':<10} {'描述':<30}\n")
            f.write("-" * 100 + "\n")
            
            for unit in units[:50]:  # 只显示前50个
                f.write(f"{unit.unit_id:<20} {unit.unit_type:<20} {unit.chromosome:<10} {unit.length:<10} {unit.description[:30]:<30}\n")
            
            if len(units) > 50:
                f.write(f"... 还有 {len(units)-50} 个单元\n")
        
        print(f"分析报告已保存: {report_file}")

def main():
    parser = argparse.ArgumentParser(description='基于GFF注释的生物学序列截断工具')
    parser.add_argument('--fasta', required=True, help='输入FASTA文件路径')
    parser.add_argument('--gff', required=True, help='输入GFF3注释文件路径')
    parser.add_argument('--output', required=True, help='输出FASTA文件路径')
    parser.add_argument('--max-length', type=int, default=20000, help='最大序列长度限制 (默认: 20000)')
    parser.add_argument('--report-dir', default='.', help='报告输出目录 (默认: 当前目录)')
    
    args = parser.parse_args()
    
    # 创建截断器
    truncator = BiologicalTruncator(
        fasta_file=args.fasta,
        gff_file=args.gff,
        max_length=args.max_length
    )
    
    # 执行截断
    units, stats = truncator.truncate_sequences(args.output)
    
    # 生成报告
    truncator.generate_report(units, stats, args.report_dir)
    
    print(f"\n🧬 生物学截断完成!")
    print(f"📁 输出文件: {args.output}")
    print(f"📊 分析报告: {Path(args.report_dir) / 'truncation_report.txt'}")

if __name__ == "__main__":
    main() 