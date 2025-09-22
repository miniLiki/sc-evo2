

# 一、任务简介
  
  该任务旨在使用构建高质量的甘蔗数据集对Evo2模型进行微调，以建立用于**甘蔗**领域的基因组**生成**任务**大模型**。任务**近一点的目标**是把原始数据喂给Evo2，加上各种trick，训练一个效果还不错的甘蔗模型；**远一点的目标**是提出一个benchmark/范式：我们自己定义甘蔗数据集的制作方式、评估方式，并给出训练好的模型。
 
 # 二、任务进度
 任务进度见下图。
<img width="2560" height="1440" alt="20250908_刘思逸_工作总结_01(2)" src="https://github.com/user-attachments/assets/f3b2b39e-5604-4555-9e59-8ccb17ddddd1" />




## 任务：
有两个任务：下一个词预测（生成）和调控元件识别。由于Evo2是基于自回归建模，所以调控元件识别本质上也是生成式的打分，所以可以先完成 数据集制作比较简单的生成任务，看看效果。

## 目录结构

```
mapping/
├── bionemo_train.py              # 主要训练脚本
├── preprocess_config.yaml        # 数据预处理配置
├── training_data_config.yaml     # 训练数据配置
├── finetuning.ipynb             # 微调流程notebook
├── hyena_modified.py            # 修改的Hyena模型实现
├── data/                        # 原始数据目录
│   ├── R570.v2023.genome.fasta  # R570品种基因组
│   ├── XTT22.genome.fa          # XTT22品种基因组
│   ├── ZZ1.v20231221.genome.fasta # ZZ1品种基因组
│   └── *.gff3.gz                # GFF3注释文件
├── split_fasta/                 # 数据划分结果
│   ├── R570_train.fa            # R570训练集
│   ├── R570_test.fa             # R570测试集
│   ├── XTT22_train.fa           # XTT22训练集
│   ├── XTT22_test.fa            # XTT22测试集
│   ├── ZZ1_train.fa             # ZZ1训练集
│   └── ZZ1_test.fa              # ZZ1测试集
├── preprocessed_data/           # 预处理后的数据
├── sequence_truncation/         # 序列截断工具
├── results/                     # 训练结果
├── prediction_results*/         # 预测结果
└── nemo2_evo2_*/               # NeMo2格式模型
XTT22_prediction_0shot.py        #零样本模型的评估代码
XTT22_prediction_0shot.py.png    #评估代码生成的图
data_ana.py                      # 数据分析并绘图的示例
```
## 研究阶段1：数据集建立

**我用的数据集**：我用的是[官网](https://sugarcane.gxu.edu.cn/scdb/)上的XTT22、R570、ZZ1。原始数据和基因注释文件在~/workspace/mapping/data下。
需要从官网重新下载对应名字的数据集放上去。


在重新下载官网数据集后，把涉及到数据的代码（例如finetuning.ipynb），把带有_train.fa、_test.fa的都换成原始的XTT22.genome.fa，或者ZZ1或R570等。例如：

**concat_path = "XTT22_train.fa"改成concat_path = "XTT22.genome.fa"**

 之前运行的结果是：零样本下的预测效果很差，微调后也只改善了一点，可能是甘蔗的高倍体复杂的特性导致。
 
**考虑改善数据集**：这个可能是该任务的核心难题，如何将异源多倍体数据喂给Evo2 以更有效地利用其模式识别能力。

参考意见：不止要输入[Genome基因型文件](https://sugarcane.gxu.edu.cn/scdb/)，也要将注释文件等信息加入到数据集中，合成一个多维矩阵喂给evo2进行训练。

但这样有一个缺点：现在已经证明了Evo2可以完成基因组的通用模式识别任务，我们是将Evo2作为Base Model来做SFT，最关心的应该是如何完成表征迁移，而不是重构陌生的数据结构进行训练。所以我觉得要在做数据的时候要权衡 有效数据的整合和对齐。


数据分析及可视化：位于~/workspace/data_ana.py。这只是一个示例。


## 研究阶段2：实现Evo2预测
Evo2的直接微调可以参考[教程](https://github.com/NVIDIA/bionemo-framework/blob/ca16c2acf9bf813d020b6d1e2d4e1240cfef6a69/docs/docs/user-guide/examples/bionemo-evo2/fine-tuning-tutorial.ipynb)实现。这里的任务就是下一个词预测，并且训练的逻辑基本遵循Evo2论文。

**我之前在代码中完成了Lora集成，而9月初的Evo2官方已经集成了Lora微调功能，所以可以直接参考官方实现，不一定用我的代码：**
<img width="1440" height="330" alt="21607b02d2bed5d71e68bc85b3879abd" src="https://github.com/user-attachments/assets/5275062d-99ae-44eb-adc4-a20616554282" />

我的运行在~/workspace/mapping/finetuning.ipynb文件前半部分实现，即命令：
!train_evo2 \
......
如果训练时torch报错权重无法更改，运行finetuning.ipynb中的：
!cp /workspace/hyena_modified.py /usr/local/lib/python3.12/dist-packages/nemo/collections/llm/gpt/model/hyena.py
在使用之前，从[huggingface](https://huggingface.co/arcinstitute/savanna_evo2_1b_base/tree/main)下载模型并保存到~/workspace/mapping/savanna_evo2_1b_base文件夹，也可以像[教程](https://github.com/NVIDIA/bionemo-framework/blob/ca16c2acf9bf813d020b6d1e2d4e1240cfef6a69/docs/docs/user-guide/examples/bionemo-evo2/fine-tuning-tutorial.ipynb)一样直接在代码中下载。

实验发现消融lora对性能几乎没有影响，但会降低显存使用。

## 研究阶段3：设计评估指标并可视化
这里用的也是切割的数据逐段输入gpu，否则会报显存。由于第一步重新下载了数据，所以这一步也要重新做一下：代码和切割后的数据和注释文档放在~/workspace/mapping/sequence_truncation/。


下一个词预测任务的评估在~/workspace/mapping/finetuning.ipynb文件后半部分实现。有predict和inference（简称infer）两种评估方式。详见[nemo框架的评估源代码](https://github.com/NVIDIA/bionemo-framework/tree/main/sub-packages/bionemo-evo2/src/bionemo/evo2/run)下的infer和predict。

infer任务(序列生成)的输出是一系列生成的DNA序列。

predict任务可以通过--output-log-prob-seqs 参数来选择**token logits**模式（输出为每个序列提供了逐个位置的预测信息）或**log probabilities**模式（输出为每个完整序列提供一个单一的分数（对数概率），代表了模型判断下该序列的适应度）。

命令示例为：
```
import os
import tempfile

os.environ['CUDA_VISIBLE_DEVICES'] = '4'
os.environ['TORCHINDUCTOR_CACHE_DIR'] = tempfile.mkdtemp()
os.environ['USER'] = 'user'
os.environ['USERNAME'] = 'user'
!predict_evo2 \
    --fasta sequence_truncation/biological_truncated_sequences_fixed.fasta \
    --ckpt-dir nemo2_evo2_1b_8k \
    --model-size 1b \
    --tensor-parallel-size 1 \
    --pipeline-model-parallel-size 1 \
    --context-parallel-size 1 \
    --batch-size 1 \
    --output-dir prediction_results_nolog_XTT22_0shot \
    --ckpt-format torch_dist \
    --log-prob-collapse-option mean \
    --prepend-bos
```

得到运行后的结果可以用来绘图，可以是下面这些图，也可以模仿evo2论文中的评估，或制定一些评估标准再进行评估可视化最好。

### **1. 序列生成（Inference）任务**

目标：分析模型生成的DNA序列的特性。

* **序列标识 (Sequence Logo)：** 最强大的可视化工具，用于理解从相关提示生成的多个序列的**保守性**。堆叠字母的高度显示序列保守性，单个字母高度表示该位置碱基的相对频率。非常适合可视化模型学到的保守**基序**或**模式**。
* **GC含量：** 绘制每个生成序列的GC含量条形图，快速了解**碱基组成**。
* **序列长度：** 如果模型生成变长序列，绘制长度直方图，了解**长度分布**。

---

### **2. 序列预测（Predict）任务：Token Logits 模式**

目标：理解模型对单个序列逐位置的预测细节。

* **逐位置 Logit 热图：**
    * 为单个序列创建热图，X轴为序列位置，Y轴为碱基（A, C, G, T）。
    * 单元格颜色代表Logit值，直观显示模型在每个位置认为最可能的核苷酸。
* **逐位置熵图：**
    * 将Logits通过SoftMax转换为概率分布。
    * 计算每个位置概率分布的**熵**，绘制熵与序列位置的关系图。
    * 熵低表示模型预测**置信度高**，熵高表示**不确定性**。

---

### **3. 序列预测（Predict）任务：Log Probabilities 模式**

目标：理解模型对每个完整序列的整体可能性或“适应度”评分。

* **模型分数 vs. 实验数据散点图：**
    * **验证模型性能的金标准。**
    * 如果具备序列对应的实验数据（如表达水平），将模型的对数概率（x轴）与实验值（y轴）绘制散点图。
    * **皮尔逊或斯皮尔曼相关系数**表明模型是否成功捕捉生物学特性。
* **分数直方图或小提琴图：**
    * 评估序列文库时，绘制对数概率分数分布图。
    * 揭示整体**适应度景观**，显示大多数突变是“有害”（分数低）还是“有益”（分数高）。
* **排序列表：**
    * 按对数概率从高到低排序序列表格。
    * 快速识别模型判断的“最佳”和“最差”序列，用于后续实验验证。
 
~~## 研究阶段4：Lora微调~~

我将lora实现位于~/workspace/mapping/bionemo_train.py，这段代码并不直接用于训练，真正的训练代码位于[docker容器配置好的底层环境](https://github.com/NVIDIA/bionemo-framework/tree/main/sub-packages/bionemo-evo2/src/bionemo/evo2/run)中。所以，我在~/workspace/mapping/finetuning.ipynb的中间部分用。
```
!cp /workspace/bionemo_train.py /usr/local/lib/python3.12/dist-packages/bionemo/evo2/run/train.py
```
将临时训练代码同步到真正的训练代码中。每次更改临时训练代码都要执行这个操作。可以通过改变bionemo_train代码并同步到真实训练代码，以改变lora配置。

## 研究阶段5：超参数优化
可以使用optuna库的自动调参框架。

 # 三、代码运行
 
 我是在Docker容器上运行的。但影响服务器的安全。如果不使用Docker，也可以尝试按照[bionemo官网文件](https://github.com/NVIDIA/bionemo-framework/blob/main/Dockerfile)配置环境。

朱博士的一页ppt：
<img width="712" alt="9ec4981707003b062d43d377685f48f2" src="https://github.com/user-attachments/assets/201581b7-b13c-48de-8a40-027b3bfa25ba" />

