# HW3: 基于大语言模型的中文分词实验

本项目探索了利用大语言模型（LLM，主要使用 Qwen-2.5 系列）进行中文分词（Chinese Word Segmentation, CWS）的能力。实验涵盖了从基准测试、跨领域数据爬取与评估、不同规模模型对比到 Prompt 工程优化的完整流程。

## 📂 目录结构

```
hw3/
├── clean_data/             # 存放清洗后的语料数据
│   ├── ChineseCorpus199801.txt  # PKU 标准语料
│   ├── corpus_news_clean.txt    # 爬取的新闻语料
│   ├── corpus_social_clean.txt  # 爬取的社交媒体语料
│   └── corpus_kepu_clean.txt    # 爬取的科普文章语料
├── q1/                     # Q1: Zero-shot 基准测试
│   └── eval_pku.py         # 在 PKU 语料上评估 LLM 分词性能
├── q2/                     # Q2: 领域适应性与数据爬取
│   ├── crawl_data.py       # 多源数据爬虫 (新闻/豆瓣/果壳)
│   └── eval_crawled.py     # 在爬取数据上评估 (以 Jieba 为伪标签)
├── q3/                     # Q3: 模型规模对比
│   └── compare_models.py   # 对比 7B/14B/72B 模型性能
├── q4/                     # Q4: 性能优化
│   └── improve_segmentation.py # Few-shot Prompting 实验
├── report/                 # 实验报告 (Typst 源码)
├── utils.py                # 通用工具函数 (API 调用, 指标计算)
└── README.md               # 项目说明文档
```

## 🛠️ 环境依赖

请确保安装以下 Python 库：

```bash
pip install requests beautifulsoup4 jieba dashscope numpy
```

此外，你需要配置阿里云 DashScope API Key。请在 `utils.py` 中设置 `DASHSCOPE_API_KEY`，或者设置环境变量：

```bash
export DASHSCOPE_API_KEY="your_api_key_here"
```

## 🚀 运行指南

### 1. 数据准备 (Q2)
虽然仓库中已包含部分示例数据，你可以运行爬虫脚本获取最新数据：

```bash
cd q2
python crawl_data.py
```
*注：爬虫包含延时和防反爬机制，运行时间可能较长。*

### 2. 基准测试 (Q1)
在 PKU 199801 语料上进行 Zero-shot 性能评估：

```bash
cd q1
python eval_pku.py
```

### 3. 领域适应性评估 (Q2)
评估模型在新闻、社交媒体、科普三个不同领域的表现（使用 Jieba 分词结果作为参考标准）：

```bash
cd q2
python eval_crawled.py
```

### 4. 模型规模对比 (Q3)
对比 Qwen-2.5-7B, 14B, 72B 在分词任务上的性能差异：

```bash
cd q3
python compare_models.py
```

### 5. 性能优化实验 (Q4)
使用 Few-shot Prompting (少样本提示) 技术提升分词效果：

```bash
cd q4
python improve_segmentation.py
```

## 📝 实验报告

实验报告源码位于 `report/` 目录下，使用 Typst 编写。如需编译报告：

```bash
cd report
typst compile main.typ
```

## 📊 实验结论摘要

1.  **基准性能**：LLM 在标准语料上 Zero-shot 表现良好，F1 值通常在 85%-90% 之间。
2.  **领域适应**：在规范文本（新闻）上表现最好，在口语化（社交媒体）和专有名词多（科普）的领域表现略有下降，但泛化能力强于传统分词工具。
3.  **模型规模**：模型参数量越大（72B vs 7B），对歧义词和长句的处理能力越强，F1 值有显著提升。
4.  **优化策略**：Few-shot Prompting 能有效规范模型的输出格式并提升对特定领域词汇的识别能力。
