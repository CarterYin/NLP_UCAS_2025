import sys
import os
import random
# 添加父目录到 path 以导入 utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import load_pku_corpus, get_llm_segmentation, calculate_metrics

def main():
    corpus_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "clean_data", "ChineseCorpus199801.txt")
    if not os.path.exists(corpus_path):
        print("语料文件不存在")
        return

    data = load_pku_corpus(corpus_path)
    random.seed(42)
    
    # 划分测试集和示例集
    # 随机选 50 个做测试
    test_samples = random.sample(data, 50)
    # 剩下的里面选 3 个做 Few-shot 示例
    remaining = [d for d in data if d not in test_samples]
    few_shot_examples = random.sample(remaining, 3)
    
    # 格式化示例: [(raw, "word1 word2 ..."), ...]
    formatted_examples = [(raw, " ".join(words)) for raw, words in few_shot_examples]

    print("正在运行 Zero-shot 基准测试...")
    gt_zero = []
    pred_zero = []
    for raw, gold in test_samples:
        res = get_llm_segmentation(raw, few_shot_examples=None)
        gt_zero.append(gold)
        pred_zero.append(res.split())
    
    p0, r0, f1_0 = calculate_metrics(gt_zero, pred_zero)
    print(f"Zero-shot F1: {f1_0:.4f}")

    print("\n正在运行 Few-shot (3-shot) 实验...")
    gt_few = []
    pred_few = []
    for raw, gold in test_samples:
        res = get_llm_segmentation(raw, few_shot_examples=formatted_examples)
        gt_few.append(gold)
        pred_few.append(res.split())
        
    p3, r3, f1_3 = calculate_metrics(gt_few, pred_few)
    print(f"Few-shot F1: {f1_3:.4f}")

    # 保存报告
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "q4_report.md")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# 提升大模型分词性能的方法实验\n\n")
        f.write("## 方法：Few-shot Prompting (少样本提示)\n")
        f.write("在提示词中提供少量（如3个）正确分词的示例，帮助模型理解分词标准（如颗粒度）。\n\n")
        
        f.write("## 实验结果\n")
        f.write("| Method | Precision | Recall | F1 Score |\n")
        f.write("|--------|-----------|--------|----------|\n")
        f.write(f"| Zero-shot | {p0:.4f} | {r0:.4f} | {f1_0:.4f} |\n")
        f.write(f"| Few-shot (3-shot) | {p3:.4f} | {r3:.4f} | {f1_3:.4f} |\n")
        
        f.write("\n## 结论\n")
        if f1_3 > f1_0:
            f.write("实验表明，Few-shot Prompting 能有效提升分词性能。示例帮助模型对齐了分词规范（例如复合词的处理）。\n")
        else:
            f.write("实验结果差异不明显，可能是因为模型本身Zero-shot能力已经很强，或者示例选择不够典型。\n")

    print(f"实验报告已生成: {output_path}")

if __name__ == "__main__":
    main()
