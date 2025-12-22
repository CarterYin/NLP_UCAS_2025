import sys
import os
import random
# 添加父目录到 path 以导入 utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import load_pku_corpus, get_llm_segmentation, calculate_metrics

def main():
    corpus_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "clean_data", "ChineseCorpus199801.txt")
    
    if not os.path.exists(corpus_path):
        print(f"错误: 找不到语料文件 {corpus_path}")
        return

    print("正在加载语料...")
    data = load_pku_corpus(corpus_path)
    print(f"共加载 {len(data)} 条句子。")

    # 随机抽取 50 条
    sample_size = 50
    if len(data) > sample_size:
        samples = random.sample(data, sample_size)
    else:
        samples = data

    print(f"随机抽取 {len(samples)} 条进行测试...")

    ground_truth = []
    predictions = []

    for i, (raw_text, gold_words) in enumerate(samples):
        print(f"[{i+1}/{len(samples)}] 处理中: {raw_text[:20]}...")
        
        # 调用 LLM
        seg_result = get_llm_segmentation(raw_text)
        pred_words = seg_result.split()
        
        ground_truth.append(gold_words)
        predictions.append(pred_words)

    # 计算指标
    p, r, f1 = calculate_metrics(ground_truth, predictions)
    
    print("\n" + "="*30)
    print("测试结果 (Q1)")
    print("="*30)
    print(f"Precision: {p:.4f}")
    print(f"Recall:    {r:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print("="*30)

    # 保存详细结果
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "q1_results.txt")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(f"Precision: {p:.4f}\nRecall: {r:.4f}\nF1: {f1:.4f}\n\n")
        for i, (gold, pred) in enumerate(zip(ground_truth, predictions)):
            f.write(f"Sentence {i+1}:\n")
            f.write(f"Raw: {''.join(gold)}\n")
            f.write(f"Gold: {' '.join(gold)}\n")
            f.write(f"Pred: {' '.join(pred)}\n")
            f.write("-" * 20 + "\n")
    print(f"详细结果已保存至: {output_path}")

if __name__ == "__main__":
    main()
