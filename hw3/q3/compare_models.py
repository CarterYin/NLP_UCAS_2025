import sys
import os
import random
# 添加父目录到 path 以导入 utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import load_pku_corpus, get_llm_segmentation, calculate_metrics

def main():
    # 定义要对比的模型列表
    # 使用不同参数量的 Qwen 模型进行对比
    models = [
        "Qwen/Qwen2.5-7B-Instruct",
        "Qwen/Qwen2.5-14B-Instruct", 
        "Qwen/Qwen2.5-72B-Instruct",
    ]
    
    corpus_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "clean_data", "ChineseCorpus199801.txt")
    if not os.path.exists(corpus_path):
        print("语料文件不存在")
        return

    data = load_pku_corpus(corpus_path)
    # 固定随机种子以保证对比公平
    random.seed(42)
    samples = random.sample(data, 50) if len(data) > 50 else data
    
    results = {}

    for model_name in models:
        print(f"\n正在测试模型: {model_name} ...")
        ground_truth = []
        predictions = []
        
        for i, (raw_text, gold_words) in enumerate(samples):
            if i % 10 == 0: print(f"进度: {i}/{len(samples)}")
            
            # 传入模型名称
            seg_result = get_llm_segmentation(raw_text, model=model_name)
            pred_words = seg_result.split()
            
            ground_truth.append(gold_words)
            predictions.append(pred_words)
            
        p, r, f1 = calculate_metrics(ground_truth, predictions)
        results[model_name] = {"p": p, "r": r, "f1": f1}
        print(f"{model_name} 结果: F1={f1:.4f}")

    # 输出对比报告
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "q3_report.md")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# 模型参数量对分词性能的影响分析\n\n")
        f.write("| Model | Precision | Recall | F1 Score |\n")
        f.write("|-------|-----------|--------|----------|\n")
        for m, res in results.items():
            f.write(f"| {m} | {res['p']:.4f} | {res['r']:.4f} | {res['f1']:.4f} |\n")
        
        f.write("\n## 分析\n")
        f.write("通常情况下，参数量更大的模型（如 GPT-4）在理解语义和处理歧义方面表现更好，因此分词准确率往往更高。")
        f.write("较小的模型（如 GPT-3.5 或 mini 版）虽然速度快，但在处理生僻词或复杂句式时可能不如大模型准确。\n")
        
    print(f"对比报告已生成: {output_path}")

if __name__ == "__main__":
    main()
