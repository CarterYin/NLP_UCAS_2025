import sys
import os
import random
import jieba
# 添加父目录到 path 以导入 utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import get_llm_segmentation, calculate_metrics

def load_crawled_data(filepath, limit=20):
    """读取爬取的清洗后数据"""
    lines = []
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if len(line) > 10: # 过滤太短的
                    lines.append(line)
    
    if len(lines) > limit:
        return random.sample(lines, limit)
    return lines

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "clean_data")
    
    # 定义要测试的文件
    files = {
        "News": "corpus_news_clean.txt",
        "Social": "corpus_social_clean.txt",
        "Kepu": "corpus_kepu_clean.txt" # 假设这是果壳爬取的结果文件名
    }

    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "q2_results.txt")
    
    with open(output_path, 'w', encoding='utf-8') as f_out:
        for type_name, filename in files.items():
            filepath = os.path.join(data_dir, filename)
            print(f"\n正在处理类型: {type_name} ({filename})...")
            
            samples = load_crawled_data(filepath)
            if not samples:
                print(f"警告: 文件 {filename} 不存在或为空。")
                f_out.write(f"\nType: {type_name}\nNo data found.\n")
                continue

            print(f"抽取了 {len(samples)} 条样本。")
            
            # 使用 Jieba 作为参考基准 (Pseudo-Gold)
            jieba_refs = []
            llm_preds = []
            
            f_out.write(f"\n{'='*20}\nType: {type_name}\n{'='*20}\n")

            for i, raw_text in enumerate(samples):
                print(f"[{i+1}/{len(samples)}] Segmenting...")
                
                # Jieba 分词
                jieba_seg = list(jieba.cut(raw_text))
                jieba_refs.append(jieba_seg)
                
                # LLM 分词
                llm_res = get_llm_segmentation(raw_text)
                llm_seg = llm_res.split()
                llm_preds.append(llm_seg)
                
                f_out.write(f"Sent {i+1}:\n")
                f_out.write(f"Raw: {raw_text}\n")
                f_out.write(f"Jieba: {' '.join(jieba_seg)}\n")
                f_out.write(f"LLM:   {' '.join(llm_seg)}\n")
                f_out.write("-" * 10 + "\n")

            # 计算与 Jieba 的一致性
            p, r, f1 = calculate_metrics(jieba_refs, llm_preds)
            print(f"{type_name} - Agreement with Jieba: F1={f1:.4f}")
            f_out.write(f"\nMetrics (Reference = Jieba):\nPrecision: {p:.4f}\nRecall: {r:.4f}\nF1: {f1:.4f}\n")

    print(f"\n所有结果已保存至: {output_path}")

if __name__ == "__main__":
    main()
