import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Paths
BASE_DIR = r"E:\homework\nlp\hw2\question1"
OUTPUT_FILE = r"E:\homework\nlp\hw2\question4\cross_lingual_results.txt"

ZH_FILE = os.path.join(BASE_DIR, "word_vectors_lstm_zh.txt")
EN_FILE = os.path.join(BASE_DIR, "word_vectors_lstm_en.txt")

PAIRS = [
    ("书", "book"),
    ("工作", "work"),
    ("工作", "job"),
    ("中国", "china"),
    ("人", "people"),
    ("发展", "development"),
    ("经济", "economy"),
    ("国家", "country"),
    ("和平", "peace"),
    ("世界", "world"),
    ("政府", "government"),
    ("合作", "cooperation")
]

def load_embeddings(file_path):
    embeddings = {}
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            f.readline() # Skip header
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2: continue
                word = parts[0]
                vec = np.array([float(x) for x in parts[1:]], dtype=np.float32)
                embeddings[word] = vec
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
    return embeddings

def main():
    print("Loading embeddings...")
    zh_embeds = load_embeddings(ZH_FILE)
    en_embeds = load_embeddings(EN_FILE)
    
    results = []
    results.append("# 跨语言词向量距离对比分析 (Cross-Lingual Vector Distance Analysis)")
    results.append("\n**注意**：以下结果基于独立训练的中英文向量空间计算。由于未进行空间对齐（Alignment），直接计算距离可能缺乏语义解释性。\n")
    results.append("| 中文词汇 (Chinese) | 英文词汇 (English) | 余弦相似度 (Cosine Sim) | 欧氏距离 (Euclidean Dist) |")
    results.append("|---|---|---|---|")
    
    for zh_word, en_word in PAIRS:
        zh_word_lower = zh_word
        en_word_lower = en_word.lower()
        
        if zh_word_lower in zh_embeds and en_word_lower in en_embeds:
            v1 = zh_embeds[zh_word_lower]
            v2 = en_embeds[en_word_lower]
            
            # Cosine Similarity
            cos_sim = cosine_similarity(v1.reshape(1, -1), v2.reshape(1, -1))[0][0]
            
            # Euclidean Distance
            euclidean_dist = np.linalg.norm(v1 - v2)
            
            results.append(f"| {zh_word} | {en_word} | {cos_sim:.4f} | {euclidean_dist:.4f} |")
        else:
            status = []
            if zh_word_lower not in zh_embeds: status.append(f"{zh_word} not found")
            if en_word_lower not in en_embeds: status.append(f"{en_word} not found")
            results.append(f"| {zh_word} | {en_word} | {' & '.join(status)} | - |")
            
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write('\n'.join(results))
    print(f"Results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
