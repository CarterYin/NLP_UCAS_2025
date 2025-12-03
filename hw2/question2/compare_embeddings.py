import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Paths
BASE_DIR = r"E:\homework\nlp\hw2\question1"
REPORT_PATH = r"E:\homework\nlp\hw2\question2\report.md"

FILES = {
    "zh": {
        "FNN": os.path.join(BASE_DIR, "word_vectors_fnn_zh.txt"),
        "RNN": os.path.join(BASE_DIR, "word_vectors_rnn_zh.txt"),
        "LSTM": os.path.join(BASE_DIR, "word_vectors_lstm_zh.txt"),
    },
    "en": {
        "FNN": os.path.join(BASE_DIR, "word_vectors_fnn_en.txt"),
        "RNN": os.path.join(BASE_DIR, "word_vectors_rnn_en.txt"),
        "LSTM": os.path.join(BASE_DIR, "word_vectors_lstm_en.txt"),
    }
}

# Test words to analyze
TEST_WORDS = {
    "zh": ["中国", "发展", "经济", "人民", "希望"],
    "en": ["china", "development", "world", "cooperation", "peace"]
}

def load_embeddings(file_path):
    embeddings = {}
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            # Skip header if present (first line usually has vocab_size dim)
            first_line = f.readline().strip().split()
            if len(first_line) == 2:
                pass # It's the header
            else:
                # No header, reset pointer
                f.seek(0)
            
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2: continue
                word = parts[0]
                vec = np.array([float(x) for x in parts[1:]], dtype=np.float32)
                embeddings[word] = vec
    except FileNotFoundError:
        print(f"Warning: File not found {file_path}")
    return embeddings

def get_nearest_neighbors(embeddings, word, k=5):
    if word not in embeddings:
        return []
    
    target_vec = embeddings[word].reshape(1, -1)
    
    # Convert dict to matrices for faster calculation
    words = list(embeddings.keys())
    matrix = np.array([embeddings[w] for w in words])
    
    sims = cosine_similarity(target_vec, matrix)[0]
    
    # Sort indices
    sorted_indices = sims.argsort()[::-1]
    
    neighbors = []
    # Skip the word itself (index 0 usually, but let's check)
    count = 0
    for idx in sorted_indices:
        w = words[idx]
        if w == word: continue
        neighbors.append((w, sims[idx]))
        count += 1
        if count >= k: break
        
    return neighbors

def calculate_overlap(list1, list2):
    set1 = set([x[0] for x in list1])
    set2 = set([x[0] for x in list2])
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    if not union: return 0.0
    return len(intersection) / len(union)

def main():
    report_content = []
    report_content.append("# 词向量模型对比报告 (FNN vs RNN vs LSTM)\n")
    
    for lang in ["zh", "en"]:
        lang_name = "中文 (Chinese)" if lang == "zh" else "英文 (English)"
        report_content.append(f"## {lang_name} 语料分析\n")
        
        models = {}
        for model_name, path in FILES[lang].items():
            print(f"Loading {lang} {model_name} embeddings...")
            models[model_name] = load_embeddings(path)
            
        # 1. Qualitative Analysis: Nearest Neighbors
        report_content.append("### 1. 词义相似度定性分析 (Nearest Neighbors)\n")
        report_content.append("选取了几个高频关键词，分别列出它们在不同模型下的 Top-5 近义词，以观察模型捕捉语义的能力差异。\n")
        
        for word in TEST_WORDS[lang]:
            report_content.append(f"#### 关键词: **{word}**\n")
            report_content.append("| Rank | FNN | RNN | LSTM |")
            report_content.append("|---|---|---|---|")
            
            neighbors = {}
            for name in ["FNN", "RNN", "LSTM"]:
                if name in models and word in models[name]:
                    neighbors[name] = get_nearest_neighbors(models[name], word, k=5)
                else:
                    neighbors[name] = [("-", 0.0)] * 5
            
            for i in range(5):
                row = f"| {i+1} |"
                for name in ["FNN", "RNN", "LSTM"]:
                    w, score = neighbors[name][i] if i < len(neighbors[name]) else ("-", 0.0)
                    row += f" {w} ({score:.3f}) |"
                report_content.append(row)
            report_content.append("\n")

        # 2. Quantitative Analysis: Model Consistency
        report_content.append("### 2. 模型一致性定量分析 (Model Consistency)\n")
        report_content.append("计算不同模型之间 Top-10 近义词的重叠度 (Jaccard Similarity)，以评估模型学习到的语义空间的一致性。\n")
        
        # Find common vocabulary
        common_vocab = set(models["FNN"].keys())
        for name in ["RNN", "LSTM"]:
            if name in models:
                common_vocab = common_vocab.intersection(models[name].keys())
        
        # Select top N frequent words from common vocab (assuming vocab list order is somewhat frequency based or just take first N)
        # Since dicts are insertion ordered in recent python and we built them from frequency, taking first N is fine.
        sample_words = list(common_vocab)[:100] 
        
        comparisons = [("FNN", "RNN"), ("RNN", "LSTM"), ("FNN", "LSTM")]
        
        report_content.append("| 模型对比 | 平均 Jaccard 相似度 (Top-10 Neighbors) |")
        report_content.append("|---|---|")
        
        for m1, m2 in comparisons:
            if m1 not in models or m2 not in models: continue
            
            total_overlap = 0
            count = 0
            for w in sample_words:
                n1 = get_nearest_neighbors(models[m1], w, k=10)
                n2 = get_nearest_neighbors(models[m2], w, k=10)
                if n1 and n2:
                    overlap = calculate_overlap(n1, n2)
                    total_overlap += overlap
                    count += 1
            
            avg_overlap = total_overlap / count if count > 0 else 0
            report_content.append(f"| {m1} vs {m2} | {avg_overlap:.4f} |")
        
        report_content.append("\n")
        
    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_content))
    
    print(f"Report generated at {REPORT_PATH}")

if __name__ == "__main__":
    main()
