import os
import random
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Paths
BASE_DIR = r"E:\homework\nlp\hw2\question1"
OUTPUT_FILE = r"E:\homework\nlp\hw2\question3\raw_results.txt"

FILES = {
    "zh": os.path.join(BASE_DIR, "word_vectors_lstm_zh.txt"),
    "en": os.path.join(BASE_DIR, "word_vectors_lstm_en.txt"),
}

def load_embeddings(file_path):
    embeddings = {}
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            # Skip header
            f.readline()
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2: continue
                word = parts[0]
                # Skip special tokens
                if word in ['<PAD>', '<UNK>']: continue
                vec = np.array([float(x) for x in parts[1:]], dtype=np.float32)
                embeddings[word] = vec
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
    return embeddings

def get_nearest_neighbors(embeddings, word, k=10):
    if word not in embeddings:
        return []
    
    target_vec = embeddings[word].reshape(1, -1)
    words = list(embeddings.keys())
    matrix = np.array([embeddings[w] for w in words])
    
    sims = cosine_similarity(target_vec, matrix)[0]
    sorted_indices = sims.argsort()[::-1]
    
    neighbors = []
    count = 0
    for idx in sorted_indices:
        w = words[idx]
        if w == word: continue
        neighbors.append((w, sims[idx]))
        count += 1
        if count >= k: break
        
    return neighbors

def main():
    results = []
    
    for lang, file_path in FILES.items():
        print(f"Processing {lang}...")
        embeddings = load_embeddings(file_path)
        if not embeddings:
            continue
            
        vocab = list(embeddings.keys())
        # Filter for words that are likely to have meaning (length > 1 for EN, or just random)
        # To make the "judgment" meaningful, let's try to pick words that are somewhat frequent 
        # (assuming the vocab list is somewhat ordered or we just pick random)
        # The previous scripts built vocab by frequency, so the first ones are most frequent.
        # Let's pick 10 from top 500 (frequent) and 10 from 500-2000 (medium) to get a mix.
        
        top_words = vocab[:500]
        mid_words = vocab[500:2000]
        
        selected_words = []
        if len(top_words) >= 10:
            selected_words.extend(random.sample(top_words, 10))
        else:
            selected_words.extend(top_words)
            
        if len(mid_words) >= 10:
            selected_words.extend(random.sample(mid_words, 10))
        else:
            selected_words.extend(random.sample(vocab, min(10, len(vocab))))
            
        # Ensure unique and exactly 20 if possible
        selected_words = list(set(selected_words))[:20]
        
        results.append(f"=== {lang.upper()} Results ===")
        for word in selected_words:
            neighbors = get_nearest_neighbors(embeddings, word)
            neighbor_str = ", ".join([f"{w}({s:.3f})" for w, s in neighbors])
            results.append(f"WORD: {word}")
            results.append(f"NEIGHBORS: {neighbor_str}")
            results.append("-" * 20)
            
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write('\n'.join(results))
    print(f"Results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
