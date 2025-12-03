import torch
import torch.nn as nn
import re
import collections
import os

def get_zh_tokens(file_path):
    tokens = []
    with open(file_path, 'r', encoding='gbk', errors='ignore') as f:
        # The file is likely GBK or UTF-8. The snippet showed Chinese characters correctly, 
        # but standard 1998 corpus is often GBK. I'll try GBK first, if fail then UTF-8.
        # Actually, the read_file output was correct, so the system handles it. 
        # Python open() default encoding depends on OS. 
        # Let's try reading with 'utf-8' first as it is standard now, but fallback if needed.
        # However, standard People's Daily 98 is often GBK.
        # Let's read binary and decode to be safe or just try utf-8.
        try:
            content = f.read()
        except UnicodeDecodeError:
            f = open(file_path, 'r', encoding='gbk', errors='ignore')
            content = f.read()
            
    lines = content.split('\n')
    for line in lines:
        if not line.strip():
            continue
        parts = line.strip().split()
        # Skip the first part (ID)
        if len(parts) > 1:
            words = parts[1:]
            for w in words:
                # Remove [ if present (start of compound)
                if w.startswith('['):
                    w = w[1:]
                # Split by / and take the word part
                # Handle cases like 1/2/m (fractions) -> 1/2
                # Usually the tag is the last part after /
                idx = w.rfind('/')
                if idx != -1:
                    word = w[:idx]
                    if word:
                        tokens.append(word)
    return tokens

def get_en_tokens(file_path):
    tokens = []
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip().lower()
            if not line:
                continue
            # Simple tokenization: keep words and numbers
            words = re.findall(r'\w+', line)
            tokens.extend(words)
    return tokens

def build_vocab(tokens, vocab_size=5000):
    counter = collections.Counter(tokens)
    # Reserve 0 for <PAD>, 1 for <UNK>
    most_common = counter.most_common(vocab_size - 2)
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for word, _ in most_common:
        vocab[word] = len(vocab)
    idx2word = {idx: word for word, idx in vocab.items()}
    return vocab, idx2word

class NgramDataset(torch.utils.data.Dataset):
    def __init__(self, tokens, vocab, n_gram=3):
        self.tokens = tokens
        self.vocab = vocab
        self.n_gram = n_gram
        self.data = []
        
        # Convert tokens to indices
        self.token_ids = [self.vocab.get(t, self.vocab['<UNK>']) for t in tokens]
        
        # Create sliding windows
        for i in range(len(self.token_ids) - n_gram):
            context = self.token_ids[i:i+n_gram]
            target = self.token_ids[i+n_gram]
            self.data.append((context, target))
            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        context, target = self.data[idx]
        return torch.tensor(context, dtype=torch.long), torch.tensor(target, dtype=torch.long)

def save_embeddings(model, idx2word, filename):
    # Assumes model has an 'embedding' layer or 'embeddings'
    if hasattr(model, 'embedding'):
        embed = model.embedding
    elif hasattr(model, 'embeddings'):
        embed = model.embeddings
    else:
        print("Could not find embedding layer to save.")
        return

    weights = embed.weight.data.cpu().numpy()
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(f"{len(idx2word)} {weights.shape[1]}\n")
        for idx, word in idx2word.items():
            if idx == 0: continue # Skip PAD
            vec_str = ' '.join([f"{x:.6f}" for x in weights[idx]])
            f.write(f"{word} {vec_str}\n")
    print(f"Embeddings saved to {filename}")

