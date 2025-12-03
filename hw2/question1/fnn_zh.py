import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import utils
import os

# Configuration
DATA_PATH = "../data/ChineseCorpus199801.txt"
OUTPUT_FILE = "word_vectors_fnn_zh.txt"
EMBED_DIM = 100
CONTEXT_SIZE = 3
VOCAB_SIZE = 5000
BATCH_SIZE = 128
EPOCHS = 2
LR = 0.001

class FNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, context_size):
        super(FNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.linear1 = nn.Linear(context_size * embed_dim, 128)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(128, vocab_size)
        
    def forward(self, inputs):
        # inputs: [batch, context_size]
        embeds = self.embedding(inputs).view((inputs.shape[0], -1))
        out = self.linear1(embeds)
        out = self.relu(out)
        out = self.linear2(out)
        return out

def main():
    print("Loading Chinese data...")
    tokens = utils.get_zh_tokens(DATA_PATH)
    print(f"Total tokens: {len(tokens)}")
    
    vocab, idx2word = utils.build_vocab(tokens, VOCAB_SIZE)
    print(f"Vocab size: {len(vocab)}")
    
    dataset = utils.NgramDataset(tokens, vocab, CONTEXT_SIZE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FNN(len(vocab), EMBED_DIM, CONTEXT_SIZE).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    print("Starting training...")
    for epoch in range(EPOCHS):
        total_loss = 0
        for i, (context, target) in enumerate(dataloader):
            context, target = context.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(context)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            if (i+1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{EPOCHS}], Step [{i+1}/{len(dataloader)}], Loss: {loss.item():.4f}")
                
    print("Training finished.")
    utils.save_embeddings(model, idx2word, OUTPUT_FILE)

if __name__ == "__main__":
    main()
