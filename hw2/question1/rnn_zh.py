import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import utils
import os

# Configuration
DATA_PATH = "../data/ChineseCorpus199801.txt"
OUTPUT_FILE = "word_vectors_rnn_zh.txt"
EMBED_DIM = 100
SEQ_LEN = 5 # Using 5 previous words to predict the next
VOCAB_SIZE = 5000
BATCH_SIZE = 128
EPOCHS = 2
LR = 0.001

class RNNModel(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super(RNNModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.RNN(embed_dim, 128, batch_first=True)
        self.linear = nn.Linear(128, vocab_size)
        
    def forward(self, inputs):
        # inputs: [batch, seq_len]
        embeds = self.embedding(inputs)
        # output: [batch, seq_len, hidden]
        out, h_n = self.rnn(embeds)
        # Predict based on the last hidden state of the sequence
        last_out = out[:, -1, :]
        logits = self.linear(last_out)
        return logits

def main():
    print("Loading Chinese data...")
    tokens = utils.get_zh_tokens(DATA_PATH)
    print(f"Total tokens: {len(tokens)}")
    
    vocab, idx2word = utils.build_vocab(tokens, VOCAB_SIZE)
    print(f"Vocab size: {len(vocab)}")
    
    # Reusing NgramDataset as it provides (context, target) pairs which fits our sequence prediction task
    dataset = utils.NgramDataset(tokens, vocab, SEQ_LEN)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RNNModel(len(vocab), EMBED_DIM).to(device)
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
