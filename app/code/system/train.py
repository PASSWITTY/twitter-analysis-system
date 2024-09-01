import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from model import SentimentAnalysisModel
import json

embedding_dim = 100
hidden_dim = 128
output_dim = 3
n_layers = 2
bidirectional = True
dropout = 0.5
n_epochs = 10
batch_size = 32


with open('vocabulary.json') as f:
    vocabulary = json.load(f)

vocab_size = len(vocabulary) + 1


model = SentimentAnalysisModel(vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())


train_sequences_padded, train_labels = torch.load('train_set.pt')
dev_sequences_padded, dev_labels = torch.load('dev_set.pt')

# DataLoader for batching
train_data = TensorDataset(train_sequences_padded, train_labels)
train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)

dev_data = TensorDataset(dev_sequences_padded, dev_labels)
dev_loader = DataLoader(dev_data, batch_size=batch_size)

# Training loop
for epoch in range(n_epochs):
    model.train()
    epoch_loss = 0
    
    for sequences, labels in train_loader:
        optimizer.zero_grad()
        predictions = model(sequences)
        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    dev_loss = 0
    correct = 0
    total = 0
    model.eval()
    
    with torch.no_grad():
        for sequences, labels in dev_loader:
            predictions = model(sequences)
            loss = criterion(predictions, labels)
            dev_loss += loss.item()
            _, predicted = torch.max(predictions, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = correct / total
    print(f'Epoch {epoch+1}/{n_epochs}, Training Loss: {epoch_loss/len(train_loader)}, Validation Loss: {dev_loss/len(dev_loader)}, Accuracy: {accuracy}')
    
# Save the model weights after training
torch.save(model.state_dict(), 'sentiment_analysis_model.pth')
