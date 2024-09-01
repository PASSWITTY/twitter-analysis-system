import torch
from torch.utils.data import DataLoader, TensorDataset
from model import SentimentAnalysisModel
import json 

# Load vocabulary
with open('vocabulary.json') as f:
    vocabulary = json.load(f)

vocab_size = len(vocabulary) + 1
# Hyperparameters
embedding_dim = 100
hidden_dim = 128
output_dim = 3
n_layers = 2
bidirectional = True
dropout = 0.5
batch_size = 32

# Instantiate the model
model = SentimentAnalysisModel(vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout)

# Load the model weights
model.load_state_dict(torch.load('sentiment_analysis_model.pth'))
model.eval()

# Load the validation set
val_sequences_padded, val_labels = torch.load('validation_set.pt')

# DataLoader for batching
val_data = TensorDataset(val_sequences_padded, val_labels)
val_loader = DataLoader(val_data, batch_size=batch_size)

correct = 0
total = 0

with torch.no_grad():
    for sequences, labels in val_loader:
        predictions = model(sequences)
        _, predicted = torch.max(predictions, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = correct / total
print(f'Validation Accuracy: {accuracy}')
