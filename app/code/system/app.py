from flask import Flask, request, jsonify
import torch
from model import SentimentAnalysisModel
import re
import json
from nltk.corpus import stopwords
from torch.nn.utils.rnn import pad_sequence
from flask_cors import CORS 

# Ensure you have the necessary NLTK data
import nltk
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load vocabulary
with open('vocabulary.json') as f:
    vocabulary = json.load(f)

vocab_size = len(vocabulary) + 1  # +1 for padding index

# Hyperparameters (must match those used during training)
embedding_dim = 100
hidden_dim = 128
output_dim = 3  # Negative, Neutral, Positive
n_layers = 2
bidirectional = True
dropout = 0.5

# Instantiate the model
model = SentimentAnalysisModel(vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout)

# Load the model weights
model.load_state_dict(torch.load('sentiment_analysis_model.pth'))
model.eval()

def clean_text(text):
    """Clean the input text by removing special characters, converting to lowercase, and removing stopwords."""
    text = re.sub(r'\W', ' ', text)
    text = text.lower()
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

def tokens_to_indices(tokens, vocabulary):
    """Convert tokens to their corresponding indices in the vocabulary."""
    return [vocabulary.get(word, 0) for word in tokens]

def pad_sequences(sequences, max_len):
    """Pad sequences to the same length."""
    padded_sequences = pad_sequence([torch.tensor(seq) for seq in sequences], batch_first=True, padding_value=0)
    if padded_sequences.size(1) < max_len:
        padding = torch.zeros((padded_sequences.size(0), max_len - padded_sequences.size(1)), dtype=torch.long)
        padded_sequences = torch.cat((padded_sequences, padding), dim=1)
    return padded_sequences[:, :max_len]

MAX_SEQUENCE_LENGTH = 50

def preprocess_text(text):
    """Preprocess input text for model prediction."""
    cleaned_text = clean_text(text)
    tokens = cleaned_text.split()
    indices = tokens_to_indices(tokens, vocabulary)
    padded_sequence = pad_sequences([indices], MAX_SEQUENCE_LENGTH)
    return padded_sequence

def predict_sentiment(text):
    """Predict the sentiment of the input text."""
    input_sequence = preprocess_text(text)
    with torch.no_grad():
        output = model(input_sequence)
        _, predicted = torch.max(output, 1)
        return predicted.item()

# Define a mapping from label indices to sentiment labels
index_to_label = {1: 'Negative', 0: 'Neutral', 2: 'Positive'}

app = Flask(__name__)
CORS(app)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    text = data.get('text')
    if text:
        prediction = predict_sentiment(text)
        return jsonify({'sentiment': index_to_label[prediction]})
    else:
        return jsonify({'error': 'No text provided'}), 400

if __name__ == '__main__':
    app.run(debug=True)
