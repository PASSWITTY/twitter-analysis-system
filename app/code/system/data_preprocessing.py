import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
import torch
import json
from collections import Counter
from torch.nn.utils.rnn import pad_sequence

# Ensure you have the necessary NLTK data
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def clean_text(text):
    # Remove special characters
    text = re.sub(r'\W', ' ', text)
    # Convert text to lowercase
    text = text.lower()
    # Remove stopwords
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

# Load datasets
train_data = pd.read_csv('twitter_training.csv')
val_data = pd.read_csv('twitter_validation.csv')

# Rename columns for clarity
train_data.columns = ['id', 'entity', 'sentiment', 'text']
val_data.columns = ['id', 'entity', 'sentiment', 'text']

# Fill missing text entries with empty strings
train_data['text'] = train_data['text'].fillna('')
val_data['text'] = val_data['text'].fillna('')

# Clean the text data
train_data['cleaned_text'] = train_data['text'].apply(clean_text)
val_data['cleaned_text'] = val_data['text'].apply(clean_text)

print("Sample cleaned text from training data:")
print(train_data['cleaned_text'].head())

# Tokenize the text
train_data['tokens'] = train_data['cleaned_text'].apply(lambda x: x.split())
val_data['tokens'] = val_data['cleaned_text'].apply(lambda x: x.split())

print("Sample tokens from training data:")
print(train_data['tokens'].head())

# Build the vocabulary from the training data tokens
all_words = [word for tokens in train_data['tokens'] for word in tokens]
word_counts = Counter(all_words)
vocabulary = {word: index + 1 for index, (word, count) in enumerate(word_counts.items())}

print("Vocabulary size:", len(vocabulary))

# Save vocabulary
with open('vocabulary.json', 'w') as vocab_file:
    json.dump(vocabulary, vocab_file)

# Convert tokens to sequences of indices with handling unknown words
def tokens_to_indices(tokens, vocabulary):
    return [vocabulary.get(word, 0) for word in tokens]

train_data['sequences'] = train_data['tokens'].apply(lambda x: tokens_to_indices(x, vocabulary))
val_data['sequences'] = val_data['tokens'].apply(lambda x: tokens_to_indices(x, vocabulary))

# Pad sequences using PyTorch's pad_sequence
def pad_sequences(sequences, max_len):
    padded_sequences = pad_sequence([torch.tensor(seq) for seq in sequences], batch_first=True, padding_value=0)
    if padded_sequences.size(1) < max_len:
        padding = torch.zeros((padded_sequences.size(0), max_len - padded_sequences.size(1)), dtype=torch.long)
        padded_sequences = torch.cat((padded_sequences, padding), dim=1)
    return padded_sequences[:, :max_len]

# Set the maximum sequence length
MAX_SEQUENCE_LENGTH = 50

# Pad the sequences
train_sequences_padded = pad_sequences(train_data['sequences'].tolist(), MAX_SEQUENCE_LENGTH)
val_sequences_padded = pad_sequences(val_data['sequences'].tolist(), MAX_SEQUENCE_LENGTH)

print("Sample padded sequences from training data:")
print(train_sequences_padded[:5])

# Convert sentiment labels to indices
label_map = {'Negative': 0, 'Neutral': 1, 'Positive': 2, 'Irrelevant': 1}
train_data['label'] = train_data['sentiment'].map(label_map)
val_data['label'] = val_data['sentiment'].map(label_map)

# Split the training data into training and development sets
train_set, dev_set = train_test_split(train_data, test_size=0.2, random_state=42)

# Pad the sequences for training and development sets
train_sequences_padded = pad_sequences(train_set['sequences'].tolist(), MAX_SEQUENCE_LENGTH)
dev_sequences_padded = pad_sequences(dev_set['sequences'].tolist(), MAX_SEQUENCE_LENGTH)

# Save the processed datasets
torch.save((train_sequences_padded, torch.tensor(train_set['label'].values)), 'train_set.pt')
torch.save((dev_sequences_padded, torch.tensor(dev_set['label'].values)), 'dev_set.pt')
torch.save((val_sequences_padded, torch.tensor(val_data['label'].values)), 'validation_set.pt')

