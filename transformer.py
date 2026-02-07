import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def get_device():
    if torch.backends.mps.is_available():
        device = 'mps'
    elif torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    
    return torch.device(device)

def scaled_dot_product(q, k, v, mask=None):
    d_k = q.size(-1)
    scaled = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scaled = scaled.permute(1, 0, 2, 3) + mask
        scaled = scaled.permute(1, 0, 2, 3)

    attention = F.softmax(scaled, dim=-1)
    values = torch.matmul(attention, v)

    return values, attention

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max__sequence_length):
        super().__init__()
        self.max_sequence_length = max__sequence_length
        self.d_model = d_model

    def forward(self, x):
        even_i = torch.arange(0, self.d_model, 2).float()
        denominator = torch.pow(10000, even_i / self.d_model)
        position = (torch.arange(self.max_sequence_length).reshape(self.max_sequence_length, 1))
        even_PE = torch.sin(position / denominator)
        odd_PE = torch.cos(position / denominator)
        stacked = torch.stack([even_PE, odd_PE], dim=2)
        PE = torch.flatten(stacked, start_dim=1, end_dim=2)

        return PE
    
class SentenceEmbedding(nn.Module):
    def __init__(self, max_sequence_length, d_model, language_to_index, START_TOKEN, END_TOKEN, PADDING_TOKEN):
        super().__init__()
        self.vocab_size = len(language_to_index)
        self.max_sequence_length = max_sequence_length
        self.embedding = nn.Embedding(self.vocab_size, d_model)
        self.language_to_index = language_to_index
        self.position_encoder = PositionalEncoding(d_model, max_sequence_length)
        self.START_TOKEN = START_TOKEN
        self.END_TOKEN = END_TOKEN
        self.PADDING_TOKEN = PADDING_TOKEN

    def batch_tokenize(self, batch, START_TOKEN, END_TOKEN):
        
        def tokenize(sentence, START_TOKEN, END_TOKEN):
            sentence_word_indices = [self.language_to_index[token] for token in list(sentence)]

            if START_TOKEN:
                sentence_word_indices.insert(0, self.language_to_index[self.START_TOKEN])
            if END_TOKEN:
                sentence_word_indices.append(self.language_to_index[self.END_TOKEN])
            
            for _ in range(len(sentence_word_indices), self.max_sequence_length):
                sentence_word_indices.append(self.language_to_index[self.PADDING_TOKEN])

            return torch.tensor(sentence_word_indices)

        tokenized = []

        for i in range(len(batch)):
            tokenized.append(tokenize(batch[i], START_TOKEN, END_TOKEN))
        
        return torch.stack(tokenized).to(get_device())

    def forward(self, x, START_TOKEN, END_TOKEN):
        x = self.batch_tokenize(x, START_TOKEN, END_TOKEN)
        x = self.embedding(x)
        pos = self.position_encoder(x).to(get_device())
        x = self.dropout(x + pos)

        return x