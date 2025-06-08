import torch
import torch.nn as nn
import torch.nn.functional as F

class RNNDecoder(nn.Module):
    def __init__(self, hidden_dim, vocab_size, embedding_dim=256, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Attention layers
        self.attn = nn.Linear(hidden_dim + embedding_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)
        
        self.lstm = nn.LSTM(embedding_dim + hidden_dim, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_token, hidden, encoder_outputs):
        # input_token: [B]
        embedded = self.embedding(input_token)  # [B, E]
        embedded = self.dropout(embedded)
        embedded = embedded.unsqueeze(1)  # [B, 1, E]

        B, Seq_len, Enc_dim = encoder_outputs.size()

        # Repeat hidden state to concat with encoder outputs for attention
        hidden_state = hidden[0].squeeze(0)  # [B, H]
        hidden_state = hidden_state.unsqueeze(1).repeat(1, Seq_len, 1)  # [B, Seq, H]

        # Compute energy scores for attention
        energy = torch.tanh(self.attn(torch.cat((hidden_state, encoder_outputs), dim=2)))  # [B, Seq, H]
        attention = self.v(energy).squeeze(2)  # [B, Seq]

        attn_weights = F.softmax(attention, dim=1).unsqueeze(1)  # [B, 1, Seq]

        # Context vector as weighted sum of encoder outputs
        context = torch.bmm(attn_weights, encoder_outputs)  # [B, 1, Enc_dim]

        # LSTM input = concatenated embedding + context
        lstm_input = torch.cat((embedded, context), dim=2)  # [B, 1, E + Enc_dim]

        output, hidden = self.lstm(lstm_input, hidden)  # output: [B, 1, H]

        prediction = self.fc_out(output.squeeze(1))  # [B, vocab_size]

        return prediction, hidden, attn_weights.squeeze(1)  # also return attention weights for visualization
