import torch
import torch.nn as nn

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device, sos_token_id=1, eos_token_id=2):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.sos_token_id = sos_token_id
        self.eos_token_id = eos_token_id


    def forward(self, images, targets=None, teacher_forcing_ratio=0.5, max_len=50):
        B = images.size(0)
    
        encoder_outputs = self.encoder(images)  # [B, Seq, output_dim]
        # --- Initialize decoder hidden state from encoder outputs ---
        B, Seq_len, enc_dim = encoder_outputs.size()
        hidden_dim = self.decoder.lstm.hidden_size  # Expected hidden state size by decoder LSTM

        # Compute mean of encoder outputs for each sample in batch
        mean_encoder = encoder_outputs.mean(dim=1)  # [B, enc_dim]

        # Match encoder dim to decoder hidden_dim (if needed)
        if enc_dim != hidden_dim:
            if not hasattr(self, 'enc2dec'):
                self.enc2dec = nn.Linear(enc_dim, hidden_dim).to(self.device)
            mean_encoder = self.enc2dec(mean_encoder)  # [B, hidden_dim]

        # Create (h_0, c_0)
        h_0 = mean_encoder.unsqueeze(0)  # [1, B, hidden_dim]
        c_0 = torch.zeros_like(h_0).to(self.device)  # [1, B, hidden_dim]
        hidden = (h_0, c_0)


        vocab_size = self.decoder.fc_out.out_features

        if targets is not None:
            max_len = targets.size(1)

        outputs = torch.zeros(B, max_len, vocab_size).to(self.device)

        inputs = torch.full((B,), self.sos_token_id, dtype=torch.long).to(self.device)

        for t in range(max_len):
            output, hidden, _ = self.decoder(inputs, hidden, encoder_outputs)
            outputs[:, t, :] = output

            if targets is not None and t < max_len - 1 and torch.rand(1).item() < teacher_forcing_ratio:
                inputs = targets[:, t + 1]
            else:
                inputs = output.argmax(1)

        return outputs
