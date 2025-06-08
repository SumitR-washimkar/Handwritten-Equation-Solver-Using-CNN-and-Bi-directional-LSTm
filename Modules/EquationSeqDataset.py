import os
from torch.utils.data import Dataset
from PIL import Image
import torch

class EquationSeqDataset(Dataset):
    def __init__(self, df, img_folder, tokenizer, transform=None, max_seq_len=100):
        self.df = df
        self.img_folder = img_folder
        self.tokenizer = tokenizer
        self.transform = transform
        self.max_seq_len = max_seq_len  # Added max length for padding/truncation

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_folder, self.df.iloc[idx, 0])
        image = Image.open(img_path).convert("RGB")

        # Augment images only if transform passed
        if self.transform:
            image = self.transform(image)

        label = self.df.iloc[idx, 1]

        # Tokenize and add special tokens
        label_ids = [self.tokenizer.token_to_id["<SOS>"]] + self.tokenizer.encode(label) + [self.tokenizer.token_to_id["<EOS>"]
]

        # Pad or truncate label to max_seq_len for uniform batch size
        if len(label_ids) < self.max_seq_len:
            label_ids += [self.tokenizer.token_to_id["<PAD>"]] * (self.max_seq_len - len(label_ids))
        else:
            label_ids = label_ids[:self.max_seq_len]

        return image, torch.tensor(label_ids)
