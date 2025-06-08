import torch
from torch.nn.utils.rnn import pad_sequence

def custom_collate_fn(batch, pad_token_id=0):
    images, labels = zip(*batch)  # Unzip images and labels

    # Stack images into a batch tensor
    images = torch.stack(images, dim=0)

    # Ensure all labels are tensors for pad_sequence compatibility
    labels = [label if isinstance(label, torch.Tensor) else torch.tensor(label, dtype=torch.long) for label in labels]

    # Use pad_sequence to pad dynamically with specified pad_token_id
    labels_tensor = pad_sequence(labels, batch_first=True, padding_value=pad_token_id)  # [B, max_len]

    # Optionally, create attention masks for labels (1 where not pad, 0 where pad)
    attention_mask = (labels_tensor != pad_token_id).long()  # [B, max_len]

    return images, labels_tensor, attention_mask
