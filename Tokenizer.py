from collections import defaultdict
import nltk
from nltk.tokenize import TreebankWordTokenizer

nltk.download('punkt')

class Tokenizer:
    def __init__(self, lower=True):
        self.token_to_id = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.id_to_token = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.next_id = 4
        self.tokenizer = TreebankWordTokenizer()
        self.lower = lower

    def _tokenize(self, text):
        if self.lower:
            text = text.lower()
        return self.tokenizer.tokenize(text)

    def fit(self, texts):
        for text in texts:
            tokens = self._tokenize(text)
            for token in tokens:
                if token not in self.token_to_id:
                    self.token_to_id[token] = self.next_id
                    self.id_to_token[self.next_id] = token
                    self.next_id += 1

    def encode(self, text):
        tokens = self._tokenize(text)
        return [self.token_to_id.get(tok, 3) for tok in tokens] + [2]  # Append <EOS>

    def decode(self, ids):
        tokens = [self.id_to_token.get(i, "<UNK>") for i in ids]
        return " ".join(tokens).replace(" <EOS>", "").strip()

    def vocab_size(self):
        return len(self.token_to_id)