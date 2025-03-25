import matplotlib.pyplot as plt
from tokenizers import Tokenizer

def compute_compression_ratio(text, tokenizer: Tokenizer):
    num_utf8_bytes = len(text.encode('utf-8'))
    encoded_ids = tokenizer.encode(text)
    if hasattr(encoded_ids, 'ids'):
        num_encoded_ids = len(encoded_ids.ids)
    else:
        num_encoded_ids = len(encoded_ids)
    return num_utf8_bytes / num_encoded_ids