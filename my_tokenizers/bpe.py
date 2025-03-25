from pathlib import Path
import json
from tqdm import tqdm
from collections import Counter
from multiprocessing import Pool
import re
from .tokenizer import Tokenizer
from .utils import split_to_process
    
class BPE(Tokenizer):
    def __init__(self, vocab_size=1000, special_tokens: list[str] = ['<unk>'], parallel_num: int = 1):
        super().__init__(vocab_size, special_tokens)
        self.merges = []
        self.parallel_num = parallel_num
    
    def __len__(self):
        return len(self.vocab)
    
    def train(self, corpus: str) -> None:
        self._init_vocab(corpus)
        self._construct_trie()
        word_freqs = self._pretokenize(corpus)
        self._train_from_word_freqs(word_freqs)

    def _train_from_word_freqs(self, word_freqs: Counter[str]) -> None:
        pbar = tqdm(total=self.vocab_size-len(self), desc="Training BPE")
        while len(self) < self.vocab_size:
            if all(word.count(' ')==1 for word in word_freqs):
                print("All words are single tokens, stopping training.")
                break
            word_freqs = self._train_single_epoch(word_freqs)
            pbar.update(1)
            pbar.refresh()
        # print(list(word_freqs.items())[-30:])
        pbar.close()
        
    def train_increment(self, corpus: str) -> None:
        word_freqs = self._pretokenize(corpus)
        self._train_from_word_freqs(word_freqs)
    
    def _init_vocab(self, corpus: str) -> None:
        end_tokens = re.findall(r'(\w\ )', corpus)
        self.vocab = self.special_tokens + list(set(corpus)) + list(set(end_tokens))
        print(self.vocab)
        print(f"Initial vocabulary: {len(self)}")
        
        if len(self.vocab) >= self.vocab_size:
            print(f"Vocabulary size {len(self.vocab)} is larger than specified size {self.vocab_size}.")
            self.vocab = self.vocab[:self.vocab_size]
        
        self.token2ids = {char: idx for idx, char in enumerate(self.vocab)}
        self.id2tokens = {idx: char for char, idx in self.token2ids.items()}
    
    def _pretokenize(self, corpus: str) -> Counter[str]:
        word_freqs = Counter(corpus.split())
        processed_word_freqs = {}
        for word, count in tqdm(word_freqs.items(), desc="Pre-tokenizing corpus"):
            word = word.strip() + ' '
            word_ids = self.encode(word, verbose=False)
            word = ' '.join(self.id2tokens[i] for i in word_ids)
            processed_word_freqs[word] = count
        return processed_word_freqs
    
    def _train_single_epoch(self, word_freqs: Counter[str]) -> Counter[str]:
        if self.parallel_num > 1:
            counter = self._count_pairs(word_freqs)
        else:
            counter = self._count_pairs_worker(word_freqs.items())
        
        most_freq_pair: tuple[str, str] = max(counter, key=counter.get)
        self._update_vocab(most_freq_pair)
        return self._merge_pair(word_freqs, most_freq_pair)
    
    def _count_pairs(self, word_freqs: Counter[str]) -> Counter[tuple[str, str]]:
        chunks = split_to_process(list(word_freqs.items()), self.parallel_num)
        with Pool(self.parallel_num) as pool:
            results = pool.map(self._count_pairs_worker, chunks)
        return sum(results, Counter())
            
    def _count_pairs_worker(self, chunk: list[tuple[str, int]]) -> Counter[tuple[str, str]]:
        counter = Counter()
        for word, freq in chunk:
            tokens = word.strip().split()
            tokens[-1] += ' '
            for pair in zip(tokens[:-1], tokens[1:]):
                counter[pair] += freq
        return counter
    
    def _update_vocab(self, pair: tuple[str, str]) -> tuple[int, str]:
        self.merges.append(pair)
        token_pair = pair[0] + pair[1]
        self.vocab.append(token_pair)
        new_id = len(self)
        self.token2ids[token_pair] = new_id
        self.id2tokens[new_id] = token_pair
        self.trie[token_pair] = new_id
        return new_id, token_pair
    
    def _merge_pair(self, word_freqs: Counter[str], pair: tuple[str, str]) -> Counter[str]:
        token_pair = pair[0] + ' ' + pair[1]
        new_token = pair[0] + pair[1]
        new_word_freqs = {}
        for word, freq in word_freqs.items():
            new_word = word.replace(token_pair, new_token)
            new_word_freqs[new_word] = freq
        return new_word_freqs

    @property
    def data(self) -> dict:
        return {
            "vocab_size": self.vocab_size,
            "special_tokens": self.special_tokens,
            "id2tokens": self.id2tokens,
            "merges": self.merges
        }
    
    @classmethod
    def load(cls, path: str | Path) -> 'BPE':
        with open(path, 'r') as f:
            data = json.load(f)
        bpe = cls(data["vocab_size"], data["special_tokens"])
        bpe._load(data)
        return bpe
    
    def _load(self, data: dict) -> None:
        super()._load(data)
        self.merges = [tuple(pair) for pair in data["merges"]]