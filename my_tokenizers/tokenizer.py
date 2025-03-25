from abc import ABC, abstractmethod
from pathlib import Path
from pygtrie import CharTrie
from tqdm import tqdm
import json

class Tokenizer(ABC):
    def __init__(self, vocab_size: int = 1000, special_tokens:list[str]=['<unk>']):
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens
        self.id2tokens: dict[int, str] = {}
        self.token2ids: dict[str, int] = {}
        self.trie = CharTrie()
    
    @abstractmethod
    def train(self, corpus: str) -> None:
        pass
    
    @abstractmethod
    def train_increment(self, corpus: str) -> None:
        pass
    
    def _construct_trie(self) -> None:
        """构建前缀树"""
        for id_, token in enumerate(self.vocab):
            self.trie[token] = id_
    
    def encode(self, text: str, verbose: bool=True) -> list[int]:
        """使用前缀树进行最长匹配编码"""
        ids = []
        l = 0
        unk_id = self.token2ids['<unk>']
        
        pbar = tqdm(total=len(text), desc="Encoding", disable=not verbose)
        while l < len(text):
            prefix, token_id = self.trie.longest_prefix(text[l:])
            if not prefix:
                l += 1
                ids.append(unk_id)
            else:
                ids.append(token_id)
                l += len(prefix)
            pbar.n = l
            pbar.refresh()
        pbar.close()
        return ids
    
    def decode(self, ids: list[int]) -> str:
        return ''.join(self.id2tokens[i] for i in ids)
    
    def save(self, path: str | Path) -> None:
        with open(path, 'w') as f:
            json.dump(self.data, f, indent=4, ensure_ascii=False)
    
    @property
    @abstractmethod
    def data(self) -> dict:
        pass
    
    def _load(self, data: dict) -> None:
        self.vocab_size = data["vocab_size"]
        self.special_tokens = data["special_tokens"]
        self.id2tokens = {int(k): v for k, v in data["id2tokens"].items()}
        self.token2ids = {v: int(k) for k, v in self.id2tokens.items()}
        self._construct_trie()