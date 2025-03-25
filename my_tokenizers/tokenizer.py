from abc import ABC, abstractmethod
from pathlib import Path
from pygtrie import CharTrie
from tqdm import tqdm
import json

class Tokenizer(ABC):
    """
    抽象基类：Tokenizer（分词器）。
    此类提供了一个框架，用于实现具有训练、编码、解码以及保存/加载分词数据方法的分词器。
    它使用前缀树（trie）在编码过程中实现高效的最长前缀匹配。
    
    属性:
        vocab_size (int): 词汇表的最大大小。默认为 1000。
        special_tokens (list[str]): 要包含在词汇表中的特殊标记列表。默认为 ['<unk>']。
        id2tokens (dict[int, str]): 从标记 ID 到标记字符串的映射。
        token2ids (dict[str, int]): 从标记字符串到标记 ID 的映射。
        trie (CharTrie): 用于高效标记匹配的前缀树。
    
    方法:
        train(corpus: str) -> None:
            抽象方法，用于在给定语料库上训练分词器。
        train_increment(corpus: str) -> None:
            抽象方法，用于在额外数据上增量训练分词器。
        _construct_trie() -> None:
            从当前词汇表构建前缀树（trie）。
        encode(text: str, verbose: bool = True) -> list[int]:
            使用最长前缀匹配将给定文本编码为标记 ID 列表。
        decode(ids: list[int]) -> str:
            将标记 ID 列表解码回原始文本。
        save(path: str | Path) -> None:
            将分词器的数据保存到指定路径的 JSON 文件中。
        data() -> dict:
            抽象属性，返回分词器的数据字典。
        _load(data: dict) -> None:
            从字典加载分词器数据并重建前缀树。
    """
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
        for id_, token in enumerate(self.vocab):
            self.trie[token] = id_
    
    def encode(self, text: str, verbose: bool=True) -> list[int]:
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