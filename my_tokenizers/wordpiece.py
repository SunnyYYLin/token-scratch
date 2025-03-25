from pathlib import Path
import json
from tqdm import tqdm
from collections import Counter
from multiprocessing import Pool
import re
from .tokenizer import MyTokenizer
from .utils import split_to_process
    
class MyWordPiece(MyTokenizer):
    """
    WordPiece 分词器实现。
    该类实现了 WordPiece 分词器，这是一种常用于自然语言处理任务的子词分词算法。
    它支持通过语料库训练来构建子词标记的词汇表，并将文本编码为标记 ID。
    方法：
        __init__(vocab_size=1000, special_tokens: list[str] = ['<unk>'], parallel_num: int = 1):
            使用指定的词汇表大小、特殊标记和并行处理配置初始化 WordPiece 分词器。
        __len__():
            返回当前词汇表的大小。
        train(corpus: str) -> None:
            在给定语料库上训练分词器以构建词汇表。
        train_increment(corpus: str) -> None:
            在额外的语料数据上增量训练分词器。
        _init_vocab(corpus: str) -> None:
            从给定的语料库初始化词汇表。
        _pretokenize(corpus: str) -> Counter[str]:
            将语料库预分词为词频表。另外给单词内部插入空格。
        _train_from_word_freqs(word_freqs: Counter[str]) -> None:
            使用词频表训练分词器。
        _train_single_epoch(word_freqs: Counter[str]) -> Counter[str]:
            通过合并最可能的标记对执行单次训练迭代。
        _count_pairs(word_freqs: Counter[str]):
            统计语料库中标记对及其频率。
        _count_pairs_worker(chunk: list[tuple[str, int]]):
            用于并行统计标记对的工作函数。
        _score(token_freqs: Counter[str], pair_freqs: Counter[tuple[str, str]]):
            根据频率对标记对进行评分。
        _score_worker(token_freqs: Counter[str], pair_freqs: Counter[tuple[str, str]]):
            用于并行评分标记对的工作函数。
        _update_vocab(pair: tuple[str, str]) -> tuple[int, str]:
            使用通过合并标记对形成的新标记更新词汇表。
        _merge_pair(word_freqs: Counter[str], pair: tuple[str, str]) -> Counter[str]:
            在词频表中合并标记对。
        data -> dict:
            返回分词器的数据，包括词汇表大小、特殊标记和标记映射。
        load(path: str | Path) -> 'WordPiece':
            从文件加载 WordPiece 分词器。
        _load(data: dict) -> None:
            从字典加载分词器数据。
    """
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
        pbar.close()
        
    def train_increment(self, corpus: str) -> None:
        word_freqs = self._pretokenize(corpus)
        self._train_from_word_freqs(word_freqs)
    
    def _init_vocab(self, corpus: str) -> None:
        end_tokens = re.findall(r'(\w\ )', corpus)
        self.vocab = self.special_tokens + list(set(corpus)) + list(set(end_tokens))
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
            word_ids = self.encode(word, verbose=False)
            word = ' '.join(self.id2tokens[i] for i in word_ids) + ' '
            processed_word_freqs[word] = count
        return processed_word_freqs
    
    def _train_single_epoch(self, word_freqs: Counter[str]) -> Counter[str]:
        token_counter, pair_counter = self._count_pairs(word_freqs)
        pair_scores = self._score(token_counter, pair_counter)
        most_probable_pair = max(pair_scores, key=pair_scores.get)
        self._update_vocab(most_probable_pair)
        return self._merge_pair(word_freqs, most_probable_pair)
    
    def _count_pairs(self, word_freqs: Counter[str]):
        if self.parallel_num > 1:
            chunks = split_to_process(list(word_freqs.items()), self.parallel_num)
            with Pool(self.parallel_num) as pool:
                results = pool.map(self._count_pairs_worker, chunks)
            return sum(zip(*results)[0], Counter()), sum(zip(*results)[1], Counter())
        else:
            return self._count_pairs_worker(word_freqs.items())
            
    def _count_pairs_worker(self, chunk: list[tuple[str, int]]):
        pair_counter = Counter()
        token_counter = Counter()
        for word, freq in chunk:
            tokens = word.strip().split()
            tokens[-1] += ' '
            for pair in zip(tokens[:-1], tokens[1:]):
                pair_counter[pair] += freq
                token_counter[pair[0]] += freq
            token_counter[tokens[-1]] += freq
        return token_counter, pair_counter
    
    def _score(self, token_freqs: Counter[str], pair_freqs: Counter[tuple[str, str]]):
        if self.parallel_num > 1:
            chunks = split_to_process(list(pair_freqs.items()), self.parallel_num)
            with Pool(self.parallel_num) as pool:
                results = pool.map(self._score_worker, chunks)
            return sum(results, {})
        else:
            return self._score_worker(token_freqs, pair_freqs)
    
    def _score_worker(self, token_freqs: Counter[str], pair_freqs: Counter[tuple[str, str]]):
        pair_scores: dict[tuple[str, str], float] = {}
        for pair, freq in pair_freqs.items():
            token1, token2 = pair
            score = freq / (token_freqs[token1] * token_freqs[token2])
            pair_scores[pair] = score
        return pair_scores
    
    def _update_vocab(self, pair: tuple[str, str]) -> tuple[int, str]:
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
        }
    
    @classmethod
    def load(cls, path: str | Path) -> 'WordPiece':
        with open(path, 'r') as f:
            data = json.load(f)
        bpe = cls(data["vocab_size"], data["special_tokens"])
        bpe._load(data)
        return bpe
    
    def _load(self, data: dict) -> None:
        super()._load(data)
        self.merges = [tuple(pair) for pair in data["merges"]]