from multiprocessing import Pool
from typing import Sequence, Iterable, TypeVar
from collections import Counter

Elem = TypeVar('Elem')

def split_to_process(seq: Sequence[Elem], n: int) -> list[Sequence[Elem]]:
    """将列表均匀分割为n个块"""
    chunk_size = len(seq) // n
    remainder = len(seq) % n
    chunks = []
    start = 0
    for i in range(n):
        end = start + chunk_size + (1 if i < remainder else 0)
        chunks.append(seq[start:end])
        start = end
    return chunks
    