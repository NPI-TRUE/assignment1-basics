from collections import Counter, defaultdict
from typing import BinaryIO

from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor
import multiprocessing

import heapq

import os
import time
import threading

import json
import regex as re

num_processes = min(8, multiprocessing.cpu_count())


def bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    # now get the representations of the other 68 integers that do need shifting
    # each will get mapped chr(256 + n), where n will grow from 0...67 in the loop
    # Get printable representations of the remaining integers 68 integers.
    n = 0
    for b in range(2**8):
        if b not in bs:
            # If this integer isn't in our list of visually-representable
            # charcters, then map it to the next nice character (offset by 256)
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    characters = [chr(n).encode('utf-8') for n in cs]
    d = dict(zip(bs, characters))
    return d

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def count_chunk(args):
    input_path, special_tokens, start, end, PAT = args

    PAT = re.compile(PAT)

    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")

    words = defaultdict(int)

    for text in re.split('|'.join(special_tokens), chunk):
        for match in re.finditer(PAT, text):
            word = match.group()
            words[tuple(word.encode('utf-8'))] += 1

    return Counter(words)

def get_pre_tokens(input_path: str, speical_tokens: list, PAT: str) -> dict:
    global num_processes
    
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, 50, b"<|endoftext|>")

    print(f"Total cpu: {multiprocessing.cpu_count()}, used: {num_processes}")

    words = Counter({})

    chunks = []

    for start, end in zip(boundaries[:-1], boundaries[1:]):
        chunks.append((input_path, speical_tokens, start, end, PAT))

    tot_cunks = len(chunks)

    print("Start reading chunks")
    
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        for i, counter in enumerate(executor.map(count_chunk, chunks)):
            print(f"Chunk: {i + 1}/{tot_cunks}")

            words += counter

    print()

    return dict(words)

class BPETokenizer():
    def __init__(self):
        self.byte_ecoder= bytes_to_unicode()
        self.vocab = {k:v for k, v in zip(range(len(self.byte_ecoder)), self.byte_ecoder.values())}
        self.encoder = {v:k for k, v in self.vocab.items()}
        self.merges = []

        self.PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

        self.lock = threading.Lock()

        self.folder = '.data'
        os.makedirs(self.folder, exist_ok=True)

    def merge(self, update_pair, token, pair, ids_list):
        for idx in ids_list:
            with self.lock:
                word, cnt = self.words[idx]
            new_word = []

            i = 0
            while i < len(word):
                if i + 1 < len(word) and word[i] == pair[0] and word[i + 1] == pair[1]:
                    left = (word[i - 1], word[i]) if i - 1 >= 0 else None
                    right = (word[i + 1], word[i + 2]) if i + 2 < len(word) else None
                    
                    with self.lock:
                        if left in self.pairs:
                            if self.pairs[left] > 0:
                                self.pairs[left] -= cnt

                        if right in self.pairs:
                            if self.pairs[right] > 0:
                                self.pairs[right] -= cnt
                    
                    new_word.append(self.vocab[token])

                    if i - 1 >= 0:
                        tpair = (word[i - 1], self.vocab[token]) 
                        with self.lock:
                            self.pairs[tpair] += cnt
                            self.ids[tpair].add(idx)
                            update_pair.add(tpair)

                    if i + 2 < len(word):
                        tpair = (self.vocab[token], word[i + 2]) 
                        with self.lock:
                            self.pairs[tpair] += cnt
                            self.ids[tpair].add(idx)
                            update_pair.add(tpair)

                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            
            with self.lock:
                self.words[idx] = (new_word, cnt)

    def train(
        self,
        input_path: str,
        vocab_size: int,
        special_tokens: list[str],
    ) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        """Given the path to an input corpus, run train a BPE tokenizer and
        output its vocabulary and merges.

        Args:
            input_path (str | os.PathLike): Path to BPE tokenizer training data.
            vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
            special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
                These strings will never be split into multiple tokens, and will always be
                kept as a single token. If these special tokens occur in the `input_path`,
                they are treated as any other string.

        Returns:
            tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
                vocab:
                    The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                    to bytes (token bytes)
                merges:
                    BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                    representing that <token1> was merged with <token2>.
                    Merges are ordered by order of creation.
        """
        from tqdm import tqdm

        global num_processes

        self.special_tokens = special_tokens
        for stoken in special_tokens:
            self.vocab[len(self.vocab)] = stoken

        num_merges = vocab_size - len(self.vocab)
        pre_tokens = get_pre_tokens(input_path, self.special_tokens, self.PAT)
        self.words = [([self.byte_ecoder[ch] for ch in word], cnt) for word, cnt in pre_tokens.items()]

        self.pairs = defaultdict(int)
        self.ids = defaultdict(set)
        pq = []

        for idx, (word, cnt) in enumerate(self.words):
            for a, b in zip(word, word[1:]):
                pair = (a, b)
                self.pairs[pair] += cnt
                self.ids[pair].add(idx)

        for (a, b), cnt in self.pairs.items():
            heapq.heappush(pq, (-cnt, tuple((a, b))))

        print(f"Debug:      len(pairs): {len(self.pairs)}  -  len(words): {len(self.words)}  -  len(ids): {len(self.ids)}", "\n")

        print("Start merging")
        
        for iMerge in tqdm(range(1, num_merges + 1)):
            pcnt, pair = heapq.heappop(pq)
            pcnt = -pcnt

            while (self.pairs[pair] != pcnt) and len(pq) != 0:
                if (self.pairs[pair] > 0):
                    heapq.heappush(pq, (-self.pairs[pair], pair))

                pcnt, pair = heapq.heappop(pq)
                pcnt = -pcnt

            if not pair:
                break

            token = len(self.vocab)
            self.vocab[token] = pair[0] + pair[1]
            self.merges.append((pair[0], pair[1]))

            start_range = 0
            word_per_thread = len(self.ids[pair]) // num_processes
            #print(f"DEBUG:  len(ids[pair]): {len(ids[pair])}  num_threads: {num_threads}  word_per_threads: {word_per_thread}")

            threads = []
            update_pair = set()
            _ids = list(self.ids[pair])

            for i in range(num_processes):
                t = None
                if i != num_processes - 1:
                    t = threading.Thread(target=self.merge, args=(update_pair, token, pair, _ids[start_range:start_range + word_per_thread]))
                else:
                    t = threading.Thread(target=self.merge, args=(update_pair, token, pair, _ids[start_range:]))
                    
                t.start()
                threads.append(t)
                start_range += word_per_thread

            for t in threads:
                t.join()

            for tpair in update_pair:
                if tpair[1] == self.vocab[token]:
                    heapq.heappush(pq, (-self.pairs[tpair], (tpair[0], self.vocab[token])))
                else:
                    heapq.heappush(pq, (-self.pairs[tpair], (self.vocab[token], tpair[1])))


            del self.pairs[pair]
            del self.ids[pair]

        return self.vocab, self.merges

if __name__ == "__main__":
    bpe = BPETokenizer()
    vocab, merges = bpe.train("../tests/fixtures/corpus.en", 500, ["<|endoftext|>"])

    print(vocab, "\n\n", merges)