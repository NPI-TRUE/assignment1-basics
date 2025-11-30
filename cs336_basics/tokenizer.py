import os
from typing import BinaryIO
from collections import Counter, defaultdict
import regex as re
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import json
import time

from tests.common import gpt2_bytes_to_unicode

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
    characters = [chr(n) for n in cs]
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
    with open(input_path, "rb") as f:
        num_processes = min(10, multiprocessing.cpu_count())
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

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
        self.vocab = bytes_to_unicode()
        self.merges = []
        self.PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        self.folder = '.data'
        os.makedirs(self.folder, exist_ok=True)
        # Cache mapping token (str/bytes) -> tuple of decoder values
        # Avoid recomputing tuple(self.decoder[ch] for ch in token) repeatedly
        self._bytes_cache = {}

    def train(
        self,
        input_path: str,
        vocab_size: int,
        special_tokens: list[str]
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

        self.special_tokens = special_tokens
        for stoken in special_tokens:
            self.vocab[len(self.vocab)] = stoken

        self.decoder = {v: k for k, v in self.vocab.items()}

        num_merges = vocab_size - len(self.vocab)
        pre_tokens = get_pre_tokens(input_path, self.special_tokens, self.PAT)
        words = [([self.vocab[token] for token in word], cnt) for word, cnt in pre_tokens.items()]

        pairs = defaultdict(int)
        ids = defaultdict(list)

        for idx, (word, cnt) in enumerate(words):
            for a, b in zip(word, word[1:]):
                pair = (a, b)

                self._bytes_cache[a] = (self.decoder[a],)
                self._bytes_cache[b] = (self.decoder[b],)
                pairs[pair] += cnt
                ids[pair].append(idx)

        print(f"Debug       len(pairs): {len(pairs)}  -  len(words): {len(words)}  -  len(ids): {len(ids)}")
        print()
        exit(0)

        print("Start merging")
        start_time = time.time()
        for iMerge in range(num_merges):

            if iMerge % 100 == 0 and iMerge != 0:
                tot_time = time.time() - start_time
                print(f"Merges: {iMerge}/{num_merges}, time: {tot_time:0.2f} sec, time to end: {(num_merges / iMerge) * tot_time:0.2f} sec")
                start_time = time.time()

            if not pair:
                break

            pair, pcnt = max(pairs.items(), key=lambda kv: (kv[1], (self._bytes_cache[kv[0][0]], self._bytes_cache[kv[0][1]])))
            
            print(f"Merge: {iMerge}  -  {pair}")

            token = len(self.vocab)
            self.vocab[token] = pair[0] + pair[1]
            self.decoder[pair[0] + pair[1]] = token
            self.merges.append((pair[0], pair[1]))

            self._bytes_cache[pair[0] + pair[1]] = tuple(self.decoder[ch] for ch in pair[0] + pair[1])

            for idx in list(ids[pair]):
                word, cnt = words[idx]

                new_word = []

                i = 0
                while i < len(word):
                    if i + 1 < len(word) and word[i] == pair[0] and word[i + 1] == pair[1]:
                        left = (word[i - 1], word[i]) if i - 1 >= 0 else None
                        right = (word[i + 1], word[i + 2]) if i + 2 < len(word) else None
                        
                        if left in pairs:
                            pairs[left] -= cnt

                            if pairs[left] <= 0:
                                del pairs[left]
                            
                            if idx in ids[left]:
                                ids[left].remove(idx)

                        if right in pairs:
                            pairs[right] -= cnt

                            if pairs[right] <= 0:
                                del pairs[right]
                            
                            if idx in ids[right]:
                                ids[right].remove(idx)

                        
                        new_word.append(self.vocab[token])

                        if i - 1 >= 0:
                            pairs[(word[i - 1], self.vocab[token])] += cnt
                            ids[(word[i - 1], self.vocab[token])].append(idx)

                        if i + 2 < len(word):
                            pairs[(self.vocab[token], word[i + 2])] += cnt
                            ids[(self.vocab[token], word[i + 2])].append(idx)

                        i += 2
                    else:
                        new_word.append(word[i])
                        i += 1
                
                words[idx] = (new_word, cnt)


            del pairs[pair]
            del ids[pair]


        final_vocab = {
            k: bytes([self.decoder[ch] for ch in v])
            for k, v in self.vocab.items()
        }

        self.save()

        return (final_vocab, [
            (
                bytes([self.decoder[token] for token in merge_token_1]),
                bytes([self.decoder[token] for token in merge_token_2])
                
            )
            for merge_token_1, merge_token_2 in self.merges
        ])

    def save(self):
        with open(f'{self.folder}/bpe_vocab.json', 'w', encoding='utf-8') as f:
            json.dump(self.decoder, f)

        with open(f'{self.folder}/bpe_merges.json', 'w', encoding='utf-8') as f:
            json.dump(self.merges, f)

    def load(self):
        with open(f'{self.folder}/bpe_vocab.json', 'r', encoding='utf-8') as f:
            self.decoder = json.load(f)

            self.vocab = {
                v: bytes([self.decoder[ch] for ch in k])
                for k,v in self.decoder.items()
            }

        with open(f'{self.folder}/bpe_merges.json', 'r', encoding='utf-8') as f:
            data = json.load(f)

            self.merges = [
                (
                    bytes([self.decoder[token] for token in merge_token_1]),
                    bytes([self.decoder[token] for token in merge_token_2])
                )
                for merge_token_1, merge_token_2 in data
            ]
        

if __name__ == "__main__":
    import cProfile, pstats
    
    pr = cProfile.Profile()
    pr.enable()
    
    bpe = BPETokenizer()
    vocab, merges = bpe.train("../.data/TinyStoriesV2-GPT4-train.txt", 10000, ["<|endoftext|>"]) # time to execute: 279.89 secs
    #vocab, merges = bpe.train("../.data/owt_valid.txt", 10000, ["<|endoftext|>"]) # 
    #vocab, merges = bpe.train("../tests/fixtures/corpus.en", 500, ["<|endoftext|>"])

    pr.disable()
    stats_path = "tokenizer_run.prof"
    print(f"Profile saved to {stats_path}")

    ps = pstats.Stats(pr).sort_stats("cumtime")
    ps.print_stats(30)