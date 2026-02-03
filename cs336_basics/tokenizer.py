from typing import Iterable, Iterator, BinaryIO
import json
import regex as re
import os
import itertools

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

def bytes_to_unicode():
    bs = list(range(ord(" "), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
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

def safe_flatten(data): 
    if data and isinstance(data[0], list):
        return list(itertools.chain.from_iterable(data))
        
    return data

class Tokenizer():
    def __init__(self, 
                 vocab: dict[int, bytes] | None = None, 
                 merges: list[tuple[bytes, bytes]] | None = None, 
                 special_tokens: list[str] | None = None
                 ):
        self.vocab = vocab
        self.encoder = {v:k for k, v in self.vocab.items()} if self.vocab else {}

        if special_tokens is not None:
            max_index = max([k for k, v in vocab.items()])

            for stoken in special_tokens:
                stoken = stoken.encode()

                if stoken not in self.encoder:
                    max_index += 1
                    self.encoder[stoken] = max_index
                    
                self.vocab[max_index] = stoken

            self.special_tokens = sorted(special_tokens, key=lambda x: len(x), reverse=True) 

        else:
            self.special_tokens = None
        self.merges = merges
        self.PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    
    def from_files(self, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None):
        utb = {v:k for k, v in bytes_to_unicode().items()}

        with open(vocab_filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        max_index = max([v for k, v in data.items()])

        self.encoder = {bytes([utb[c] for c in k]): v for k, v in data.items()}

        if special_tokens is not None:
            special_tokens = sorted(special_tokens, key=lambda x: len(x), reverse=True)
            for stoken in  special_tokens:
                max_index += 1
                self.encoder[stoken.encode()] = max_index

        self.vocab = {v: k for k, v in self.encoder.items()}

        with open(merges_filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.merges = [
            (
                bytes([utb[token] for token in merge_token_1]),
                bytes([utb[token] for token in merge_token_2])
            )
            for merge_token_1, merge_token_2 in data
        ]
            
        self.special_tokens = special_tokens

    def encode(self, text: str) -> list[int]:
        pre_tokens = []

        if self.special_tokens:
            for t in re.split(r'(' + '|'.join(re.escape(st) for st in self.special_tokens) + ')', text):
                if t in self.special_tokens:
                    pre_tokens.append(t)
                    continue
                
                for match in re.finditer(self.PAT, t):
                    word = match.group()
                    pre_tokens.append([bytes([c]) for ch in word for c in ch.encode('utf-8')])
                
        else:
            for match in re.finditer(self.PAT, text):
                word = match.group()
                pre_tokens.append([bytes([c]) for ch in word for c in ch.encode('utf-8')])



        pairs = {}

        for idx, ptoken in enumerate(pre_tokens):
            if isinstance(ptoken, str):
                continue
            
            for a, b in zip(ptoken, ptoken[1:]):
                tpair = (a, b)

                if tpair not in pairs:
                    pairs[tpair] = set()

                pairs[tpair].add(idx)

        for merge in self.merges:
            if merge in pairs:
                for idx in pairs[merge]:
                    ptoken = pre_tokens[idx]
                    
                    new_ptoken = []

                    i = 0
                    while i < len(ptoken):
                        if i + 1 < len(ptoken) and ptoken[i] == merge[0] and ptoken[i + 1] == merge[1]:
                            new_ptoken.append(ptoken[i] + ptoken[i + 1])
                            i += 2
                        else:
                            new_ptoken.append(ptoken[i])
                            i += 1

                    pre_tokens[idx] = new_ptoken

                    for a, b in zip(new_ptoken, new_ptoken[1:]):
                        tpair = (a, b)

                        if tpair not in pairs:
                            pairs[tpair] = set()

                        pairs[tpair].add(idx)


        encoded = []

        for ptoken in pre_tokens:
            if isinstance(ptoken, str):
                encoded.extend([self.encoder[ptoken.encode('utf-8')]])
            else:
                encoded.extend([self.encoder[ch] for ch in ptoken])
        
        return encoded

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        binary_file = getattr(iterable, "buffer", iterable)
        boundaries = find_chunk_boundaries(file=binary_file, 
                                           desired_num_chunks=10, 
                                           split_special_token=b"<|endoftext|>")

        for start, end in zip(boundaries[:-1], boundaries[1:]):
            binary_file.seek(start)
            text = binary_file.read(end - start).decode("utf-8", errors="ignore")

            yield self.encode(text)

    def decode(self, ids: list[int]) -> str:
        ids = safe_flatten(ids)
        return b''.join([self.vocab[ch] for ch in ids]).decode('utf-8', errors='replace')
    
if __name__ == "__main__":
    bpe = Tokenizer()
    bpe.from_files("tok_file/bpe_vocab_owt.json", "tok_file/bpe_merges_owt.json", ["<|endoftext|>", "<|endoftext|><|endoftext|>"])

    #print(bpe.decode(bpe.encode("🙃")))
    #print(bpe.decode(bpe.encode("Hello world!<|endoftext|>")))

    ids = bpe.encode("Héllò hôw <|endoftext|><|endoftext|> are ü? 🙃<|endoftext|>")

    print(bpe.decode(ids))

    for idx in ids:
        print(idx, bpe.vocab[idx], bpe.decode([idx]))