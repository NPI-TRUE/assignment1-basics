from typing import Iterable, Iterator, BinaryIO
import json
import regex as re
import os

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


def gpt2_bytes_to_unicode():
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

class Tokenizer:
    def __init__(self, vocab: dict[int, bytes] | None = None, merges: list[tuple[bytes, bytes]] | None = None, special_tokens: list[str] | None = None):
        self.vocab = vocab
        self.byte_encoder = gpt2_bytes_to_unicode()
        self.encoder = {v:k for k,v in self.vocab.items()} if self.vocab else None
        self.merges = merges
        self.special_tokens = special_tokens
        self.PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    def from_files(self, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None):
        with open(vocab_filepath, 'r', encoding='utf-8') as f:
            self.encoder = json.load(f)

            self.vocab = {
                v: bytes([self.encoder[ch] for ch in k])
                for k,v in self.encoder.items()
            }

        self.byte_encoder = gpt2_bytes_to_unicode()

        with open(merges_filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

            self.merges = [
                (
                    bytes([self.encoder[token] for token in merge_token_1]),
                    bytes([self.encoder[token] for token in merge_token_2])
                )
                for merge_token_1, merge_token_2 in data
            ]
            
        self.special_tokens = special_tokens

    def encode(self, text: str) -> list[int]:
        pre_tokens = []

        if self.special_tokens:
            for t in re.split('|'.join(self.special_tokens), text):
                for match in re.finditer(self.PAT, t):
                    word = match.group()
                    pre_tokens.append([self.byte_encoder[ch] for ch in word.encode('utf-8')])
        else:
            for match in re.finditer(self.PAT, text):
                word = match.group()
                pre_tokens.append([self.byte_encoder[ch].encode('utf-8') for ch in word.encode('utf-8')])

        #print(f"text: {text}  -  pre_tokens: {pre_tokens}")
        print(pre_tokens)

        encoded = []

        for ptoken in pre_tokens:

            modified = True

            while modified:
                token = []
                modified = False

                idx = 0

                while idx < len(ptoken):

                    if idx + 1 < len(ptoken) and (ptoken[idx], ptoken[idx + 1]) in self.merges:
                        modified = True
                        token.append(ptoken[idx] + ptoken[idx + 1])
                        idx += 2
                    else:
                        token.append(ptoken[idx])
                        idx += 1
                
                ptoken = token

            encoded.extend([self.encoder[ch] for ch in ptoken])

        return encoded

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        boundaries = find_chunk_boundaries(file=iterable, desired_num_chunks=10, split_special_token=b"<|endoftext|>")

        for start, end in zip(boundaries[:-1], boundaries[1:]):
            iterable.seek(start)
            text = iterable.read(end - start).decode("utf-8", errors="ignore")

            yield self.encode(text)

    def decode(self, ids: list[int]) -> str:
        return b''.join([self.vocab[ch] for ch in ids]).decode('utf-8', errors='replace')

if __name__ == "__main__":
    bpe = Tokenizer()
    bpe.from_files("tok_file/bpe_vocab_owt.json", "tok_file/bpe_merges_owt.json", ["<|endoftext|>"])

    print("-" + bpe.decode(bpe.encode("🙃")) + "-")

   
    