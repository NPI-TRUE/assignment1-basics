from typing import Iterable, Iterator
import json
import regex as re

class Tokenizer:
    def __init__(self, vocab: dict[int, bytes] | None = None, merges: list[tuple[bytes, bytes]] | None = None, special_tokens: list[str] | None = None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens
        self.PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    def from_files(self, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None):
        with open(vocab_filepath, 'r', encoding='utf-8') as f:
            self.decoder = json.load(f)

            self.vocab = {
                v: bytes([self.decoder[ch] for ch in k])
                for k,v in self.decoder.items()
            }

        with open(merges_filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

            self.merges = [
                (
                    bytes([self.decoder[token] for token in merge_token_1]),
                    bytes([self.decoder[token] for token in merge_token_2])
                )
                for merge_token_1, merge_token_2 in data
            ]
            
        self.special_tokens = special_tokens

    def encode(self, text: str) -> list[int]:
        pre_tokens = []

        for t in re.split('|'.join(self.special_tokens), text):
            for match in re.finditer(self.PAT, t):
                word = match.group()
                pre_tokens.append([self.vocab[ch] for ch in word.encode('utf-8')])

        encoded = []

        for ptoken in pre_tokens:
            ptoken = list([(a, b) for a, b in zip(ptoken, ptoken[1:])])

            for merge in self.merges:
                try:
                    idx = ptoken.index(merge)
                    a, b = ptoken[idx]
                    pair = a + b

                    left = ptoken[idx - 1] if idx > 0 else None
                    right = ptoken[idx + 1] if idx < len(ptoken) - 1 else None

                    ptoken.pop(idx)
                    
                    if left:
                        ptoken.pop(idx - 1)
                        ptoken.insert(idx - 1, (left[0], pair))

                    if right: 
                        ptoken.pop(idx)
                        ptoken.insert(idx, (pair, right[1]))

                except ValueError:
                    continue

                
            encoded.append(ptoken)

        return encoded

                    
        


    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        pass

    def decode(self, ids: list[int]) -> str:
        pass

if __name__ == "__main__":
    bpe = Tokenizer()
    bpe.from_files("tok_file/bpe_vocab_owt.json", "tok_file/bpe_merges_owt.json", ["<|endoftext|>"])

    print(bpe.encode("Hello world!"))