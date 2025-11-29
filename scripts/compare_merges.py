from pathlib import Path
import json
import sys
from pathlib import Path

# Ensure repo root is on sys.path so we can import the local packages
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from tests.common import gpt2_bytes_to_unicode, FIXTURES_PATH
from cs336_basics.tokenizer import BPETokenizer

reference_merges_path = FIXTURES_PATH / "train-bpe-reference-merges.txt"

bpe = BPETokenizer()
vocab, merges = bpe.train(str(FIXTURES_PATH / "corpus.en"), 500, ["<|endoftext|>"])

# load reference
byte_to_unicode = gpt2_bytes_to_unicode()
# gpt2_byte_decoder: unicode->int
gpt2_byte_decoder = {v: k for k, v in byte_to_unicode.items()}

def parse_reference():
    with open(reference_merges_path, encoding='utf-8') as f:
        lines = [tuple(line.rstrip().split(' ')) for line in f]
    ref = [(
        bytes([gpt2_byte_decoder[token] for token in m1]),
        bytes([gpt2_byte_decoder[token] for token in m2])
    ) for m1, m2 in lines]
    return ref

ref = parse_reference()

# find first difference
for i, (a, b) in enumerate(zip(merges, ref)):
    if a != b:
        print('First diff at index', i)
        print('yours:', a)
        print('ref :', b)
        break
else:
    print('No differences in common prefix; lengths', len(merges), len(ref))

# Print neighbors to inspect
idx = max(0, i-3)
print('\nContext around diff:')
for j in range(idx, idx+10):
    print(j, 'y:', merges[j])
    print(j, 'r:', ref[j])

print('\nTotal merges equal prefix length:', i)
