"""
Demonstration of two BPE tie-breaking strategies:

- String lexicographic tie-break (what the original code used): compares display strings
  lexicographically (Unicode codepoints).
- Underlying-byte tie-break (reference behavior): compares the original byte integers
  that those display characters map to (the GPT-2 bytes->unicode inverse mapping).

Run: python3 scripts/bpe_tiebreak_example.py
"""

from collections import Counter


def bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    characters = [chr(n) for n in cs]
    return dict(zip(bs, characters))


# Build inverse mapping: displayed char -> original byte int
GPT2 = bytes_to_unicode()
GPT2_INV = {v: k for k, v in GPT2.items()}


def select_by_string_tiebreak(pairs_counter):
    """Select best pair using (count, (left_str, right_str)) as key."""
    return max(pairs_counter.items(), key=lambda kv: (kv[1], kv[0]))[0]


def select_by_byte_tiebreak(pairs_counter):
    """Select best pair using (count, (left_bytes_tuple, right_bytes_tuple)) as key.

    left_bytes_tuple is the sequence of original byte ints that composed the displayed
    token string (e.g., 'Ġc' -> (32, 99)).
    """

    def key(kv):
        pair, count = kv[0], kv[1]
        left_str, right_str = pair
        left_bytes = tuple(GPT2_INV[ch] for ch in left_str)
        right_bytes = tuple(GPT2_INV[ch] for ch in right_str)
        return (count, (left_bytes, right_bytes))

    return max(pairs_counter.items(), key=key)[0]


if __name__ == "__main__":
    # Construct two example pairs that have the same frequency
    # Pair A: left token is a display string starting with the special space char (Ġ)
    #          e.g., displayed token for byte 32 is GPT2[32] (Ġ), so token 'Ġc' stands for bytes (32, 99)
    # Pair B: left token is just 't' (byte 116)

    left_a = GPT2[32] + GPT2[99]   # 'Ġ' + 'c'  (display of bytes [32, 99])
    right_a = GPT2[111] + GPT2[109]  # 'o' + 'm' (display of bytes [111, 109])

    left_b = GPT2[116]  # 't'
    right_b = GPT2[104]  # 'h'

    pairs = Counter({(left_a, right_a): 5, (left_b, right_b): 5})

    print("Pairs (display):")
    for pair, cnt in pairs.items():
        print(f"  {pair} -> count={cnt}")

    pick_str = select_by_string_tiebreak(pairs)
    pick_byte = select_by_byte_tiebreak(pairs)

    print("\nSelected by string lexicographic tie-break:")
    print(f"  {pick_str}")

    print("\nSelected by underlying-byte tie-break:")
    print(f"  {pick_byte}")

    print("\nExplanation:")
    print(f"  Left token A first char codepoint: {ord(left_a[0])} (display '{left_a[0]}')")
    print(f"  Left token B first char codepoint: {ord(left_b[0])} (display '{left_b[0]}')")
    print(f"  Left token A first original byte: {GPT2_INV[left_a[0]]}")
    print(f"  Left token B first original byte: {GPT2_INV[left_b[0]]}")

    print('\nSo string ordering compares codepoints (Ġ > t), but underlying bytes compare (32 < 116).')
