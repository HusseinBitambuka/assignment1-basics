from collections import defaultdict
import regex as re

class BPE:
    def __init__(self, corpus: str, vocab_size: int, special_tokens: list[str]) -> None:
        """
        Initialize BPE tokenizer with corpus, vocab size, and list of special tokens.
        """
        self.corpus = corpus
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens

        # GPT-style regex pattern
        self.PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

        # Initial byte-level vocab: {0..255}
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        self.merge_sets: dict[int, tuple[int, int]] = {}

        # Special token registration
        self.special_token_to_id: dict[str, int] = {}
        for i, token in enumerate(special_tokens):
            token_id = 256 + i
            self.vocab[token_id] = token.encode("utf-8")
            self.special_token_to_id[token] = token_id

        # Next token ID (after special tokens)
        self.next_token_id = 256 + len(special_tokens)

        # Build pre-token frequency table
        self.pretoken_table_count: dict[tuple[int, ...], int] = self.get_pre_token_freq_table()

    def get_pretoken(self) -> list[str]:
        return re.findall(self.PAT, self.corpus)

    def get_pre_token_freq_table(self) -> dict[tuple[int, ...], int]:
        """
        Converts pre-tokens into tuples of bytes and counts their frequency.
        """
        freq_pre_tokens = defaultdict(int)
        for pre_token in self.get_pretoken():
            pretoken_stream = tuple(pre_token.encode("utf-8"))
            freq_pre_tokens[pretoken_stream] += 1
        return freq_pre_tokens

    def get_token_freq_pairs(self) -> tuple[dict[tuple[int, int], int], dict[tuple[int, int], list[tuple[int, ...]]]]:
        """
        Count how often each byte pair appears and record where it appeared.
        """
        token_pairs = defaultdict(int)
        index_table = defaultdict(list)

        for pretoken, freq in self.pretoken_table_count.items():
            for token1, token2 in zip(pretoken, pretoken[1:]):
                token_pairs[(token1, token2)] += freq
                index_table[(token1, token2)].append(pretoken)

        return token_pairs, index_table

    def update_pre_token_table(self, most_frequent: tuple[int, int], pretokens_appeared: list[tuple[int, ...]]) -> None:
        """
        Merge the most frequent token pair in all affected pre-tokens.
        """
        token1, token2 = most_frequent
        new_token_id = self.next_token_id
        self.next_token_id += 1

        new_token_bytes = self.vocab[token1] + self.vocab[token2]
        self.vocab[new_token_id] = new_token_bytes
        self.merge_sets[new_token_id] = (token1, token2)

        for pretoken in pretokens_appeared:
            freq = self.pretoken_table_count[pretoken]
            new_pre_token = []
            i = 0
            while i < len(pretoken):
                if i < len(pretoken) - 1 and pretoken[i] == token1 and pretoken[i + 1] == token2:
                    new_pre_token.append(new_token_id)
                    i += 2
                else:
                    new_pre_token.append(pretoken[i])
                    i += 1
            del self.pretoken_table_count[pretoken]
            self.pretoken_table_count[tuple(new_pre_token)] += freq

    def merge(self) -> None:
        """
        Iteratively perform merges until reaching the target vocab size.
        """
        while len(self.vocab) < self.vocab_size:
            pair_freq, pair_index = self.get_token_freq_pairs()
            if not pair_freq:
                break
            most_frequent = max(pair_freq.items(), key=lambda x: x[1])[0]
            pretokens_appeared = pair_index[most_frequent]
            self.update_pre_token_table(most_frequent, pretokens_appeared)

    def encode(self, text: str) -> list[int]:
        """
        Tokenizes new input text using the trained merge rules.
        """
        pre_tokens:list[str] = re.findall(self.PAT, text)
        encoded_tokens:list[int] = []

        for token in pre_tokens:
            if token in self.special_token_to_id:
                encoded_tokens.append(self.special_token_to_id[token])
            else:
                byte_sequence = list(token.encode("utf-8"))
                byte_sequence = self.apply_merges(byte_sequence)
                encoded_tokens.extend(byte_sequence)

        return encoded_tokens

    def apply_merges(self, byte_sequence: list[int]) -> list[int]:
        """
        Applies learned BPE merges to a sequence of byte-level token IDs.
        """
        merges = sorted(self.merge_sets.items())  # Sort by token_id

        for token_id, (a, b) in merges:
            i = 0
            new_sequence = []
            while i < len(byte_sequence):
                if i < len(byte_sequence) - 1 and byte_sequence[i] == a and byte_sequence[i + 1] == b:
                    new_sequence.append(token_id)
                    i += 2
                else:
                    new_sequence.append(byte_sequence[i])
                    i += 1
            byte_sequence = new_sequence

        return byte_sequence

    def decode(self, tokens: list[int]) -> str:
        """
        Converts token IDs back into a string using the vocab mapping.
        """
        byte_stream = b''.join(self.vocab.get(token, b'') for token in tokens)
        return byte_stream.decode("utf-8", errors="replace")

    def print_vocab(self):
        """
        Print all tokens in the vocab.
        """
        print(" Vocabulary:")
        for tid in sorted(self.vocab.keys()):
            print(f"{tid}: {self.vocab[tid]}")

    def print_merges(self):
        """
        Print all merge rules.
        """
        print(" Merge rules:")
        for tid, pair in sorted(self.merge_sets.items()):
            print(f"{tid}: {pair}")
