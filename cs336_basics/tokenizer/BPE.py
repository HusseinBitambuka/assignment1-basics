from collections import defaultdict
import regex as re
import pickle

class BPE:
    def __init__(self, vocab_size: int, special_tokens: list[str]) -> None:
        """
        Initialize BPE tokenizer with target vocab size and special tokens.
        """
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens

        # GPT-2 style pretokenization regex
        self.PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

        # Byte-level vocab: start with {0..255}
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        self.merge_sets: dict[int, tuple[int, int]] = {}

        # Special tokens
        self.special_token_to_id: dict[str, int] = {}
        for i, token in enumerate(special_tokens):
            token_id = 256 + i
            self.vocab[token_id] = token.encode("utf-8")
            self.special_token_to_id[token] = token_id

        # Track next available token ID
        self.next_token_id = 256 + len(special_tokens)

        # Frequency of each byte-sequence pretoken (as tuples of ints)
        self.pretoken_table_count: dict[tuple[int, ...], int] = defaultdict(int)

    def add_chunk_to_pre_token_table(self, text_chunk: str) -> None:
        """
        Accepts a chunk of raw text and updates the pre-token frequency table.
        """
        for pre_token in re.findall(self.PAT, text_chunk):
            if pre_token in self.special_token_to_id:
                continue
            token_bytes = tuple(pre_token.encode("utf-8"))
            self.pretoken_table_count[token_bytes] += 1

    def get_token_freq_pairs(self) -> tuple[dict[tuple[int, int], int], dict[tuple[int, int], list[tuple[int, ...]]]]:
        """
        Compute frequency of each adjacent byte pair across all pre-token sequences.
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
        Applies the most frequent merge rule to all matching pre-token sequences.
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
        Train the BPE tokenizer by merging the most frequent token pairs.
        """
        while len(self.vocab) < self.vocab_size:
            pair_freq, pair_index = self.get_token_freq_pairs()
            if not pair_freq:
                break
            most_frequent = max(pair_freq.items(), key=lambda x: x[1])[0]
            pretokens_appeared = pair_index[most_frequent]
            self.update_pre_token_table(most_frequent, pretokens_appeared)

    def apply_merges(self, byte_sequence: list[int]) -> list[int]:
        """
        Applies learned merges to a sequence of byte-level token IDs.
        """
        merges = sorted(self.merge_sets.items())  # Sort by merge order (token ID)

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

    def encode(self, text: str) -> list[int]:
        """
        Encode a new text string using the trained tokenizer.
        """
        pre_tokens = re.findall(self.PAT, text)
        encoded_tokens = []

        for token in pre_tokens:
            if token in self.special_token_to_id:
                encoded_tokens.append(self.special_token_to_id[token])
            else:
                byte_sequence = list(token.encode("utf-8"))
                byte_sequence = self.apply_merges(byte_sequence)
                encoded_tokens.extend(byte_sequence)

        return encoded_tokens

    def decode(self, tokens: list[int]) -> str:
        """
        Decode a sequence of token IDs into a UTF-8 string.
        """
        byte_stream = b''.join(self.vocab.get(token, b'') for token in tokens)
        return byte_stream.decode("utf-8", errors="replace")

    def print_vocab(self):
        """
        Print the full vocabulary.
        """
        print("Vocabulary:")
        for tid in sorted(self.vocab.keys()):
            print(f"{tid}: {self.vocab[tid]}")

    def print_merges(self):
        """
        Print all applied BPE merges.
        """
        print("Merge rules:")
        for tid, pair in sorted(self.merge_sets.items()):
            print(f"{tid}: {pair}")

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump({
                "vocab_size": self.vocab_size,
                "special_tokens": self.special_tokens,
                "vocab": self.vocab,
                "merge_sets": self.merge_sets,
                "special_token_to_id": self.special_token_to_id,
            }, f)

    @classmethod
    def load(cls, path: str) -> "BPE":
        with open(path, "rb") as f:
            state = pickle.load(f)
        
        bpe = cls(
            vocab_size=state["vocab_size"],
            special_tokens=state["special_tokens"]
        )
        bpe.vocab = state["vocab"]
        bpe.merge_sets = state["merge_sets"]
        bpe.special_token_to_id = state["special_token_to_id"]
        bpe.next_token_id = max(bpe.vocab) + 1
        return bpe
