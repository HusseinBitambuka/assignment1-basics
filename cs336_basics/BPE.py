from collections import defaultdict
class BPE:
    def __init__(self, tokens:str, vocab_size:int) -> None:

        self.tokens:str = tokens
        self.vocab_size:int = vocab_size
        self.vocab:dict[int, bytes] = {i:bytes([i]) for i in range(256)}
        self.merge_sets:dict[int, tuple] = {}
    
    def get_tokens(self) -> list[int]:

        return list(self.tokens.encode("utf-8"))
    
    def get_stats(self, tokens:list[int]) -> dict:

        freq_pair:dict[tuple, int] = defaultdict(int)
        
        for fisrt_word, second_word in zip (tokens, tokens[1:]):
            freq_pair[(fisrt_word, second_word)] += 1
        return freq_pair
    
    def get_most_frequent_pair(self, freq_pair:dict[tuple, int]) -> tuple:

        pair_result:tuple
        freq:int = 0
        for pair, count in freq_pair.items():
            if count > freq:
                pair_result = pair
                freq = count
        return pair_result

        
    def merge(self, tokens: list[int], index: int) -> list[int]:
        
        freq_pair: dict[tuple, int] = self.get_stats(tokens)

        if not freq_pair:
            raise ValueError("No frequent pairs found — cannot perform merge. The token list may be too short or already fully merged.")

        most_freq_pair = self.get_most_frequent_pair(freq_pair=freq_pair)
        a, b = most_freq_pair

        self.vocab[index] = self.vocab[a] + self.vocab[b]
        self.merge_sets[index] = (a, b)

        new_tokens = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and tokens[i] == a and tokens[i + 1] == b:
                new_tokens.append(index)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1

        return new_tokens

    def train(self) -> None:

        tokens:list[int] = self.get_tokens()
        num_merges:int = self.vocab_size - 256
        for i in range(num_merges):
            index:int = i + 256
            tokens = self.merge(tokens, index)
    
    def decode(self, tokens: list[int]) -> str:
        byte_sequence = b''.join(self.vocab[token] for token in tokens)
        return byte_sequence.decode("utf-8", errors="replace")

    def tokenize(self, text: str) -> list[int]:
        tokens = list(text.encode("utf-8"))

        for index in sorted(self.merge_sets.keys()):
            a, b = self.merge_sets[index]
            i = 0
            merged = []
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == a and tokens[i + 1] == b:
                    merged.append(index)
                    i += 2
                else:
                    merged.append(tokens[i])
                    i += 1
            tokens = merged

        return tokens
