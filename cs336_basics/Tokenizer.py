from BPE import BPE
import time

def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str]) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    start_time: float = time.time()

    try:
        with open(input_path, "r", encoding="utf-8") as f:
            corpus = f.read()
    except FileNotFoundError:
        raise RuntimeError(f"Input file not found: {input_path}")

    bpe = BPE(corpus=corpus, vocab_size=vocab_size, special_tokens=special_tokens)  # type: ignore
    bpe.merge()

    end_time: float = time.time()
    print(f"Training time: {(end_time - start_time) / 60:.2f} minutes")

    merges: list[tuple[bytes, bytes]] = [
        (bpe.vocab[a], bpe.vocab[b]) for _, (a, b) in sorted(bpe.merge_sets.items())
    ]

    return bpe.vocab, merges


if __name__ == "__main__":
    file_path = "/home/husseinbitambuka/Dev/assignment1-basics/data/TinyStoriesV2-GPT4-train.txt"
    vocab_size = 10000
    special_tokens = ["<|endoftext|>"]

    vocab, merges = train_bpe(file_path, vocab_size, special_tokens)
