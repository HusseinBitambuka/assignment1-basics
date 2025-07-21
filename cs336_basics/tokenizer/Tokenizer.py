import cProfile
import pstats
from BPE import BPE
from file_processing import find_chunk_boundaries
import time
from datetime import datetime
import os

def train_tokenizer():
    file_path = "/home/husseinbitambuka/Dev/assignment1-basics/data/TinyStoriesV2-GPT4-train.txt"
    vocab_size = 10000
    special_tokens = ["<|endoftext|>"]
    split_token = "<|endoftext|>".encode("utf-8")
    num_chunks = 8

    start = time.time()
    bpe = BPE(vocab_size=vocab_size, special_tokens=special_tokens)

    with open(file_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, desired_num_chunks=num_chunks, split_special_token=split_token)

        for start_offset, end_offset in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start_offset)
            chunk_bytes = f.read(end_offset - start_offset)
            chunk_text = chunk_bytes.decode("utf-8", errors="ignore")
            bpe.add_chunk_to_pre_token_table(chunk_text)

    bpe.merge()

    end = time.time()
    print(f"\nFinished training BPE tokenizer")
    print(f"Time elapsed: {end - start:.2f} seconds")
    print(f"Final vocab size: {len(bpe.vocab)}")
    print(f"Merge operations: {len(bpe.merge_sets)}")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_basename = os.path.basename(file_path).replace(".txt", "")
    tokenizer_name = f"tokenizer_{file_basename}_vs{vocab_size}_{timestamp}.pkl"

    bpe.save(tokenizer_name)
    print(f"Tokenizer saved to {tokenizer_name}")

if __name__ == "__main__":
    profile_path = "bpe_profile.prof"
    cProfile.run("train_tokenizer()", profile_path)

    print(f"\nProfile saved to {profile_path}")
    stats = pstats.Stats(profile_path)
    stats.sort_stats("cumulative").print_stats(20)  # Print top 20 time-consuming calls
