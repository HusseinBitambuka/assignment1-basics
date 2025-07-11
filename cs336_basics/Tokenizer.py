import time
import mmap
from concurrent.futures import ThreadPoolExecutor
from BPE import BPE
from file_processing import find_chunk_boundaries

def process_chunk(mm_obj: mmap.mmap, start: int, end: int) -> str:
    chunk_bytes = mm_obj[start:end]
    return chunk_bytes.decode("utf-8", errors="ignore")

if __name__ == "__main__":
    file_path = "/home/husseinbitambuka/Dev/assignment1-basics/data/TinyStoriesV2-GPT4-train.txt"
    vocab_size = 10000
    special_tokens = ["<|endoftext|>"]
    split_token = "<|endoftext|>".encode("utf-8")
    num_chunks = 30  # Tune for your system

    start = time.time()

    # Step 1: Initialize tokenizer
    bpe = BPE(vocab_size=vocab_size, special_tokens=special_tokens)

    # Step 2: Open file and memory-map it
    with open(file_path, "rb") as f:
        mmapped_file = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        boundaries = find_chunk_boundaries(mmapped_file, desired_num_chunks=num_chunks, split_special_token=split_token)

        # Step 3: Decode chunks in parallel
        with ThreadPoolExecutor(max_workers=num_chunks) as executor:
            futures = [
                executor.submit(process_chunk, mmapped_file, start, end)
                for start, end in zip(boundaries[:-1], boundaries[1:])
            ]
            for future in futures:
                chunk_text = future.result()
                bpe.add_chunk_to_pre_token_table(chunk_text)

        mmapped_file.close()

    # Step 4: Train merges
    bpe.merge()

    # Step 5: Timing and stats
    end = time.time()
    elapsed = end - start

    print(f"\n Finished training BPE tokenizer")
    print(f" Time elapsed: {elapsed:.2f} seconds ({elapsed / 60:.2f} minutes)")
    print(f" Final vocab size: {len(bpe.vocab)}")
    print(f" Merge operations: {len(bpe.merge_sets)}")


    '''
    Finished training BPE tokenizer
    Time elapsed: 1928.86 seconds (32.15 minutes)
    Final vocab size: 10000
    Merge operations: 9743
    '''
