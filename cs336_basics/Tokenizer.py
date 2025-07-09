from BPE import BPE
import time
from file_processing import find_chunk_boundaries

if __name__ == "__main__":
    file_path = "/home/husseinbitambuka/Dev/assignment1-basics/data/TinyStoriesV2-GPT4-train.txt"
    vocab_size = 10000
    special_tokens = ["<|endoftext|>"]
    split_token = "<|endoftext|>".encode("utf-8")
    num_chunks = 30

    start = time.time()

    # Step 1: Initialize tokenizer
    bpe = BPE(vocab_size=vocab_size, special_tokens=special_tokens)

    # Step 2: Open file and find chunk boundaries
    with open(file_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, desired_num_chunks=num_chunks, split_special_token=split_token)

        # Step 3: For each chunk, decode and feed to BPE
        for start_offset, end_offset in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start_offset)
            chunk_bytes = f.read(end_offset - start_offset)
            chunk_text = chunk_bytes.decode("utf-8", errors="ignore")
            bpe.add_chunk_to_pre_token_table(chunk_text)

    # Step 4: Train merges
    bpe.merge()

    # Step 5: Print stats
    end = time.time()
    elapsed = end - start

    print(f"\n Finished training BPE tokenizer")
    print(f" Time elapsed: {elapsed:.2f} seconds ({elapsed / 60:.2f} minutes)")
    print(f" Final vocab size: {len(bpe.vocab)}")
    print(f" Merge operations: {len(bpe.merge_sets)}")



 '''
 First output of the code:
 ---------------------------------

 Finished training BPE tokenizer
 Time elapsed: 1911.68 seconds (31.86 minutes)
 Final vocab size: 10000
 Merge operations: 9743
 
 
 info about the file:
 --------------------
  File: TinyStoriesV2-GPT4-train.txt
  Size: 2227753162 (2.7 GB)      Blocks: 4351096    IO Block: 4096   regular file
Device: 259,2   Inode: 12334704    Links: 1
Access: (0664/-rw-rw-r--)  Uid: ( 1000/husseinbitambuka)   Gid: ( 1000/husseinbitambuka)
Access: 2025-07-08 02:07:58.717304936 -0400
Modify: 2023-05-19 15:23:25.000000000 -0400
Change: 2025-07-03 17:49:24.997496925 -0400
 Birth: 2025-07-03 17:47:34.617640099 -0400
 
 '''

  


    
