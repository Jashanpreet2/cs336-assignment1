import os
from typing import BinaryIO
import multiprocessing
import regex as re
import sys
import time

SPECIAL_TOKENS = ["<|endoftext|>"]


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))
    

def read_file(filename, start_boundary, end_boundary, count_dict, lk):
    PAT = "|".join(re.escape(t) for t in sorted(SPECIAL_TOKENS, key=len, reverse=True)) + r"""|'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    my_count_dict = {}
    with open(filename, "rb") as f:
        f.seek(start_boundary)
        raw_content = f.read(end_boundary-start_boundary).decode("utf-8", errors="ignore")

    for token in re.finditer(PAT, raw_content):
        byte_val = token.group().encode("utf-8")
        if byte_val not in my_count_dict:
            my_count_dict[byte_val] = 0
        my_count_dict[byte_val] += 1
    with lk:
        for token in my_count_dict:
            if token not in count_dict:
                count_dict[token] = 0
            count_dict[token] += my_count_dict[token]


## Usage
filename = "data/TinyStoriesV2-GPT4-valid.txt"
with open(filename, "rb") as f:
    num_processes = 4
    boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
    count_dict = multiprocessing.Manager().dict()
    lk = multiprocessing.Lock()
    # The following is a serial implementation, but you can parallelize this
    # by sending each start/end pair to a set of processes.
    processes: list[multiprocessing.Process] = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        p = multiprocessing.Process(target=read_file, args=(filename, start, end, count_dict, lk))
        p.start()
        processes.append(p)
        # start subprocesses with their respective boundaries
        # subprocesses returns their counts
        # total the counts
        # find the most common pair
        #     create a new token for the pair
        #     send the pair to the subprocesses
        #     continue loop until max vocab length reached
    for p in processes:
        p.join()
    
    
    PAT = "|".join(re.escape(t) for t in sorted(SPECIAL_TOKENS, key=len, reverse=True)) + r"""|'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    f.seek(0)
    text = f.read().decode("utf-8", errors="ignore")
    counts = {}
    for t in re.finditer(PAT, text):
            byte_val = t.group().encode("utf-8")
            if byte_val not in counts:
                counts[byte_val] = 0
            counts[byte_val] += 1
    differents = []
    for key in counts:
        if counts[key] != count_dict[key]:
            print(key, counts[key], count_dict[key])
        assert counts[key] == count_dict[key]
    print("passed")
    print(sys.getsizeof(counts))
