import os
from typing import BinaryIO
import multiprocessing as mp
from multiprocessing import Process, Queue, Pool
import regex as re
import sys
import time
import gc

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
    

def read_file(filename, start_boundary, end_boundary, special_tokens: list[str]):
    PAT = r"""|'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    
    my_count_dict = {}
    with open(filename, "rb") as f:
        f.seek(start_boundary)
        raw_content = f.read(end_boundary-start_boundary).decode("utf-8", errors="ignore")
    for chunk in re.split("|".join([re.escape(st) for st in special_tokens]), raw_content):
        for word in re.finditer(PAT, chunk):
            word = tuple(i.to_bytes() for i in word.group().encode("utf-8"))
            if word not in my_count_dict:
                my_count_dict[word] = 0
            my_count_dict[word] += 1
    return my_count_dict
    


## Usage
# filename = "data/TinyStoriesV2-GPT4-valid.txt"
def train_bpe(input_path: str | os.PathLike, vocab_size: int, special_tokens: list[str]):
    print(f"input path: {input_path}")
    once = True
    vocab = {i: i.to_bytes() for i in range(256)}
    for st in special_tokens:
        vocab[len(vocab)] = st.encode("utf-8")
    merges = []
    
    num_processes = 14
    
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
    # The following is a serial implementation, but you can parallelize this
    # by sending each start/end pair to a set of processes.
    
    counts = {}
    start_time = time.perf_counter()
    with Pool() as pool:
        results = []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            results.append(pool.apply_async(read_file, (input_path, start, end, special_tokens)))
        for r in results:
            process_count = r.get()
            for k in process_count:
                if k not in counts:
                    counts[k] = 0
                counts[k] += process_count[k]

    print("time to read by multiprocessing: ", time.perf_counter() - start_time)
    print(f"{sys.getsizeof(counts)}")
    
    pairCounts = {}
    pairToWords = {}
    
    for word in counts:
        for i in range(1, len(word)):
            pair = (word[i-1], word[i])
            if pair not in pairCounts:
                pairCounts[pair] = 0
                pairToWords[pair] = set()
            pairCounts[pair] += counts[word]
            pairToWords[pair].add(word)
                
    start_time = time.perf_counter()
    while len(vocab) < vocab_size:
        mostCommonPairs = []
        for pair in pairCounts:
            if len(mostCommonPairs) == 0 or \
                pairCounts[pair] == pairCounts[mostCommonPairs[0]]:
                mostCommonPairs.append(pair)
            elif pairCounts[pair] > pairCounts[mostCommonPairs[0]]:
                mostCommonPairs = [pair]
        if once:
            once = False
            print("most common pairs: ", mostCommonPairs)
            print(f"in freq: {pairCounts[mostCommonPairs[0]]}, ' t' freq: {pairCounts[tuple(i.to_bytes() for i in " t".encode("utf-8"))]}")
        if tuple(int(10).to_bytes() for _ in range(2)) in mostCommonPairs:
            print("most common: ", mostCommonPairs)
        pair = max(mostCommonPairs)
        if pair[0]  == b' g' and pair[1] == b'ive':
            print("specials: ", special_tokens)
        if pairCounts[pair] == 1:
            break
        for word in pairToWords[pair]:
            i = 1
            curWord = None
            while i < len(word):
                # print("doing word ", word)
                if word[i-1] != pair[0] or word[i] != pair[1]:
                    i += 1
                    continue
                if word not in counts:
                    print(f"{word} not in counts for pair: {pair}, all words: {pairToWords[pair]}")
                                    
                newWord = word[:i-1]+tuple([pair[0]+pair[1]])+word[i+1:]
                for j in range(1, len(word)):
                    if (word[j-1], word[j]) == pair:
                        continue
                    if (word[j-1], word[j]) not in pairToWords:
                        print(f"absent: {(word[j-1], word[j])}, cur pair: {pair}")
                    pairToWords[(word[j-1], word[j])].discard(word)
                    pairCounts[(word[j-1], word[j])] -= counts[word]
                counts[newWord] = counts[word]
                for j in range(1, len(newWord)):
                    curPair = (newWord[j-1], newWord[j])
                    if curPair == pair:
                        continue
                    if curPair not in pairCounts:
                        pairCounts[curPair] = 0
                        pairToWords[curPair] = set()
                    pairCounts[curPair] += counts[newWord]
                    pairToWords[curPair].add(newWord)
                word = newWord
                i += 1
        del pairCounts[pair]
        del pairToWords[pair]
        vocab[len(vocab)] = pair[0] + pair[1]
        merges.append(pair)
        # print("merge time: ", time.perf_counter()-start_time, " curren vocab len: ", len(vocab), "max vocab len: ", vocab_size)
    print("Time to tokenize: ", time.perf_counter() - start_time)
    return vocab, merges
                