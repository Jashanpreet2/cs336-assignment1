from collections.abc import Iterable
import ast
from .common import PAT
import regex as re
import sys

debug = True
def printdebug(*args):
    global debug
    if debug:
        print(args)
    
class Tokenizer():
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens=None):
        if special_tokens ==  None:
            special_tokens = []
        for st in special_tokens:
            if st not in vocab:
                vocab[len(vocab)] = st.encode("utf-8")
        
        self.itot = vocab
        self.ttoi = {vocab[id]: id for id in vocab}
        self.merges = merges
        self.mergeIndices = {merge: i for i, merge in enumerate(merges)}
        self.special_tokens = [st.encode("utf-8") for st in special_tokens]
    
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=[]):
        with open(vocab_filepath, "r") as f:
            vocab = f.read()
        with open(merges_filepath, "r") as f:
            merges = f.read()
        printdebug(vocab)
        raise NotImplementedError("its over")
        return
        # return Tokenizer(astvocab, merges, special_tokens)
                
        
    def encode(self, text) -> list[int]:
        assert b"Hello" in self.ttoi
        def get_earliest_pair(word):
            earliestPair = None
            for i in range(1, len(word)):
                pair = (word[i-1], word[i])
                if pair not in self.merges:
                    continue
                if earliestPair is None or self.mergeIndices[pair] < self.mergeIndices[earliestPair]:
                    earliestPair = pair
                    continue
            return earliestPair

        global PAT
        
        special = "|".join(
        re.escape(st.decode("utf-8"))
            for st in sorted(self.special_tokens, key=len, reverse=True)
        )
        
        mypat = special + "|" + PAT.replace(
            r" ?[^\s\p{L}\p{N}]+",
            r" ?(?:(?!" + special + r")[^\s\p{L}\p{N}])+",
        )
        
        print(f"pat: {mypat}")
        text = text.encode("utf-8")
        print(text)
        # Pre-tokenization
        words = list()
        for word in re.finditer(mypat, text):
            pre_token = word.group()
            print(f"word: {pre_token}")
            if pre_token in self.special_tokens:
                words.append(tuple([pre_token]))
                print(f"appended st: {pre_token}")
            else:
                words.append(tuple(i.to_bytes() for i in pre_token))
        for word_index, word in enumerate(words):
            while True:
                earliestPair = get_earliest_pair(word)
        
                # No pairs to merge
                if earliestPair is None:
                    printdebug("no pair quitting")
                    break
                
                newToken = []
                i = 0
                while i < len(word):
                    if i == len(word)-1:
                        newToken.append(word[i])
                        break
                    oldToken = newToken
                    curPair = (word[i], word[i+1])
                    if curPair == earliestPair:
                        newToken.append(curPair[0] + curPair[1])
                        i += 1
                    else:
                        newToken.append(curPair[0])
                    i += 1
                newToken = tuple(newToken)
                words[word_index] = newToken
                word = newToken
                        
        res = []
        for word in words:
            for token in word:
                if token in self.special_tokens:
                    print(f"found special token: f{token}")
                res.append(self.ttoi[token])
            # Found the earliest pair, now replace
        
        return res
        

    def encode_iterable(self, iterable: Iterable[str]) -> Iterable[int]:
        return list()
    
    def decode(self, ids: list[int]) -> str:
        res = b"".join([self.itot[i] for i in ids]).decode("utf-8", errors='replace')
        # printdebug(f"Decoded {ids} to {res}")
        return res