import numpy as np
from typing import List, Dict

class SimpleTokenizer:
    """
    A word-level tokenizer with special tokens.
    """
    
    def __init__(self):
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self.vocab_size = 0
        
        # Special tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
    
    def build_vocab(self, texts: List[str]) -> None:
        """
        Build vocabulary from a list of texts.
        Add special tokens first, then unique words.
        """
        self.vocab={}
        self.all_word=set()
        for text in texts:
            for word in text.lower().split():
                self.all_word.add(word)
        self.vocab[self.pad_token]=0
        self.vocab[self.unk_token]=1
        self.vocab[self.bos_token]=2
        self.vocab[self.eos_token]=3

        self.all_word_s=sorted(self.all_word)
        
        unique_words=self.all_word_s
        # 1. Assign IDs to special tokens
        special_tokens = [self.pad_token, self.unk_token, self.bos_token, self.eos_token]
        for idx, token in enumerate(special_tokens):
            self.word_to_id[token] = idx
            
        # 2. Assign IDs to unique words (starting from index 4)
        for idx, word in enumerate(sorted(unique_words)):
            self.word_to_id[word] = idx + len(special_tokens)

        # 3. Create the reverse mapping for decoding
        self.id_to_word = {v: k for k, v in self.word_to_id.items()}
        self.vocab_size = len(self.word_to_id)
        
    def encode(self, text: str) -> List[int]:
        """
        Convert text to list of token IDs.
        Use UNK for unknown words.
        """
        self.encode_ids=[]
        for word in text.lower().split():
            if word not in self.word_to_id:
                self.encode_ids.append(1)
            else:
                id=self.word_to_id[word]
                self.encode_ids.append(id)
        print(self.vocab)
        return self.encode_ids
    
    def decode(self, ids: List[int]) -> str:
        """
        Convert list of token IDs back to text.
        """
        self.decode_text=[]
        for id in ids:
            if id not in self.id_to_word:
                self.decode_text.append(self.unk_token)
            else:
                self.decode_text.append(self.id_to_word[id])
        return " ".join(self.decode_text)
        
