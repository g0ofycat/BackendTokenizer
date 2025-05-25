import re

class Tokenizer:

    def __init__(self, vocab=None, lowercase=True, tokenizer_type='whitespace', merges=None):
        
        VocabFile = "C:/Users/clint/g0ofycat/programming/python/machine_learning/Concepts/BERTVocab.txt"
        with open(VocabFile, "r", encoding="utf-8") as f:
            tokens = f.read().splitlines()
            vocab = {token: idx for idx, token in enumerate(tokens)}

        if merges is None:
            merges_path = "C:/Users/clint/g0ofycat/programming/python/machine_learning/Concepts/merges.txt"
            with open(merges_path, "r", encoding="utf-8") as f:
                lines = f.read().splitlines()
                merges = [tuple(line.strip().split()) for line in lines if line and not line.startswith("#")]

        self.vocab = vocab or {}
        self.id_to_token = {idx: token for token, idx in self.vocab.items()}
        self.lowercase = lowercase
        self.tokenizer_type = str.lower(tokenizer_type)
        self.merges = merges or [] # Only for BPE
        self.bpe_seperator = "Ġ"

        validTypes = {"whitespace", "char", "bpe"}

        if tokenizer_type not in validTypes:
            return print("Invalid Tokenizer Type")
        
    def _merge_pair(self, tokens, pair): # Combines all adjacent occurrences of a specific pair of tokens into a single merged token

        merged_tokens = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i + 1] == pair[1]: # "If the current index of the token we're on is the same as the pair as well as the index after then append both"
                # Merge the pair into a single token
                merged_tokens.append(tokens[i] + tokens[i + 1])
                i += 2
            else:
                merged_tokens.append(tokens[i]) # Singular token, most likely punctuation
                i += 1
        return merged_tokens

    def _get_pairs(self, tokens): # Returns a set of valid pairs in the string

        pairs = set()
        for i in range(len(tokens) - 1):
            pairs.add((tokens[i], tokens[i + 1]))
        return pairs   
        
    def _split_on_punctuation(self, text):
        # Split on word boundaries, keeping punctuation separate
        tokens = re.findall(r'\w+|[^\w\s]', text)
        return tokens
    
    def ApplyBPE(self, text):
        words = self._split_on_punctuation(text)
        final_tokens = []

        for word in words:
            if self.lowercase:
                word = word.lower()

            if word.isalnum():  # Split word and add separator to end
                tokens = list(word) + [self.bpe_seperator]

                while True:
                    pairs = self._get_pairs(tokens)
                    candidate = None

                    for merge_pair in self.merges:
                        if merge_pair in pairs:
                            candidate = merge_pair
                            break  # Only merge the first valid pair in your merge priority

                    if candidate is None:
                        break

                    tokens = self._merge_pair(tokens, candidate)

            else:
                tokens = [word]  # Punctuation, 1 token

            final_tokens.extend(tokens)

        return final_tokens


    
    def tokenize(self, text): # Splits the String with BPE, Whitespace, etc.
        if self.tokenizer_type == "whitespace":
            return text.split(" ")
        elif self.tokenizer_type == "char":
            return list(text)
        elif self.tokenizer_type == "bpe":
            return self.ApplyBPE(text)
        
    
    def encode(self, text): # Encode the tokenized string
        TokenizedText = self.tokenize(text)
        tokens = []
        for char in TokenizedText:
            token_id = self.vocab.get(char, self.vocab.get("[UNK]"))
            tokens.append(token_id)
        return tokens
    
    def decode(self, token_ids):
        decodedtokens = []
        for token_id in token_ids:
            str_id = self.id_to_token.get(token_id, "[UNK]") # [UNK] is prob the seperator from BPE
            decodedtokens.append(str_id)
        return decodedtokens


NewTokenizer = Tokenizer(None, False, 'bpe', merges=None)

Encoded = NewTokenizer.encode("Supercalafragalisticexpealadocious")

print(Encoded)
