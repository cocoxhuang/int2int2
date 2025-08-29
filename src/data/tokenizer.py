from typing import List, Dict, Tuple, Union
import json

class BaseTokenizer:
    """
    Tokenizer that converts sequences of integers into sign and base representation.
    
    Example:
    Input: [1, -100, 200]
    Base representation: ['BOS', '+', '1', '-', '1', '0', '0', '+', '2', '0', '0', 'EOS'] with a base of 10
    Tokens: [0, 2, 5, 3, 5, 4, 4, 2, 6, 4, 4, 1]  # using vocabulary mapping
    """
    
    def __init__(self, base: int = 10):
        """
        Initialize the tokenizer.
        
        Args:
            base: The base for number representation (default: 10)
            max_digits: Maximum number of digits to support (for vocabulary size estimation)
        """
        self.base = base
        
        # Create vocabulary
        self.vocab = self._create_vocabulary()
        self.vocab_size = len(self.vocab)
        
        # Create reverse mapping for decoding
        self.id_to_token = {v: k for k, v in self.vocab.items()}
        
        # Special tokens
        self.pad_token = '<PAD>'
        self.unk_token = '<UNK>'
        self.sep_token = '<SEP>'  # separator between numbers
        self.bos_token = '<BOS>'
        self.eos_token = '<EOS>'
        
    def _create_vocabulary(self):
        """Create vocabulary dictionary mapping tokens to IDs."""
        vocab = {}
        
        # Special tokens
        vocab['<PAD>'] = 0
        vocab['<UNK>'] = 1
        vocab['<SEP>'] = 2
        vocab['<BOS>'] = 3
        vocab['<EOS>'] = 4
        
        # Sign tokens
        vocab['+'] = 5
        vocab['-'] = 6
        
        # Digit tokens
        for i in range(self.base):
            vocab[str(i)] = 7 + i
            
        return vocab
    
    def encode_integer(self, value: int):
        """
        Convert a single integer to sign and base representation.
        Example:
            encode_integer(-123) -> ['-', '1', '2', '3']
            encode_integer(0) -> ['+', '0']
        """
        if value == 0:
            return ['+', '0']
        sign = '+' if value >= 0 else '-'
        abs_value = abs(value)
        # Convert to base representation
        digits = []
        while abs_value > 0:
            digits.append(str(abs_value % self.base))
            abs_value = abs_value // self.base
        # Reverse to get most significant digit first
        digits.reverse()
        return [sign] + digits
    
    def encode_sequence(self, integers, add_special_tokens = True):
        """Convert a sequence of integers to token representation."""
        tokens = []
        if add_special_tokens:  # first add bos
            tokens.append(self.bos_token)
        for i, integer in enumerate(integers):
            tokens.extend(self.encode_integer(integer))
        if add_special_tokens:  # end with eos
            tokens.append(self.eos_token)
        return tokens
    
    def decode_integer(self, tokens):
        """Convert sign and base representation back to integer."""
        if not tokens or len(tokens) < 2:
            raise ValueError("Invalid token sequence for decoding integer.")
        sign = tokens[0]
        digits = tokens[1:]
        # Convert from base representation
        value = 0
        for digit in digits:
            value = value * self.base + int(digit)                
        return -value if sign == '-' else value
    
    def decode_sequence(self, tokens):
        """Convert tokens back to sequence of integers."""
        # Remove special tokens
        filtered_tokens = [t for t in tokens if t not in [self.bos_token, self.eos_token, self.pad_token, self.sep_token, self.unk_token]]
        # decode
        integers = []
        current_tokens = []
        for token in filtered_tokens:
            integers.append(token)
        return integers
    
    def tokenize(self, integers, add_special_tokens = True):
        """Convert sequence of integers to token IDs."""
        string_tokens = self.encode_sequence(integers, add_special_tokens)
        return [self.vocab.get(token, self.vocab[self.unk_token]) for token in string_tokens]
    
    def detokenize(self, token_ids):
        """ Convert token IDs back to sequence of integers."""
        string_tokens = [self.id_to_token.get(token_id, self.unk_token) for token_id in token_ids]
        return self.decode_sequence(string_tokens)
    
    def _get_vocab(self):
        return self.vocab.copy()
    
    def _get_vocab_size(self):
        return self.vocab_size
    
    def _save_vocab(self, filepath):
        with open(filepath, 'w') as f:
            json.dump(self.vocab, f, indent=2)
    
    def _load_vocab(self, filepath: str):
        import json
        with open(filepath, 'r') as f:
            self.vocab = json.load(f)
        self.vocab_size = len(self.vocab)
        self.id_to_token = {v: k for k, v in self.vocab.items()}