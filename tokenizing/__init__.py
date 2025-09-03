import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from tokenizing.numeral_tokenizer import NumeralTokenizer
from multiprocessing import Pool

# class Tokenizer:
#     def __init__(self, encoder, decoder, vocab_size, name=None):
#         self.encode = encoder
#         self.decode = decoder
#         self.vocab_size = vocab_size
#         self.name = name

#     def tokenize(self, data_list):
#         """
#         Takes a list of prefix-target pairs, tokenizes and concatenates them
#         """
#         out = []
#         prefix_len = len(self.encode(data_list[0][0]))
#         target_len = len(self.encode(data_list[0][1]))
#         same_len = True

#         for prefix, target in data_list:
#             prefix = torch.tensor(self.encode(prefix))
#             target = torch.tensor(self.encode(target))
#             if not (len(prefix) == prefix_len and len(target) == target_len):
#                 same_len = False
#             seq = torch.concatenate([prefix, target], dim=-1).long()
#             out.append(seq)

#         # Check if all prefixes and all targets have the same length
#         if not same_len:
#             print('Not all prefixes or targets have the same length!!')
#         else:
#             print('Equal sequence lengths!')

#         return out, prefix_len, target_len

class Tokenizer:
    def __init__(self, encoder, decoder, vocab_size, name=None):
        self.encode = encoder
        self.decode = decoder
        self.vocab_size = vocab_size
        self.name = name

    def tokenize(self, data_list, use_multiprocessing=False, num_processes=None):
        """
        Takes a list of prefix-target pairs, tokenizes and concatenates them efficiently in a batch.
        
        Args:
            data_list (list): A list of (prefix, target) string pairs.
            use_multiprocessing (bool): If True, uses a process pool to parallelize encoding.
            num_processes (int): Number of processes for the pool. Defaults to CPU count.

        Returns:
            A tuple containing:
            - A single 2D torch.Tensor where each row is a tokenized sequence.
            - The length of the tokenized prefixes.
            - The length of the tokenized targets.
        """
        if not data_list:
            return torch.tensor([], dtype=torch.long), 0, 0

        # 1. Unzip the list of pairs into separate lists of prefixes and targets
        prefixes, targets = zip(*data_list)

        # 2. Encode all prefixes and targets in a batch
        # This can be parallelized for a significant speedup on large datasets
        if use_multiprocessing:
            with Pool(processes=num_processes) as pool:
                encoded_prefixes = pool.map(self.encode, prefixes)
                encoded_targets = pool.map(self.encode, targets)
        else:
            # List comprehensions are much faster than for-loops for this task
            encoded_prefixes = [self.encode(p) for p in tqdm(prefixes)]
            encoded_targets = [self.encode(t) for t in tqdm(targets)]

        # 3. Get sequence lengths from the first encoded pair
        prefix_len = len(encoded_prefixes[0])
        target_len = len(encoded_targets[0])

        # 4. Concatenate encoded lists and check for consistent lengths
        all_sequences = []
        same_len = True
        for p, t in zip(encoded_prefixes, encoded_targets):
            if same_len and (len(p) != prefix_len or len(t) != target_len):
                same_len = False
            all_sequences.append(p + t)
        
        if not same_len:
            print('Warning: Not all prefixes or targets have the same length!')
        else:
            print('Equal sequence lengths!')

        # 5. Convert the entire batch to a single tensor in one operation
        # This is massively more efficient than creating a tensor for each sequence
        out_tensor = torch.tensor(all_sequences, dtype=torch.long)

        return out_tensor, prefix_len, target_len


def get_tokenizer(args):
    if args.model == 'gpt':
        t = NumeralTokenizer(args.num_nodes)
        tokenizer = Tokenizer(encoder=t.encode, decoder=t.decode, vocab_size=args.num_nodes + 4, name='numeral')
    elif args.model.startswith('gpt2'):
        t = AutoTokenizer.from_pretrained('gpt2')
        tokenizer = Tokenizer(encoder=t.encode, decoder=t.decode, vocab_size=50257 , name='gpt2')
    elif args.model.startswith('pythia'):
        t = AutoTokenizer.from_pretrained('EleutherAI/' + args.model)
        tokenizer = Tokenizer(encoder=t.encode, decoder=t.decode, vocab_size=50304, name='gpt2')
    elif args.model.startswith('phi'):
        t = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
        tokenizer = Tokenizer(encoder=t.encode, decoder=t.decode, vocab_size=51200, name='phi')

    return tokenizer
