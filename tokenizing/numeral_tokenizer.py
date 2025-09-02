# numbers = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']


# class NumeralTokenizer:
#     def __init__(self, num_nodes):
#         self.num_nodes = num_nodes
#         # Define encoder and decoder as a dictionary
#         self.encoder = {str(i): i for i in range(num_nodes)}
#         self.encoder['|'] = num_nodes
#         self.encoder['='] = num_nodes + 1
#         self.encoder['/'] = num_nodes + 2
#         self.encoder['$'] = num_nodes + 3

#         self.decoder = {i: i for i in range(num_nodes)}
#         self.decoder[num_nodes] = '|'
#         self.decoder[num_nodes + 1] = '='
#         self.decoder[num_nodes + 2] = '/'
#         self.decoder[num_nodes + 3] = '$'
#         self.decoder[-1] = ':'

#     def encode(self, x):
#         out = []
#         i = 0
#         while i < len(x):
#             if x[i] == ',':
#                 i += 1
#                 continue
#             s = ''
#             j = 0
#             while i + j < len(x) and x[i + j] in numbers:
#                 s += x[i + j]
#                 j += 1
#             if s == '':
#                 s = x[i]
#                 i += 1
#             else:
#                 i += j
#             out.append(self.encoder[s])

#         return out

#     def decode(self, x):
#         return [self.decoder[i] for i in x]

import re

numbers = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']


class NumeralTokenizer:
    def __init__(self, num_nodes):
        self.num_nodes = num_nodes
        
        # Define encoder and decoder as a dictionary
        self.encoder = {str(i): i for i in range(num_nodes)}
        self.encoder['|'] = num_nodes
        self.encoder['='] = num_nodes + 1
        self.encoder['/'] = num_nodes + 2
        self.encoder['$'] = num_nodes + 3

        self.decoder = {i: i for i in range(num_nodes)}
        self.decoder[num_nodes] = '|'
        self.decoder[num_nodes + 1] = '='
        self.decoder[num_nodes + 2] = '/'
        self.decoder[num_nodes + 3] = '$'
        self.decoder[-1] = ':'

        # Pre-compile the regular expression for a significant performance boost in encode().
        # This regex finds sequences of one or more digits (\d+) OR any single character
        # that is not a comma ([^,]), perfectly replicating the original logic.
        self._token_regex = re.compile(r'\d+|[^,]')

    def encode(self, x):
        """
        Encodes a string into a list of integers using a pre-compiled regular expression
        for significantly faster tokenization.
        """
        # The findall() method is highly optimized and quickly splits the string into all matching tokens.
        tokens = self._token_regex.findall(x)
        
        # A list comprehension provides an efficient way to map the string tokens to their integer codes.
        return [self.encoder[token] for token in tokens]

    def decode(self, x):
        """
        Decodes a list of integers back into their original representation.
        The original implementation using a list comprehension is already highly
        efficient for most use cases.
        """
        return [self.decoder[i] for i in x]