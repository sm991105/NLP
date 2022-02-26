"""
author-gh: @adithya8
editor-gh: ykl7
"""

import collections
import random
import numpy as np
import torch

np.random.seed(1234)
torch.manual_seed(1234)


# Read the data into a list of strings.
def read_data(filename):
    with open(filename) as file:
        text = file.read()
        data = [token.lower() for token in text.strip().split(" ")]
    return data


def build_dataset(words, vocab_size):
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(vocab_size - 1))
    # token_to_id dictionary, id_to_taken reverse_dictionary
    vocab_token_to_id = dict()
    for word, _ in count:
        vocab_token_to_id[word] = len(vocab_token_to_id)
    data = list()
    unk_count = 0
    for word in words:
        if word in vocab_token_to_id:
            index = vocab_token_to_id[word]
        else:
            index = 0  # dictionary['UNK']
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    vocab_id_to_token = dict(zip(vocab_token_to_id.values(), vocab_token_to_id.keys()))
    return data, count, vocab_token_to_id, vocab_id_to_token


class Dataset:
    def __init__(self, data, batch_size=128, num_skips=8, skip_window=4):
        self.data_index = 0
        self.data = data
        assert batch_size % num_skips == 0
        assert num_skips <= 2 * skip_window

        self.batch_size = batch_size
        self.num_skips = num_skips
        self.skip_window = skip_window

    def reset_index(self, idx=0):
        self.data_index = idx

    def generate_batch(self):
        """
        Write the code generate a training batch

        batch will contain word ids for context words. Dimension is [batch_size].
        labels will contain word ids for predicting(target) words. Dimension is [batch_size, 1].
        """
        # batch
        center_word = np.ndarray(shape=self.batch_size, dtype=np.int32)
        # labels
        context_word = np.ndarray(shape=self.batch_size, dtype=np.int32)

        # stride: for the rolling window
        stride = 1

        ### TODO(students): start
        span = 2 * self.skip_window + 1
        buffer = collections.deque(maxlen=span)
        for _ in range(span):
            buffer.append(self.data[self.data_index])
            self.data_index = (self.data_index+1) % len(self.data)

        for i in range(self.batch_size):
            target = self.skip_window
            targets_to_avoid = [self.skip_window]
            for j in range(self.num_skips):
                while target in targets_to_avoid:
                    target = random.randint(0, span-1)
                targets_to_avoid.append(target)
                center_word[i * self.num_skips + j] = self.buffer[self.skip_window]
                context_word[i * self.num_skips + j, 0] = self.buffer[target]
            buffer.append(self.data[self.data_index])
            self.data_index = (self.data_index + 1) % len(self.data)
        ### TODO(students): end

        return torch.LongTensor(center_word), torch.LongTensor(context_word)
    # torch.tensor 은 자료형이라고 생각하면 됨
    # extract context & center words -> return