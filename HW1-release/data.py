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
        center_word = np.ndarray(shape=(self.batch_size), dtype=np.int32)
        # labels
        context_word = np.ndarray(shape=(self.batch_size, 1), dtype=np.int32)

        # stride: for the rolling window
        stride = 1

        ### TODO(students): start
        center_pos = 0  # 2
        index = center_pos
        for i in self.data:  # c
            for j in (self.data[center_pos - self.skip_window: center_pos + self.skip_window + 1]):  # b c d
                # data 앞이나 끝에서 예외처리
                if (center_pos - self.skip_window < 0) or (center_pos + self.skip_window > (len(self.data) - 1)):
                    continue
                else:
                    # 자기자신은 넣지 않으므로 예외처리
                    if j == i:
                        continue
                    else:
                        if index == self.batch_size:
                            # 이 리턴문 여기로 옮겼어
                            return torch.LongTensor(center_word), torch.LongTensor(context_word)
                        center_word[index] = i  # c c
                        context_word[index] = j  # b d
                        index += 1
            center_pos += 1

        '''current_batch_size = 0
        if self.data_index == 0:
            self.data_index += self.skip_window
        else:
            self.data_index

        self.data_index %= len(self.data)
        # Used to keep track of the number of words in the batch so far

        while current_batch_size < self.batch_size:
            context_word[current_batch_size:current_batch_size + self.num_skips] = self.data[self.data_index]
            # Extracting all possible context words in the window
            temp_window = self.data[self.data_index - self.skip_window:self.data_index] + self.data[self.data_index + 1:self.data_index + 1 + self.skip_window]
            # Random sampling of context words. num_skips could be much lesser than the window size at times
            sampled_window = np.random.choice(temp_window, size=self.num_skips, replace=False)
            center_word[current_batch_size:current_batch_size + self.num_skips] = sampled_window
            # Updation for exit condition
            current_batch_size += self.num_skips
            self.data_index += stride
        ### TODO(students): end

        return torch.LongTensor(center_word), torch.LongTensor(context_word)'''


