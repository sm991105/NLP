"""
author-gh: @adithya8
editor-gh: ykl7
"""

import os
import pickle
import argparse

import numpy as np
import torch

from data import read_data, build_dataset, Dataset
from model import WordVec
from train import Trainer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--loss_model", help="The loss function for training the word vector", 
        default="nll", choices=["nll", "neg"])
    parser.add_argument("--checkpoint_model_path", help="Directory to save model checkpoints into", 
        default="./checkpoints")
    parser.add_argument("--model_path", help="Directory to save the final model into", default="./baseline_models")
    parser.add_argument("--skip_window", help="Context window size", type=int, default=1)
    parser.add_argument("--vocab_size", help="Size of vocabulary for word2vec model", type=int, 
        default=100000)
    parser.add_argument("--num_skips", help="Number of samples to draw in a window", type=int, default=2)
    parser.add_argument("--batch_size", help="Batch size", type=int, default=64)
    parser.add_argument("--embedding_size", help="Embedding size", type=int, default=128)
    parser.add_argument("--checkpoint_step", help="Frequency of saving checkpoints", type=int, default=50000)
    parser.add_argument("--max_num_steps", help="Maximum number of steps to train for", type=int, 
        default=100001)
    args, _ = parser.parse_known_args()
    return args

def create_path(path):
    if not os.path.exists(path):
        os.mkdir(path)
        print ("Created a path: %s"%(path))

if __name__ == "__main__":

    args = parse_args()
    print(vars(args))
    checkpoint_model_path = f'{args.checkpoint_model_path}_{args.loss_model}/'
    create_path(checkpoint_model_path)

    # Read data     
    words = read_data("./data/text8")
    print('Data size', len(words))

    vocab_size = args.vocab_size
    data, count, vocab_token_to_id, vocab_id_to_token = build_dataset(words, vocab_size)
    # save dictionary as vocabulary
    print('Most common words (+UNK)', count[:5])
    print('Sample data', data[:10], [vocab_id_to_token[i] for i in data[:10]])
    # Calculate the probability of unigrams
    # unigram_cnt = [c for w, c in count]
    count_dict = dict(count)
    unigram_cnt = [count_dict[vocab_id_to_token[i]] for i in sorted(list(vocab_token_to_id.values()))]
    data_index = 0

    dataset = Dataset(data, batch_size=args.batch_size, num_skips=args.num_skips, skip_window=args.skip_window)
    center, context = dataset.generate_batch() # return values of data.py
    for i in range(8):
        print(center[i].item(), vocab_id_to_token[center[i].item()],'->', context[i].item(), vocab_id_to_token[context[i].item()])
    dataset.reset_index()

    valid_size = 16     # Random set of words to evaluate similarity on.                                                                                                                                              
    valid_window = 100  # Only pick dev samples in the head of the distribution.                                                                                                                                      
    valid_examples = np.random.choice(valid_window, valid_size, replace=False)

    embedding_size = args.embedding_size
    model = WordVec(V=args.vocab_size, embedding_dim=embedding_size, loss_func=args.loss_model, counts=np.array(unigram_cnt))
    trainer = Trainer(model, checkpoint_model_path, vocab_id_to_token)

    max_num_steps = args.max_num_steps
    checkpoint_step = args.checkpoint_step
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Device: {device}')
    trainer.train(dataset, max_num_steps, checkpoint_step, valid_examples, device)
    model_path = args.model_path
    create_path(model_path)
    model_filepath = os.path.join(model_path, 'word2vec_%s.model'%(args.loss_model))
    pickle.dump([vocab_token_to_id, model.center_embeddings.weight.detach().cpu().numpy()], open(model_filepath, 'wb'))