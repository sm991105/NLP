import os

import argparse
import json
import numpy as np
import pickle
from tqdm import tqdm
from scipy.spatial import distance

np.random.seed(1234)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weat_file_path", type=str, help="weat file where the tasks are defined", 
        default="./data/weat.json")
    parser.add_argument("--out_file", type=str, help="output JSON file where the output is stored", 
        required=True)
    parser.add_argument("--model_path", help="Full model path (including filename) to load from", 
        required=True)
    args, _ = parser.parse_known_args()
    return args

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def unit_vector(vec):
    return vec / np.linalg.norm(vec)

def cos_sim(v1, v2):

    """
    Cosine Similarity between the 2 vectors
    """

    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.clip(np.tensordot(v1_u, v2_u, axes=(-1, -1)), -1.0, 1.0)

def weat_association(W, A, B):

    """
    Compute Weat score for given target words W, along the attributes A & B.
    """

    return np.mean(cos_sim(W, A), axis=-1) - np.mean(cos_sim(W, B), axis=-1)

def weat_score(X, Y, A, B):

    """
    Compute differential weat score across the given target words X & Y along the attributes A & B.
    """

    x_association = weat_association(X, A, B)
    y_association = weat_association(Y, A, B)

    tmp1 = np.mean(x_association, axis=-1) - np.mean(y_association, axis=-1)
    tmp2 = np.std(np.concatenate((x_association, y_association), axis=0))

    return tmp1 / tmp2

def balance_word_vectors(vec1, vec2):
    diff = len(vec1) - len(vec2)

    if diff > 0:
        vec1 = np.delete(vec1, np.random.choice(len(vec1), diff, 0), axis=0)
    else:
        vec2 = np.delete(vec2, np.random.choice(len(vec2), -diff, 0), axis=0)

    return (vec1, vec2)

def get_word_vectors(words, model, vocab_token_to_id):

    """
    Return list of word embeddings for the given words using the passed model and tokeniser
    """

    output = []

    emb_size = len(model[0])

    for word in words:
        try:
            output.append(model[vocab_token_to_id[word]])
        except:
            pass

    return np.array(output)

def compute_weat(weat_path, model, vocab_token_to_id):

    """
    Compute WEAT score for the task as defined in the file at `weat_path`, and generating word embeddings from the passed model and tokeniser.
    """

    with open(weat_path) as f:
        weat_dict = json.load(f)

    all_scores = {}

    for data_name, data_dict in weat_dict.items():
        # Target
        X_key = data_dict['X_key']
        Y_key = data_dict['Y_key']
        
        # Attributes
        A_key = data_dict['A_key']
        B_key = data_dict['B_key']

        X = get_word_vectors(data_dict[X_key], model, vocab_token_to_id)
        Y = get_word_vectors(data_dict[Y_key], model, vocab_token_to_id)
        A = get_word_vectors(data_dict[A_key], model, vocab_token_to_id)
        B = get_word_vectors(data_dict[B_key], model, vocab_token_to_id)

        if len(X) == 0 or len(Y) == 0:
            print('Not enough matching words in dictionary')
            continue

        X, Y = balance_word_vectors(X, Y)
        A, B = balance_word_vectors(A, B)

        score = weat_score(X, Y, A, B)
        all_scores[data_name] = str(score)

    return all_scores

def dump_dict(obj, output_path):
    with open(output_path, "w") as file:
        json.dump(obj, file)

if __name__ == '__main__':

    args = parse_args()

    vocab_token_to_id, model = pickle.load(open(args.model_path, 'rb'))

    bias_score = compute_weat(args.weat_file_path, model, vocab_token_to_id)

    print("Final Bias Scores")
    print(json.dumps(bias_score, indent=4))

    dump_dict(bias_score, args.out_file)