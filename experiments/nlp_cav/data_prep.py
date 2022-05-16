"""
New features and data cleaning
"""

import pandas as pd
from collections import Counter
import re
import pickle
from sklearn.model_selection import train_test_split
import os
import argparse
from model.utils import save_vocab_to_txt_file, clean

parser = argparse.ArgumentParser()
parser.add_argument('-dd', '--data_dir', default='data')
parser.add_argument('-mf', '--min_freq', type=int, default=5)
parser.add_argument('-s', '--sample', action='store_true')

parser.set_defaults(sample=False)


def update_vocabulary(tokens, counter):
    counter.update(tokens)


def create_vocab(pd_series, min_count=0):
    vocabulary = Counter()
    _ = pd_series.apply(lambda x: update_vocabulary(x.split(), vocabulary))
    vocabulary = [tok for tok, count in vocabulary.most_common() if count >= min_count]
    vocabulary.insert(0, '<PAD>')
    vocabulary.insert(1, '<UNK>')

    vocab_length = len(vocabulary)
    return vocabulary, vocab_length

def loadConceptNetModel(conceptFile):
    print("Loading ConceptNet Model")
    f = open(conceptFile,'r')
    model = {}
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        model[word] = embedding
    print("Done.",len(model)," words loaded!")
    return model


if __name__ == '__main__':

    args = parser.parse_args()

    if args.sample:
        nrows = 10000
    else:
        nrows = None

    data = pd.read_csv(os.path.join(args.data_dir, 'dataset.csv'), nrows=nrows)

    data['text'] = data['text'].apply(lambda x: clean(x))

    train, valid = train_test_split(data, stratify=data['labels'], test_size=0.3, random_state=24)

    data.to_csv(os.path.join(args.data_dir, 'full.csv'), index=False)
    train.to_csv(os.path.join(args.data_dir, 'train.csv'), index=False)
    valid.to_csv(os.path.join(args.data_dir, 'eval.csv'), index=False)

    vocabulary, vocab_size = create_vocab(data['text'], min_count=args.min_freq)
    print(f'Vocab size = {vocab_size}')

    mdl = loadConceptNetModel("/Users/ben/PycharmProjects/untitled1/numberbatch/numberbatch-en-17.06.txt")

    i=0
    embedding_matrix = np.zeros((vocab_size, 300))
    for word in vocabulary:
        embedding_vector = mdl.get(word)
        if embedding_vector is not None: # words not found in embedding index will be all zeros.
            embedding_matrix[i] = embedding_vector
        i+=1

    pickle.dump(embedding_matrix, open(os.path.join(args.data_dir, 'emb.pkl'), 'wb'))
    save_vocab_to_txt_file(vocabulary, os.path.join(args.data_dir, 'vocab.txt'))
