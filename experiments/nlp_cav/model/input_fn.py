"""
Here we define inputs to the model
"""

import tensorflow as tf
import pandas as pd
import numpy as np

from keras.preprocessing.text import Tokenizer


def build_vocab(file_name):
    tokens = tf.contrib.lookup.index_table_from_file(
        file_name,
        num_oov_buckets=0,
        default_value=1,
        delimiter='\n',
        name='vocab'
    )

    return tokens

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

def vectorize(string, vocab, seq_len, emb_dim):
    splitted = tf.string_split([string]).values
    vectorized = vocab.lookup(splitted)
    vectorized = vectorized[:seq_len]

    # # splitted = tf.string_split([string]).values
    # vectorized = vocab.lookup(splitted)
    # conceptnet = loadConceptNetModel("/Users/ben/PycharmProjects/untitled1/numberbatch/numberbatch-en-17.06.txt")
    # # variable = np.vstack([conceptnet, [[0.] * emb_dim]])
    # variable = tf.Variable(conceptnet, dtype=tf.float32, trainable=False)
    # embeddings = tf.nn.embedding_lookup(variable, vectorized)
    # vectorized = embeddings[:seq_len]

    # NEW

    # embedding_path = "/Users/ben/PycharmProjects/untitled1/numberbatch/numberbatch-en-17.06.txt"
    # embed_size = 300
    # max_features = 30000
    # vocab_size = 111230

    # def get_coefs(word, *arr):
    #     return word, np.asarray(arr, dtype='float32')
    #
    # embedding_index = dict(get_coefs(*o.strip().split(" ")) for o in open(embedding_path))
    #
    #
    # tokenizer = Tokenizer(num_words=vocab_size)
    # tokenizer.fit_on_texts(string)
    # tokenizer.fit_on_sequences(string)
    # word_index = tokenizer.word_index
    # embedding_matrix = np.zeros((vocab_size + 1, embed_size))
    #
    # for word, i in word_index.items():
    #     if i >= max_features: continue
    #     embedding_vector = embedding_index.get(word)
    #     if embedding_vector is not None: embedding_matrix[i] = embedding_vector
    #
    # vectorized = embedding_matrix[:seq_len]

    return vectorized


def input_fn(data_path, params, train_time=True):

    data = pd.read_csv(data_path)
    dataset = tf.data.Dataset.from_tensor_slices(({'tokens': data['text'].values}, data['labels'].values))

    vocab = build_vocab(params['vocab_path'])
    pad_value = vocab.lookup(tf.constant('<PAD>'))
    fake_padding = tf.constant(-1, dtype=tf.int64)



    if train_time:
        dataset = dataset.shuffle(params['train_size'])
        dataset = dataset.repeat(params['num_epochs'])

    dataset = dataset.map(lambda feats, labs: (vectorize(feats['tokens'], vocab, params['seq_len'], params['emb_dim']), labs))

    padded_shapes = (tf.TensorShape([params['seq_len']]), tf.TensorShape([]))
    padding_values = (pad_value, fake_padding)

    dataset = dataset.padded_batch(
        batch_size=params['batch_size'],
        padded_shapes=padded_shapes,
        padding_values=padding_values
    )

    dataset = dataset.map(lambda x, y: ({'x': x, 'y': y}, y))
    dataset = dataset.prefetch(buffer_size=2)

    return dataset
