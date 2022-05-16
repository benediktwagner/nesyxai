import os
import re
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import argparse
import pandas as pd
import numpy as np
from nltk import ngrams
from collections import Counter
import pickle
from model.utils import clean, get_yaml_config, get_ngrams, parse_kb, sequentialise
from model.model_fn import ModelWrapper
from model.input_fn import input_fn
import tensorflow as tf
from cav import CAV


parser = argparse.ArgumentParser()

parser.add_argument('-dd', '--data_dir', default='data')
parser.add_argument('-md', '--model_dir', default='experiments')
parser.add_argument('--cuda', default='2')
parser.add_argument('-c', '--concept_names', default='Family Hierarchy Drugs')
parser.add_argument('--ngrams', type=int, default=3)


swem_max_endpoints = dict(
    input_='model/emb_matrix_lookup/Identity',
    bottleneck='model/dim_reduction', #model/bottleneck/BiasAdd' DEPENDING ON WHICH LAYER
    probs='model/Softmax',  # redundant
    output='model/output_logits/BiasAdd'
)

lstm_endpoints = dict(
    input_='model/emb_matrix_lookup/Identity',
    bottleneck='model/bottleneck/BiasAdd',
    output='model/output_logits/BiasAdd'
)


if __name__ == '__main__':
    tf.reset_default_graph()
    tf.logging.set_verbosity(tf.logging.INFO)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ', '.join([i for i in args.cuda])

    params = get_yaml_config(os.path.join(args.model_dir, 'config.yaml'))
    architecture = params['architecture']

    if architecture == 'swem_max':
        endpoints = swem_max_endpoints
    elif architecture == 'lstm':
        endpoints = lstm_endpoints
    else:
        raise ValueError('No such architecture')

    # CONCEPTS
    print('Getting concepts...')
    train = pd.read_csv(os.path.join(args.data_dir, 'train.csv'))
    train['text'] = train['text'].apply(clean)
    labels = train['labels'].values
    data = pd.read_csv(os.path.join('data_path', 'concept_search.csv'), nrows=50000)
    data['text'] = data['text'].apply(clean)

    #
    # def skipi(index):
    #     if index % 10 == 0:
    #         return True
    #     return False
    #
    # fields = ['labels', 'test1']
    # n_rows = 1655633
    # skip = np.arange(n_rows)
    # skip = np.delete(skip, np.arange(0, n_rows, 10))    # longform = pd.read_csv('/Users/ben/GitHub/nlp_cav/data_path/cleaaneed.csv',header=0,skiprows=lambda x: skipi(x)) #CHANGED THIS
    # longform = pd.read_csv('/Users/ben/GitHub/nlp_cav/data_path/cleaaneed.csv',usecols=fields,skiprows=skip) #CHANGED THIS


    counter = Counter()
    for t in data['text']:
        counter.update(get_ngrams(t, n=args.ngrams))

    ngram_tokens = [' '.join(a) for a, i in counter.most_common()]
    concept_names = args.concept_names.split()

    concepts = {key: {'pos': None, 'neg': None} for key in concept_names}

    for conc in concept_names:

        word_lis = []
        word_lis.append(conc)
        words = parse_kb(conc.lower(),5)
        word_lis.extend(words)

        concepts[conc]['pos'] = [c for c in ngram_tokens if conc.lower() in c]
        # concepts[conc]['pos'] = [c for c in ngram_tokens if clean(word_lis) in c]
        for word in word_lis:
            collected_ngrams = [c for c in ngram_tokens if re.search(r'\b' + word + r'\b', c)]
            concepts[conc]['pos'].extend(collected_ngrams)
        # concepts[conc]['pos'] = [c for c in ngram_tokens if [word for word in word_lis] in c] #TOMORROW WORK ON HERE
        concepts[conc]['neg'] = np.random.choice(ngram_tokens, size=len(concepts[conc]['pos']), replace=False)

    pickle.dump(concepts, open(os.path.join(args.model_dir, 'concepts.pkl'), 'wb'))

    # BOTTLENECKS
    print('Getting bottlenecks...')
    mw = ModelWrapper(args.model_dir, endpoints)

    graph = tf.get_default_graph()

    sess = tf.Session(graph=graph)
    sess.run(tf.global_variables_initializer())
    cav_bottlenecks = dict()

    for key, val in concepts.items():

        X_conc = mw.calculate_bottleneck(sess, concepts[key]['pos'])
        X_rand = mw.calculate_bottleneck(sess, concepts[key]['neg'])
        X = np.append(X_conc, X_rand, axis=0)
        y = np.array([1] * len(X_conc) + [0] * len(X_rand))

        cav_bottlenecks[key] = (X, y)

    pickle.dump(cav_bottlenecks, open(os.path.join(args.model_dir, 'cav_bottlenecks.pkl'), 'wb'))

    # CAVS
    print('Getting CAVs...')
    cavs = dict()

    for key, val in cav_bottlenecks.items():
        cav = CAV()
        v = cav.fit(cav_bottlenecks[key][0], cav_bottlenecks[key][1])  # (X, y)
        cavs[key] = v

    pickle.dump(cavs, open(os.path.join(args.model_dir, 'cavs.pkl'), 'wb'))

    # GRADIENTS
    print('Getting gradients...')
    # set = lambda: input_fn(os.path.join(args.data_dir, 'train.csv'), params, True)
    df_seq = sequentialise(train,params['seq_len'])
    labels = df_seq['labels'].values
    dat = df_seq['text'].tolist()

    grads = mw.calculate_grad(sess, labels, dat)
    # grads = mw.calculate_grad(sess, labels, train['text'].tolist())

    # set = input_fn(os.path.join(args.data_dir, 'train.csv'), params, True)
    # iterator = set.make_initializable_iterator()
    # data = iterator.get_next()
    # data_X, data_y = iterator.get_next()
    # data_in = data_X['x']
    # data_lab = data_X['y']
    # with tf.Session() as sess:
    #     try:
    #         while True:
    #             grads = mw.calculate_grad(sess, data_lab, data_in)
    #     except tf.errors.OutOfRangeError:
    #         pass

    pickle.dump(grads, open(os.path.join(args.model_dir, 'grads.pkl'), 'wb'))
