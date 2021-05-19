import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import Adam
import numpy as np
import logictensornetworks as ltn
import argparse

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--csv-path',type=str,default=None)
    parser.add_argument('--epochs',type=int,default=1000)
    parser.add_argument('--batch-size',type=int,default=64)
    args = parser.parse_args()
    dict_args = vars(args)
    return dict_args

args = parse_args()
batch_size = args['batch_size']
EPOCHS = args['epochs']
csv_path = args['csv_path']

# # Data
# Sample data from [0,1]^2.
# The groundtruth positive is data close to the center (.5,.5) (given a threshold)
# All the other data is considered as negative examples
nr_samples = 100
data = np.random.uniform([0,0],[1,1],(nr_samples,2))
t_data = torch.tensor(data)
labels = np.sum(np.square(data-[.5,.5]),axis=1)<.09
t_labels = torch.tensor(labels)
# 50 examples for training; 50 examples for testing
ds_train = TensorDataset(t_data[:50],t_labels[:50])
dl_train = DataLoader(ds_train, batch_size=batch_size)
ds_test = TensorDataset(t_data[50:],t_labels[50:])
dl_test = DataLoader(ds_test, batch_size=batch_size)

# # LTN

A = ltn.Predicate.MLP([2],hidden_layer_sizes=(16,16))
print('-----\n',A,'\n------')

# # Axioms
# 
# ```
# forall x_A: A(x_A)
# forall x_not_A: ~A(x_not_A)
# ```

Not = ltn.Wrapper_Connective(ltn.fuzzy_ops.Not_Std())
And = ltn.Wrapper_Connective(ltn.fuzzy_ops.And_Prod())
Or = ltn.Wrapper_Connective(ltn.fuzzy_ops.Or_ProbSum())
Implies = ltn.Wrapper_Connective(ltn.fuzzy_ops.Implies_Reichenbach())
Forall = ltn.Wrapper_Quantifier(ltn.fuzzy_ops.Aggreg_pMeanError(p=2),semantics="forall")
Exists = ltn.Wrapper_Quantifier(ltn.fuzzy_ops.Aggreg_pMean(p=2),semantics="exists")

formula_aggregator = ltn.fuzzy_ops.Aggreg_pMeanError(p=2)

def axioms(data, labels):
    x_A = ltn.variable("x_A",data[labels])
    x_not_A = ltn.variable("x_not_A",data[torch.logical_not(labels)])
    axioms = [
        Forall(x_A, A(x_A)),
        Forall(x_not_A, Not(A(x_not_A)))
    ]
    axioms = torch.stack(axioms)
    sat_level = formula_aggregator(axioms, axis=0)
    return sat_level, axioms


# Initialize all layers and the static graph.

for data, labels in dl_test:
    print("Initial sat level %.5f"%axioms(data, labels)[0])
    break

# # Training
# 
# Define the metrics

# TODO: create metrics with Pytorch
# metrics_dict = {
#     'train_sat': tf.keras.metrics.Mean(name='train_sat'),
#     'test_sat': tf.keras.metrics.Mean(name='test_sat'),
#     'train_accuracy': tf.keras.metrics.BinaryAccuracy(name="train_accuracy",threshold=0.5),
#     'test_accuracy': tf.keras.metrics.BinaryAccuracy(name="test_accuracy",threshold=0.5)
# }
metrics_dict = {}

optimizer = Adam(params=A.parameters() ,lr=0.001)

def train_step(data, labels):
    # sat and update
    optimizer.zero_grad()
    sat = axioms(data,labels)[0]
    loss = 1.-sat
    loss.backward()
    optimizer.step()
    # TODO: collect metrics
    # metrics_dict['train_sat'](sat)
    # # accuracy
    # predictions = A.model(data)
    # metrics_dict['train_accuracy'](labels,predictions)
    return sat

def test_step(data, labels):
    # sat and update
    sat = axioms(data, labels)[0]
    # TODO: collect metrics
    # metrics_dict['test_sat'](sat)
    # # accuracy
    # predictions = A.model(data)
    # metrics_dict['test_accuracy'](labels,predictions)
    return sat

import commons_pytorch

commons_pytorch.train(
    EPOCHS,
    metrics_dict,
    dl_train,
    dl_test,
    train_step,
    test_step,
    csv_path=csv_path,
    track_metrics=20
)

x = ltn.variable("x",data[:50])
result=A(x)
print(result)
