# xcimport tensorflow as tf
import pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

"""
Element-wise fuzzy logic operators for tensorflow.
Supports traditional NumPy/Tensorflow broadcasting.

To use in LTN formulas (broadcasting w.r.t. ltn variables appearing in a formula), 
wrap the operator with `ltn.WrapperConnective` or `ltn.WrapperQuantifier`. 
"""

eps = 1e-4
not_zeros = lambda x: (1-eps)*x + eps
not_ones = lambda x: (1-eps)*x

class Not_Std:
    def __call__(self,x):
        return 1.-x
class Not_Godel:
    def __call__(self,x):
        return torch.equal(x,0).type(x.dtype)
        # return tf.cast(tf.equal(x,0),x.dtype)

class And_Min:
    def __call__(self,x,y):
        return torch.min(x, y)
        # return tf.minimum(x,y)
class And_Prod:
    def __init__(self,stable=True):
        self.stable = stable
    def __call__(self,x,y,stable=None):
        stable = self.stable if stable is None else stable
        if stable:
            x, y = not_zeros(x), not_zeros(y)
        return torch.mul(x,y)
class And_Luk:
    def __call__(self,x,y):
        return torch.max(x+y-1.,0.)

class Or_Max:
    def __call__(self,x,y):
        return torch.max(x,y)
class Or_ProbSum:
    def __init__(self,stable=True):
        self.stable = stable
    def __call__(self,x,y,stable=None):
        stable = self.stable if stable is None else stable
        if stable:
            x, y = not_ones(x), not_ones(y)
        return x + y - torch.mul(x,y)
class Or_Luk:
    def __call__(self,x,y):
        return torch.min(x+y,1.)

class Implies_KleeneDienes:
    def __call__(self,x,y):
        return torch.max(1.-x,y)
class Implies_Godel:
    def __call__(self,x,y):
        # return tf.where(tf.less_equal(x, y), tf.ones_like(x), y)
        return torch.where(torch.less_equal(x,y),torch.ones_like(x),y)
class Implies_Reichenbach:
    def __init__(self,stable=True):
        self.stable = stable
    def __call__(self,x,y,stable=None):
        stable = self.stable if stable is None else stable
        if stable:
            x, y = not_zeros(x), not_ones(y)
        return 1.-x+torch.mul(x,y)
class Implies_Goguen:
    def __init__(self,stable=True):
        self.stable = stable
    def __call__(self,x,y,stable=None):
        stable = self.stable if stable is None else stable
        if stable:
            x = not_zeros(x)
        return torch.where(torch.less_equal(x,y),torch.ones_like(x),torch.divide(y,x))
class Implies_Luk:
    def __call__(self,x,y):
        return torch.min(1.-x+y,1.)

class Equiv:
    """Returns an operator that computes: And(Implies(x,y),Implies(y,x))"""
    def __init__(self, and_op, implies_op):
        self.and_op = and_op
        self.implies_op = implies_op
    def __call__(self, x, y):
        return self.and_op(self.implies_op(x,y), self.implies_op(y,x))

class Aggreg_Min:
    def __call__(self,xs,axis=None,keepdims=False):
        return tf.reduce_min(xs,axis=axis,keepdims=keepdims)
class Aggreg_Max:
    def __call__(self,xs,axis=None,keepdims=False):
        return tf.reduce_max(xs,axis=axis,keepdims=keepdims)
class Aggreg_Mean:
    def __call__(self,xs,axis=None,keepdims=False):
        return tf.reduce_mean(xs,axis=axis,keepdims=keepdims)
class Aggreg_pMean:
    def __init__(self,p=2,stable=True):
        self.p = p
        self.stable = stable
    def __call__(self,xs,axis=None,keepdims=False,p=None,stable=None):
        p = self.p if p is None else p 
        stable = self.stable if stable is None else stable
        if stable:
            xs = not_zeros(xs)
        return tf.pow(tf.reduce_mean(tf.pow(xs,p),axis=axis,keepdims=keepdims),1/p)
class Aggreg_pMeanError:
    def __init__(self,p=2,stable=True):
        self.p = p
        self.stable = stable
    def __call__(self,xs,axis=None,keepdims=False,p=None,stable=None):
        p = self.p if p is None else p
        stable = self.stable if stable is None else stable
        if stable:
            xs = not_ones(xs)
        return 1.-tf.pow(tf.reduce_mean(tf.pow(1.-xs,p),axis=axis,keepdims=keepdims),1/p)