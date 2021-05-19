import logging; logging.basicConfig(level=logging.INFO)
import torch
import numpy as np
import logictensornetworks as ltn
import networkx as nx
import itertools

entities = ["sue", "diana", "john", "edna", "paul", "francis", "john2",
                "john3", "john4", "joe", "jennifer", "juliet", "janice",
                "joey", "tom", "bonnie", "katie"]

parents = [
        ("sue", "diana"),
        ("john", "diana"),
        ("sue", "bonnie"),
        ("john", "bonnie"),
        ("sue", "tom"),
        ("john", "tom"),
        ("diana", "katie"),
        ("paul", "katie"),
        ("edna", "sue"),
        ("john2", "sue"),
        ("edna", "john3"),
        ("john2", "john3"),
        ("francis", "john"),
        ("john4", "john"),
        ("francis", "janice"),
        ("john4", "janice"),
        ("janice", "jennifer"),
        ("joe", "jennifer"),
        ("janice", "juliet"),
        ("joe", "juliet"),
        ("janice", "joey"),
        ("joe", "joey")]

all_relationships = list(itertools.product(entities, repeat=2))
not_parents = [item for item in all_relationships if item not in parents]

# Ground Truth Parents
parDG_truth = nx.DiGraph(parents)
# pos= nx.drawing.nx_agraph.graphviz_layout(parDG_truth, prog='dot')
# nx.draw(parDG_truth,pos,with_labels=True)

# Ground Truth Ancestors
def get_descendants(entity, DG):
    all_d = []
    direct_d = list(DG.successors(entity))
    all_d += direct_d
    for d in direct_d:
        all_d += get_descendants(d, DG)
    return all_d

ancestors = []
for e in entities:
    for d in get_descendants(e, parDG_truth):
        ancestors.append((e,d))

# ancDG_truth = nx.DiGraph(ancestors)
# pos= nx.drawing.nx_agraph.graphviz_layout(parDG_truth, prog='dot')
# nx.draw(ancDG_truth,pos,with_labels=True)

embedding_size = 4

Ancestor = ltn.Predicate.MLP([embedding_size,embedding_size],hidden_layer_sizes=[8,8])
Parent = ltn.Predicate.MLP([embedding_size,embedding_size],hidden_layer_sizes=[8,8])

g_e = {
    l: ltn.constant(np.random.uniform(low=0.,high=1.,size=embedding_size), trainable=True)
    for l in entities
}


Not = ltn.Wrapper_Connective(ltn.fuzzy_ops.Not_Std())
And = ltn.Wrapper_Connective(ltn.fuzzy_ops.And_Prod())
Or = ltn.Wrapper_Connective(ltn.fuzzy_ops.Or_ProbSum())
Implies = ltn.Wrapper_Connective(ltn.fuzzy_ops.Implies_Reichenbach())
Forall = ltn.Wrapper_Quantifier(ltn.fuzzy_ops.Aggreg_pMeanError(p=5),semantics="forall")
Exists = ltn.Wrapper_Quantifier(ltn.fuzzy_ops.Aggreg_pMean(p=5),semantics="exists")

formula_aggregator = ltn.fuzzy_ops.Aggreg_pMeanError(p=5)

# defining the theory
#@tf.function
def axioms():
    # Variables created in the training loop, so tf.GradientTape
    # keeps track of the connection with the trainable constants.
    a = ltn.variable("a",torch.stack(list(g_e.values())))
    b = ltn.variable("b",torch.stack(list(g_e.values())))
    c = ltn.variable("c",torch.stack(list(g_e.values())))

    ## Complete knowledge about parent relationships.
    ## The ancestor relationships are to be learned with these additional rules.
    axioms = [
        # forall pairs of individuals in the parent relationships: Parent(ancestor,child)
        Parent([g_e[a],g_e[c]])
        for a,c in parents
    ] + \
    [
        # forall pairs of individuals not in the parent relationships: Not(Parent([n_parent,n_child])))
        Not(Parent([g_e[a],g_e[c]]))
        for a,c in not_parents
    ] + \
    [
        # if a is parent of b, then a is ancestor of b
        Forall((a,b), Implies(Parent([a,b]),Ancestor([a,b]))),
        # parent is anti reflexive
        Forall(a, Not(Parent([a,a]))),
        # ancestor is anti reflexive
        Forall(a, Not(Ancestor([a,a]))),
        # parent is anti symmetric
        Forall((a,b), Implies(Parent([a,b]),Not(Parent([b,a])))),
        # if a is parent of an ancestor of c, a is an ancestor of c too  
        Forall(
            (a,b,c),
            Implies(And(Parent([a,b]),Ancestor([b,c])), Ancestor([a,c])),
            p=6
        ),
        # if a is an ancestor of b, a is a parent of b OR a parent of an ancestor of b
        Forall(
            (a,b),
            Implies(Ancestor([a,b]), 
                    Or(Parent([a,b]), 
                       Exists(c, And(Ancestor([a,c]),Parent([c,b])),p=6)
                      )
                   )
        )
    ]    
    # computing sat_level
    axioms = torch.stack([torch.squeeze(ax) for ax in axioms])
    sat_level = formula_aggregator(axioms, axis=0)
    return sat_level, axioms

print("Initial sat level %.5f"%axioms()[0])

trainable_variables = list(Parent.parameters())+list(Ancestor.parameters())+list(g_e.values())
optimizer = torch.optim.Adam(trainable_variables, lr=0.001)


for epoch in range(3000):
    optimizer.zero_grad()
    loss_value = 1. - axioms()[0]
    loss_value.backward()
    optimizer.step()
    if epoch%200 == 0:
        print("Epoch %d: Sat Level %.3f"%(epoch, axioms()[0]))
print("Training finished at Epoch %d with Sat Level %.3f"%(epoch, axioms()[0]))


