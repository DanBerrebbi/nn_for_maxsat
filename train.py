import os
import networkx as nx
from pysat.formula import CNF

import numpy as np
import torch


def constraints_to_graph(constraints):
  G = nx.Graph()
  for i,clause in enumerate(constraints[0]):
    G.add_nodes_from(clause)
    G.add_node("c_{}".format(i))
    for node in clause:
      G.add_edge(node,"c_{}".format(i))

  # add hard constraint # TODO see how to handle those edges, different type ?
  hard_c = get_hard_constraints(constraints[0])
  for c in hard_c:
      G.add_edge(c[0],c[1])
  return G



def get_hard_constraints(constraints):
  hard=set()
  for clause in constraints:
    for litteral in clause :
      hard.add((abs(litteral),(-1)*abs(litteral)))
  return hard

def n_nodes_from_constraints(constraints):
  return len(constraints_to_graph(constraints).nodes)


def get_hard_constraints(constraints):
  hard=set()
  for clause in constraints:
    for litteral in clause :
      hard.add((abs(litteral),(-1)*abs(litteral)))
  hard_bis=[[x[0],x[1]] for x in list(hard)]
  return hard_bis

def constraint_to_ids(constraint, liste_nodes):
  new_constraint=[]
  for c in constraint[0]:
    new_c = []
    for x in c:
      new_c.append(liste_nodes.index(x))
    new_constraint.append(new_c)
  return new_constraint


def get_broken(hard_constraints, sftm):
  broken=0
  for c in hard_constraints:
    if (sftm[c[0]][0]>0.5 and sftm[c[1]][0]>0.5) or (sftm[c[0]][0]<0.5 and sftm[c[1]][0]<0.5):
      broken+=1
  return broken


def read_dimacs_directory(path):
    constraints, objectives = [], []
    for file in os.listdir(path):
        stats=file.split("_")
        for stat in stats:
            if "nclauses" in stat:
                obj=int(stat.split("=")[-1])
            if "maxsatcost" in stat:
                obj-=int(stat.split("=")[-1].split(".")[0])
        constraints.append([CNF(from_file=path+'/'+file).clauses])  # je rajoute les crochets pour batch_size 1 # lets see after
        objectives.append(torch.tensor(float(obj), requires_grad=True))
    return constraints, objectives




# for lots of reasons, batch size 1 is better to use, lets discuss it later. If needed we accum grad instead.
def train_model(constraints, objectives, model, optimizer, criterion, bs):

    # log :
    print(model)
    print("Number of parameters : ", sum(p.numel() for p in model.parameters()))


    # enumerate epochs
    for epoch in range(20):
        print("--- Epoch {} ---".format(epoch))
        for i, constraint in enumerate(constraints):
            optimizer.zero_grad()

            # get initial embeddings from node2Vec (no gradient here) # TODO : no more Node2Vec --> degree would be better IMO
            nb_nodes = n_nodes_from_constraints(constraint)
            graph = constraints_to_graph(constraint)
            liste_nodes = list(graph.nodes)
            adj_mat = torch.tensor(np.array([nx.to_numpy_matrix(graph)]))
            #import pdb;pdb.set_trace()
            nodes_init_embeddings = []  # TODO : replace by one-hot encodings with degree if litteral (or 0 if clause)
            for i , (node,degree) in enumerate(list(dict(graph.degree).items())):
                if "c" in str(node) :
                    nodes_init_embeddings.append(0)
                else :
                    nodes_init_embeddings.append(degree)
            nodes_init_embeddings=torch.nn.functional.one_hot(torch.tensor(nodes_init_embeddings), num_classes=model.init_dim)



            # compute the model output
            logits = model(nodes_init_embeddings[0], adj_mat.permute(1, 2, 0))

            # add projection/summation head to get the number of satisfied assignments
            # trick
            temp = 0.001
            sftm = torch.nn.functional.softmax(logits[0] / temp, -1)
            sat = torch.sum(torch.max(sftm[:, 0][constraint_to_ids(constraint, liste_nodes)], -1).values,
                            dtype=torch.float32)

            # calculate loss
            targets = objectives[i]  # TODO get real !! Check
            loss = criterion(sat, targets)
            # add hard constraints :
            hard_constraints = constraint_to_ids([list(get_hard_constraints(constraint[0]))], liste_nodes)
            hard_loss = 0
            for c in hard_constraints:
                hard_loss += criterion(torch.tensor(1.0), sftm[c[0], 1] + sftm[c[1], 1])

            loss += 10 * hard_loss
            loss.backward()
            optimizer.step()

            if i % 10 == 0 :
                print("loss :", loss.item())
                print("broken : ", get_broken(hard_constraints, sftm), "   sat : ", sat.item())
    print("broken : ", get_broken(hard_constraints, sftm), "   sat : ", sat)
    # print(sftm)


