import os
import networkx as nx
from pysat.formula import CNF

import numpy as np
import torch
import torch.nn.functional as F


def constraints_to_graph(constraints, directed=True):
  if directed:
    G = nx.DiGraph()
    for i,clause in enumerate(constraints[0]):
      for node in clause:
        if node == abs(node):   # only one node for x_1 and -x_1
          G.add_edge(node,"c_{}".format(i))
        else :
          G.add_edge("c_{}".format(i), abs(node))
    return G
  else:
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
def train_model(constraints, objectives, model, optimizer, criterion, log=True, n_epochs=20, debug=False, temp=0.01, gumbel=False):

    if log :
        print(model)
        print("Number of parameters : ", sum(p.numel() for p in model.parameters()))

    # enumerate epochs
    for epoch in range(n_epochs):
        if log:
            print("--- Epoch {} ---".format(epoch))
        for I, constraint in enumerate(constraints):  # iterate over the set of SAT problems, constraint is a list of clauses
            optimizer.zero_grad()

            nb_nodes = n_nodes_from_constraints(constraint)
            graph = constraints_to_graph(constraint)  # build the graph : a directed edge links a clause to its literal, the direction depends on the sign of the literal
            liste_nodes = list(graph.nodes)
            adj_mat = torch.tensor(np.array([nx.to_numpy_matrix(graph)]))


            # initial emeddings : degree ;  TODO : we should now try using the outputs of Neurosat
            nodes_init_embeddings = []
            for i , (node, degree) in enumerate(list(dict(graph.in_degree).items())):   # need directed graphs to differentiate x_1 and -x_1 !
                nodes_init_embeddings.append([float(degree), float(graph.out_degree[node])])
            nodes_init_embeddings=torch.tensor(nodes_init_embeddings)

            # compute the model output
            logits = model(nodes_init_embeddings, adj_mat, temp, gumbel)

            # compute the loss/objective value

            # First, only keep values of literal embeddings (not clauses embeddings)
            litteral_lines = []
            for ii, node in enumerate(liste_nodes):
                if type(node) == int:
                    litteral_lines.append(ii)
            liste_nodes_litterals = [liste_nodes[x] for x in litteral_lines]
            liste_nodes_litterals_full = [x for x in liste_nodes_litterals]
            for jj, lit in enumerate(liste_nodes_litterals):
                liste_nodes_litterals_full.append(-abs(lit))

            # need to duplicate sftm so that there is the negatives. Probably there exist smarter way to do it
            sftm_lit = logits[0][litteral_lines]
            sftm_lit_bar = torch.ones_like(sftm_lit, requires_grad=True) - sftm_lit
            sftm_lit_full = torch.cat((sftm_lit, sftm_lit_bar), 0)
            AUX = torch.stack(
                [sftm_lit_full[:, 0][cc] for cc in constraint_to_ids(constraint, liste_nodes_litterals_full)]) # AUX : list of clauses with the value of literals of this clause

            # chose a way to calculate the SAT value / objective / number of clauses satisfied

            # 1: sum all (then multiply targets by 3 for the loss value)
            sat = torch.sum(AUX)

            # 2 : softmaxed weighted sum
            # sat = torch.sum(torch.nn.functional.softmax(AUX / temp, -1) * AUX) # TODO : we can put another temerature here, like temp2, a bit overloading

            # 3 : max : but problem of exploration (can be solved by setting gumbel=True)
            #sat = torch.sum(
                #    torch.max(sftm_lit_full[:, 0][constraint_to_ids(constraint, liste_nodes_litterals_full)],
                #              -1).values, dtype=torch.float32)


            # calculate loss
            targets = objectives[I]*3.0
            loss = criterion(sat, targets) + 0*abs(sftm_lit_full[:,0]-sftm_lit_full[:,1]).sum() # second term to enforce parity and so exploration
                # no need for additional loss for hard constraints here !

            loss.backward()

            if debug :
                for name, param in model.named_parameters():
                    print(name, param.grad)

            optimizer.step()
            if log :
                print(logits[0][litteral_lines])

            if log :
                print("loss :", loss.item(), "   sat : ", sat.item())
    print(AUX)
    print("sat : ", sat)
    return sat
    # print(sftm)


