import os
import networkx as nx
from pysat.formula import CNF

import numpy as np
import torch


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
def train_model(constraints, objectives, model, optimizer, criterion, bs):

    # log :
    print(model)
    print("Number of parameters : ", sum(p.numel() for p in model.parameters()))


    # enumerate epochs
    for epoch in range(20):
        print("--- Epoch {} ---".format(epoch))
        for I, constraint in enumerate(constraints):
            optimizer.zero_grad()
            #import pdb;pdb.set_trace()
            # get initial embeddings from node2Vec (no gradient here) # TODO : no more Node2Vec --> degree would be better IMO
            nb_nodes = n_nodes_from_constraints(constraint)
            graph = constraints_to_graph(constraint)
            liste_nodes = list(graph.nodes)
            adj_mat = torch.tensor(np.array([nx.to_numpy_matrix(graph)]))
            
            nodes_init_embeddings = []  # TODO : replace by one-hot encodings with degree if litteral (or 0 if clause)
            for i , (node, degree) in enumerate(list(dict(graph.in_degree).items())):   # needs directed graphs ! 
                if False and "c" in str(node) :
                    nodes_init_embeddings.append([-1,-1])   # maybe not, maybe we should embed them as "normal nodes"
                else :
                    nodes_init_embeddings.append([float(degree), float(graph.out_degree[node])])
            nodes_init_embeddings=torch.tensor(nodes_init_embeddings)
            #nodes_init_embeddings=torch.rand(len(liste_nodes), 64)
            #import pdb;pdb.set_trace()

            # compute the model output
            #logits = model(nodes_init_embeddings, adj_mat.permute(1, 2, 0))
            logits = model(nodes_init_embeddings, adj_mat)
            
            # add projection/summation head to get the number of satisfied assignments
            # trick
            #import pdb; pdb.set_trace()
            temp = 0.1
            sftm = torch.nn.functional.softmax(logits[0] / temp, -1)
            # need to duplicate sftm so that there is the negatives. Probably smarter way to do it
            litteral_lines=[]
            for ii,node in enumerate(liste_nodes):
              if type(node)==int:
                litteral_lines.append(ii)
            liste_nodes_litterals = [liste_nodes[x] for x in litteral_lines]
            liste_nodes_litterals_full = [x for x in liste_nodes_litterals]
            for jj,lit in enumerate(liste_nodes_litterals):
              liste_nodes_litterals_full.append(-abs(lit))
            sftm_lit = sftm[litteral_lines]
            
            sftm_lit_bar = torch.ones_like(sftm_lit, requires_grad=True) - sftm_lit
            sftm_lit_full = torch.cat((sftm_lit, sftm_lit_bar), 0)
            
            sat = torch.sum(torch.max(sftm_lit_full[:, 0][constraint_to_ids(constraint, liste_nodes_litterals_full)], -1).values,   # ici faut faire attention a si c'est un moins ou un plus
                            dtype=torch.float32)
            
            # calculate loss
            targets = objectives[I]  # TODO get real !! Check
            loss = 10*criterion(sat, targets)
            # no need for additional loss for hard constraints here !
            loss.backward()
            optimizer.step()

            if I % 1 == 0 :
                print("loss :", loss.item(), "   sat : ", sat.item())
    print("sat : ", sat)
    # print(sftm)


