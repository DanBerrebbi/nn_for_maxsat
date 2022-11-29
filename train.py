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
def train_model(constraints, objectives, model, optimizer, criterion, bs, n_epochs=20, debug=False, temp=0.01, temp2=0.1, gumbel=False, tau=0.1):

    # log :
    print(model)
    print("Number of parameters : ", sum(p.numel() for p in model.parameters()))


    # enumerate epochs
    for epoch in range(n_epochs):
        print("--- Epoch {} ---".format(epoch))
        for I, constraint in enumerate(constraints):
            optimizer.zero_grad()

            nb_nodes = n_nodes_from_constraints(constraint)
            graph = constraints_to_graph(constraint)
            liste_nodes = list(graph.nodes)
            adj_mat = torch.tensor(np.array([nx.to_numpy_matrix(graph)]))


            # initial emeddings : degree
            nodes_init_embeddings = []  # TODO : replace by one-hot encodings with degree if litteral (or 0 if clause)
            for i , (node, degree) in enumerate(list(dict(graph.in_degree).items())):   # needs directed graphs ! 
                if True and "c" in str(node) :
                    nodes_init_embeddings.append([-1,-1])   # maybe not, maybe we should embed them as "normal nodes"
                else :
                    nodes_init_embeddings.append([float(degree), float(graph.out_degree[node])])
            nodes_init_embeddings=torch.tensor(nodes_init_embeddings)

            # compute the model output
            #logits = model(nodes_init_embeddings, adj_mat.permute(1, 2, 0))
            logits = model(nodes_init_embeddings, adj_mat)

            litteral_lines = []
            for ii, node in enumerate(liste_nodes):
                if type(node) == int:
                    litteral_lines.append(ii)
            liste_nodes_litterals = [liste_nodes[x] for x in litteral_lines]
            liste_nodes_litterals_full = [x for x in liste_nodes_litterals]
            for jj, lit in enumerate(liste_nodes_litterals):
                liste_nodes_litterals_full.append(-abs(lit))


            # add projection/summation head to get the number of satisfied assignments
            # trick

            if gumbel : # TODO maybe get rid of the softmax at the end of the model !
                liste_sat = []
                #import pdb; pdb.set_trace()
                liste_one_hot = [F.gumbel_softmax(logits, tau=tau, hard=True) for _ in range(30)] # TODO : try with hard = False also !
                print([x[litteral_lines] for x in liste_one_hot])
                for assign in liste_one_hot:
                    sftm_lit = assign[0][litteral_lines]
                    sftm_lit_bar = torch.ones_like(sftm_lit, requires_grad=True) - sftm_lit
                    sftm_lit_full = torch.cat((sftm_lit, sftm_lit_bar), 0)
                    #print(torch.max(sftm_lit_full[:, 0][constraint_to_ids(constraint, liste_nodes_litterals_full)]))
                    #sat = torch.sum(
                    #    torch.max(sftm_lit_full[:, 0][constraint_to_ids(constraint, liste_nodes_litterals_full)],
                    #              -1).values, dtype=torch.float32)
                    #assert 7==0, sftm_lit_full[:, 0].shape
                    AUX=torch.stack([sftm_lit_full[:, 0][cc] for cc in constraint_to_ids(constraint, liste_nodes_litterals_full)])
                    print(AUX)
                    sat = torch.sum(torch.nn.functional.softmax(AUX / temp2, -1) * AUX)
                    liste_sat.append(sat)
                    #print(torch.sum(torch.round(torch.max(sftm_lit_full[:, 0][constraint_to_ids(constraint, liste_nodes_litterals_full)],-1).values)))
                # print(liste_sat)
                sat = torch.sum(torch.stack(liste_sat)) / len(liste_sat)

            else : # pr recup les perf que j avais avant
                sftm = torch.nn.functional.softmax(logits[0] / temp, -1)

                #print(logits)
                # need to duplicate sftm so that there is the negatives. Probably smarter way to do it
                sftm_lit = sftm[litteral_lines]
                sftm_lit_bar = torch.ones_like(sftm_lit, requires_grad=True) - sftm_lit
                sftm_lit_full = torch.cat((sftm_lit, sftm_lit_bar), 0)
                AUX = torch.stack(
                    [sftm_lit_full[:, 0][cc] for cc in constraint_to_ids(constraint, liste_nodes_litterals_full)])
                sat = torch.sum(torch.nn.functional.softmax(AUX / temp2, -1) * AUX)
                print("AUX : ", AUX)
                # calculate loss
            targets = objectives[I]
            loss = criterion(sat, targets)
                # no need for additional loss for hard constraints here !

            loss.backward()

            if debug :
                for name, param in model.named_parameters():
                    print(name, param.grad)

            optimizer.step()
            #import pdb; pdb.set_trace()
            print(logits[0][litteral_lines])

            if I % 1 == 0 :
                print("loss :", loss.item(), "   sat : ", sat.item())

                #print(liste_sat)
    print("sat : ", sat)
    return sat
    # print(sftm)


