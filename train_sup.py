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

def read_dimacs_directory_ass(path):
    constraints, objectives, ass = [], [], []
    for file in os.listdir(path):
        stats=file.split("_")
        for stat in stats:
            if "nclauses" in stat:
                obj=int(stat.split("=")[-1])
            if "maxsatcost" in stat:
                obj-=int(stat.split("=")[-1].split(".")[0])
        constraints.append([CNF(from_file=path+'/'+file).clauses])  # je rajoute les crochets pour batch_size 1 # lets see after
        ass.append([int(p) for p in CNF(from_file=path+'/'+file).comments[-1][2:].split("[")[1].split(']')[0].split(",")])
        objectives.append(torch.tensor(float(obj), requires_grad=True))
    ass = ass_to_obj(ass)
    return constraints, objectives, ass

def ass_to_obj(ass):
    obj = []
    for asss in ass :
        one_obj = []
        for lit in asss:
            if abs(lit)==lit:
                one_obj.append([1.0,0.0])
            else:
                one_obj.append([0.0, 1.0])
        obj.append(one_obj)
    return torch.tensor(obj)

def constraint_to_embeddings(constraint, seed, init="random", init_dim=64):
    graph = constraints_to_graph(
        constraint)  # build the graph : a directed edge links a clause to its literal, the direction depends on the sign of the literal
    liste_nodes = list(graph.nodes)
    adj_mat = torch.tensor(np.array([nx.to_numpy_matrix(graph)]))
    np.random.seed(seed)
    # initial emeddings : degree ;  TODO : we should now try using the outputs of NeuroSat or random embeddings
    nodes_init_embeddings = []
    for i, (node, degree) in enumerate(
            list(dict(graph.in_degree).items())):
        if init == "degree":  # need directed graphs to differentiate x_1 and -x_1 !
            nodes_init_embeddings.append([float(degree), float(graph.out_degree[node])])
        elif init == "random":
            nodes_init_embeddings.append(np.float32(np.random.random((init_dim))))
        elif init == "neurosat":
            raise Exception("to be implemented")
    nodes_init_embeddings = torch.tensor(np.array(nodes_init_embeddings))
    return nodes_init_embeddings, adj_mat, liste_nodes



# for lots of reasons, batch size 1 is better to use, lets discuss it later. If needed we accum grad instead.
def train_model(constraints, objectives, model, optimizer, criterion, log=True, n_epochs=20, debug=False, temp=0.01, gumbel=False, init_emb="random"):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    if log :
        print(model)
        print("Number of parameters : ", sum(p.numel() for p in model.parameters()))

    # enumerate epochs
    for epoch in range(n_epochs):
        acc_train = 0
        if log or True:
            print("--- Epoch {} ---".format(epoch))
            eval_model(constraints[9*len(constraints)//10:], objectives[9*len(constraints)//10:], model)
        for I, constraint in enumerate(constraints[:9*len(constraints)//10]):  # iterate over the set of SAT problems, constraint is a list of clauses
            optimizer.zero_grad()
            nodes_init_embeddings , adj_mat, liste_nodes = constraint_to_embeddings(constraint, seed=I, init=init_emb, init_dim=model.in_features)

            # compute the model output
            logits = model(nodes_init_embeddings.to(device), adj_mat.to(device), temp, gumbel)

            # compute the loss/objective value

            # First, only keep values of literal embeddings (not clauses embeddings)
            litteral_lines = {}
            for ii, node in enumerate(liste_nodes):
                if type(node) == int:
                    litteral_lines[node]=ii
            litteral_lines_s = {k:v for k,v in sorted(litteral_lines.items(), key= lambda x:x[0])}
            selected_lines = list(litteral_lines_s.values())

            keep_target = [k-1 for k in litteral_lines_s.keys()]
            # need to duplicate sftm so that there is the negatives. Probably there exist smarter way to do it
            sftm_lit = logits[0][selected_lines]

            # calculate loss

            targets = objectives[I][keep_target] #*3.0
            loss = criterion(sftm_lit, targets.to(device))

            loss.backward()

            if debug :
                for name, param in model.named_parameters():
                    print(name, param.grad)

            optimizer.step()
            if log :
                print(logits[0][litteral_lines])

                #print("loss :", loss.item(), "   sftm : ", sftm_lit)
                print("sftm : ", sftm_lit)
                print("loss :", loss.item())
                print("$$$"*30)
            acc_train += ((sftm_lit.max(dim=-1).indices == targets.to(device).max(dim=-1).indices).float().sum()).item()/len(sftm_lit)
            if np.random.random()<0.00 :
                print("loss :", loss.item())
                #import pdb; pdb.set_trace()
                print("acc :", (sftm_lit.max(dim=-1).indices == targets.to(device).max(dim=-1).indices).float().sum())
        print("train accuracy :",acc_train/(I+1))
    #print("sftm : ", sftm_lit)
    #return sftm_lit
    # print(sftm)



def eval_model(constraints, objectives, model, init_emb="random"):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    acc = 0.0
    for I, constraint in enumerate(
            constraints):  # iterate over the set of SAT problems, constraint is a list of clauses
        nodes_init_embeddings, adj_mat, liste_nodes = constraint_to_embeddings(constraint, seed=I, init=init_emb,
                                                                               init_dim=model.in_features)

        # compute the model output
        logits = model(nodes_init_embeddings.to(device), adj_mat.to(device), temp=0.001, gumbel=False)

        # compute the loss/objective value

        # First, only keep values of literal embeddings (not clauses embeddings)
        litteral_lines = {}
        for ii, node in enumerate(liste_nodes):
            if type(node) == int:
                litteral_lines[node] = ii
        litteral_lines_s = {k: v for k, v in sorted(litteral_lines.items(), key=lambda x: x[0])}
        selected_lines = list(litteral_lines_s.values())

        keep_target = [k - 1 for k in litteral_lines_s.keys()]
        # need to duplicate sftm so that there is the negatives. Probably there exist smarter way to do it
        sftm_lit = logits[0][selected_lines]

        # calculate loss

        targets = objectives[I][keep_target]  # *3.0
        acc += ((sftm_lit.max(dim=-1).indices == targets.to(device).max(dim=-1).indices).float().sum()).item()/len(sftm_lit)
        #print(acc/I)
    print("valid accuracy :", acc/len(constraints))
    return acc/len(constraints)