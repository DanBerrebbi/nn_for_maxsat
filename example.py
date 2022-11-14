from train import train_model
from train import read_dimacs_directory
from classes import GATCodeur

import torch
from torch.nn import MSELoss, L1Loss


bs=1
init_dim = 2
hidden_dim_gat = 128
output_dim_gat = 128
ff_dim = 128
n_heads = 1
emb_dim = 128
dimacs_directory = './dimacs_files/test_small'
n_transformer_layers = 0
dropout = 0.0


model = GATCodeur(n_layers=n_transformer_layers,
                  in_features=init_dim,
                  n_hidden=hidden_dim_gat,
                  ff_dim=ff_dim,
                  n_heads=n_heads,
                  emb_dim=emb_dim,
                  qk_dim=emb_dim,
                  v_dim=emb_dim,
                  dropout=dropout)

# loss and optimization  # TODO : see if GAT has a particular optimization
criterion = MSELoss()

optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, betas=(0.9, 0.98))
#optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

constraints, objective = read_dimacs_directory(dimacs_directory)

train_model([constraints[0]], objective, model, optimizer, criterion, bs)
