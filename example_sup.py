from train_sup import train_model, eval_model
from train_sup import read_dimacs_directory_ass
from classes import GATCodeur
import torch
from torch.nn import CrossEntropyLoss, L1Loss


init_dim = 64
hidden_dim_gat = 128
output_dim_gat = 128
ff_dim = 128
n_heads = 1
emb_dim = 128
dimacs_directory = './dimacs_files/test3_ass/grp1'
n_transformer_layers = 0
dropout = 0.0

#torch.manual_seed(222222)
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
criterion = CrossEntropyLoss()

#optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-09)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

constraints, objective, ass = read_dimacs_directory_ass(dimacs_directory)
#objective = torch.tensor([[1.,0.],[0.,1.],[0.,1.]])

train_model(constraints[:1000], ass[:1000], model, optimizer, criterion, log=False, n_epochs=3, debug=False, temp=1, gumbel=False)

eval_model(constraints[-100:], ass[-100:], model)
