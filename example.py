from train import train_model
from train import read_dimacs_directory
from classes import GATCodeur
import torch
from torch.nn import MSELoss, L1Loss


bs=1
init_dim = 2
hidden_dim_gat = 64
output_dim_gat = 64
ff_dim = 64
n_heads = 1
emb_dim = 64
dimacs_directory = './dimacs_files/test_vsmall'
n_transformer_layers = 2
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

optimizer = torch.optim.Adam(model.parameters(), lr=0.1, betas=(0.9, 0.98), eps=1e-09)
#optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

constraints, objective = read_dimacs_directory(dimacs_directory)

#train_model([constraints[0]], [objective[0]], model, optimizer, criterion, bs, n_epochs=10, debug=False, temp=0.1, temp2=0.01,  gumbel=True, tau=0.01)

train_model([constraints[0]], torch.tensor([float(len(constraints[0][0]))]), model, optimizer, criterion, bs, n_epochs=40, debug=False, temp=0.1, temp2=0.1,  gumbel=False, tau=1)

if False :
    somme=0
    for i in range(35):
        model = GATCodeur(n_layers=n_transformer_layers,
                          in_features=init_dim,
                          n_hidden=hidden_dim_gat,
                          ff_dim=ff_dim,
                          n_heads=n_heads,
                          emb_dim=emb_dim,
                          qk_dim=emb_dim,
                          v_dim=emb_dim,
                          dropout=dropout)
        somme+=train_model([constraints[0]], torch.tensor([3.]), model, optimizer, criterion, bs, n_epochs=10, debug=False, temp=0.1, temp2=0.01,  gumbel=False, tau=1)
    print(somme/35)
