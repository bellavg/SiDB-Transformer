import torch.nn as nn
import torch
import math
from hyperparameters import MAXDBS


class BaseAbsPE(nn.Module):
    def __init__(self, embedding_dim, grid_size):
        super(BaseAbsPE, self).__init__()
        self.embed = torch.nn.Embedding(grid_size, embedding_dim)
        self.ed = embedding_dim
        self.grid_size = grid_size

    # normalize cardinal coordinates grid
    def forward(self, b):
        grid_size = self.grid_size
        e_dim = self.ed
        y_embedded_coords = self.embed(torch.arange(grid_size).cuda()).cuda()
        y_embedded_coords = y_embedded_coords.repeat(grid_size, 1, 1)
        x_embedded_coords = y_embedded_coords.transpose(0, 1).cuda()
        embedded_coords = x_embedded_coords + y_embedded_coords
        embedded_coords = embedded_coords.reshape(grid_size, grid_size, e_dim)

        return embedded_coords.repeat(b, 1, 1, 1).cuda()



def get_physical(x, mindim=True, gridsize=16):
    distance_matrix = torch.load("/home/igardner/finSiDBTransformer/16distance_matrix.pth")


    rdm = torch.zeros(x.shape[0], gridsize, gridsize, MAXDBS)
    for batch_index, batch in enumerate(x):
        batchnz = torch.nonzero(batch)
        for i, currenti in enumerate(batchnz):
            for j, comp_cord in enumerate(batchnz[i:]):
                loc = comp_cord[0] * gridsize + comp_cord[1]
                rdm[batch_index][currenti[0]][currenti[1]][j] = distance_matrix[currenti[0]][currenti[1]][loc]
                loc2 = currenti[0] * gridsize + currenti[1]
                rdm[batch_index][comp_cord[0]][comp_cord[1]][i] = distance_matrix[comp_cord[0]][comp_cord[1]][loc2]

    if mindim:
        dis_neighbor = torch.where(rdm == 0, torch.tensor(100.0), rdm)
        dmn = torch.min(dis_neighbor, keepdim=True, dim=-1).values
        dmn = torch.where(dmn == 100, torch.tensor(0.0), dmn)
        return dmn
    else:
        return rdm


class AbsolutePositionalEncoding(nn.Module):
    def __init__(self, d_model, grid_size):
        super().__init__()
        self.d_model = d_model
        self.grid_size = grid_size
        self.embedding = nn.Parameter(self.generate_positional_encoding(), requires_grad=False)

    def generate_positional_encoding(self):
        position = torch.arange(0, self.grid_size).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * -(math.log(10000.0) / self.d_model))
        pos_enc = torch.zeros((self.grid_size, self.grid_size, self.d_model))
        pos_enc[:, :, 0::2] = torch.sin(position * div_term)
        pos_enc[:, :, 1::2] = torch.cos(position * div_term)
        return pos_enc.unsqueeze(0)  # Adding extra dimension for batch size
    def forward(self,b):
        # Assuming x is of shape (batch_size, grid_size, grid_size, d_model)
        return self.embedding.repeat(b, 1, 1, 1)
