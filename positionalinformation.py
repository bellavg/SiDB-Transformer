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



class AIAYNPE(nn.Module):
    def __init__(self, embedding_dim, grid_size):
        super(AIAYNPE, self).__init__()

        self.embedding_dim = embedding_dim
        self.grid_size = grid_size

    def forward(self, batch_size):
        embedding_dim = self.embedding_dim
        gs = self.grid_size
        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2).float() * (
                    -torch.log(torch.tensor(10000.0)) / embedding_dim))

        pos_x = torch.arange(gs).unsqueeze(0).repeat(gs, 1).view(-1, 1)
        pos_y = torch.arange(gs).unsqueeze(1).repeat(1, gs).view(-1, 1)

        positional_encodings_x = torch.zeros(gs * gs, embedding_dim)
        positional_encodings_y = torch.zeros(gs * gs, embedding_dim)

        positional_encodings_x[:, 0::2] = torch.sin(pos_x.float() / div_term)
        positional_encodings_x[:, 1::2] = torch.cos(pos_x.float() / div_term)

        positional_encodings_y[:, 0::2] = torch.sin(pos_y.float() / div_term)
        positional_encodings_y[:, 1::2] = torch.cos(pos_y.float() / div_term)

        positional_encodings_x = positional_encodings_x.view(1, gs, gs, embedding_dim).repeat(batch_size, 1, 1, 1)
        positional_encodings_y = positional_encodings_y.view(1, gs, gs, embedding_dim).repeat(batch_size, 1, 1, 1)

        return positional_encodings_x + positional_encodings_y


def positionalencoding2d(d_model, height, width, batch_size):
    """
    :param d_model: dimension of the model
    :param height: height of the positions
    :param width: width of the positions
    :return: d_model*height*width position matrix
    """
    if d_model % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dimension (got dim={:d})".format(d_model))
    pe = torch.zeros(d_model, height, width)
    # Each dimension use half of d_model
    d_model = int(d_model / 2)
    div_term = torch.exp(torch.arange(0., d_model, 2) *
                         -(math.log(10000.0) / d_model))
    pos_w = torch.arange(0., width).unsqueeze(1)
    pos_h = torch.arange(0., height).unsqueeze(1)
    pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)

    return pe.permute(1, 2, 0).unsqueeze(0).repeat(batch_size, 1, 1, 1)


def get_physical(x, mindim, gridsize=42):
    distance_matrix = torch.load("/home/igardner/SiDBTransformer/distance_matrix.pth")
    x = x.squeeze(1)

    rdm = torch.zeros(x.shape[0], gridsize, gridsize, MAXDBS)
    for batch_index, batch in enumerate(x):
        batchnz = torch.nonzero(batch)
        for i, currenti in enumerate(batchnz):
            for j, comp_cord in enumerate(batchnz[i:]):
                loc = comp_cord[0] * 42 + comp_cord[1]
                rdm[batch_index][currenti[0]][currenti[1]][j] = distance_matrix[loc][currenti[0]][currenti[1]]
                loc2 = currenti[0] * 42 + currenti[1]
                rdm[batch_index][comp_cord[0]][comp_cord[1]][i] = distance_matrix[loc2][comp_cord[0]][comp_cord[1]]

    if mindim:
        dis_neighbor = torch.where(rdm == 0, torch.tensor(100.0), rdm)
        dmn = torch.min(dis_neighbor, keepdim=True, dim=-1).values
        dmn = torch.where(dmn == 100, torch.tensor(0.0), dmn)
        return torch.nn.functional.normalize(dmn, p=2.0, dim=-1)
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
