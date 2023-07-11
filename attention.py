import math
import torch
import MinkowskiEngine as ME
import torch.nn as nn
import gc

"""
Architecture
3 linear embeddings of x of nn.Linear(out = embeddings * heads)
get positions, embed positions in nn.Linear to emb,emb
dot plus scaled for queries and keys and positions
look at masks
softmax dot product of qk
multiple dotqk soft it by values
unify heads

"""


class SparseAttention(nn.Module):
    def __init__(self, heads, ed):
        super().__init__()
        assert ed % heads == 0, f'Embedding dimension ({ed}) should be divisible by nr. of heads ({heads})'

        self.n_heads = heads
        self.ed = ed

        d_head = ed// heads

        self.q_mappings = nn.ModuleList([nn.Linear(d_head, d_head, bias=True) for _ in range(self.n_heads)])
        self.k_mappings = nn.ModuleList([nn.Linear(d_head, d_head, bias=True) for _ in range(self.n_heads)])
        self.v_mappings = nn.ModuleList([nn.Linear(d_head, d_head, bias=True) for _ in range(self.n_heads)])

        self.unifyheads = nn.Linear(ed, ed, bias=True)

        self.softmax = nn.Softmax(dim=-1)

        self.div=math.sqrt(ed)


    def forward(self, inputx, b, gs, mask):

        h = self.n_heads

        e = self.ed

        s = e // h
        #div_t = torch.full((inputx.shape[0], s), fill_value=(math.sqrt(e)))

        inputx,_,_ = inputx.dense(shape=torch.Size([b, e, gs, gs]))
        inputx = inputx.view(b, h, s, gs*gs).permute(1, 0, 3, 2) #head, batch, hw, s
        #ex = ME.to_sparse(inputx[0])
        #inputx = inputx.transpose(-2,-1).reshape(h, -1, s)

        result = []
        for hi, seq in enumerate(inputx): #for each head
          q_mapping = self.q_mappings[hi]
          k_mapping = self.k_mappings[hi]
          v_mapping = self.v_mappings[hi]
          # check is unsqueeze dim 1 to have BCXX necessary or if B, hw, s is okay should be dim S for features has to be B s, h
          q, k, v = q_mapping(seq), k_mapping(seq), v_mapping(seq)
          attention = self.softmax((q @ k.transpose(-2,-1)) / self.div)
          #qk = ME.MinkowskiSumPooling(kernel_size=seq.shape[0], stride=1, dimension=(qk.coordinates.shape[-1]-1))(qk)
          result.append(attention @ v)


        result = torch.cat(result).reshape(h, b, gs, gs, s).permute(1, 2, 3, 0,4).reshape(b, gs, gs, e)
        result = self.unifyheads(result)
        result = result.reshape(-1,e)
        result = result[mask]


        return result.view(-1,e)
