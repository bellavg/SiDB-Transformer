
from attention import SparseAttention
import torch
import torch.nn as nn
import MinkowskiEngine as me
import gc


class Block(nn.Module):
    """Architecture
    - Self Attention
    - Dropout?
    - Add and Norm
    - Feed Forward
    - Dropout?
    - Add and Norm
    """

    def __init__(self, pe, h, ed,  d_rate):
        super().__init__()

        self.pe_type = pe

        self.attn = SparseAttention(heads=h, ed=ed)

        self.norm1 = me.MinkowskiBatchNorm(ed)

        self.mlp = nn.Sequential(
            me.MinkowskiLinear(ed, ed*4),
            me.MinkowskiGELU(),
            me.MinkowskiDropout(d_rate),
            me.MinkowskiLinear(ed*4, ed),
            )

        self.norm2 = me.MinkowskiBatchNorm(ed)

        self.medrop = me.MinkowskiDropout(d_rate)



    def forward(self, x, mask, b, gs):

        x = x + me.SparseTensor(features=self.attn(self.norm1(x), b, gs, mask),
                             coordinate_manager=x.coordinate_manager,
                             coordinate_map_key=x.coordinate_map_key)


        x = self.medrop(x)

        x = x + self.mlp(self.norm2(x))

        x = self.medrop(x)


        return x
