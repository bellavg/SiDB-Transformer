import numpy as np
import torch
from hyperparameters import MAXDBS
from sklearn.metrics import classification_report
import pandas as pd

import numpy as np
import torch
from hyperparameters import MAXDBS
import torch.nn as nn
import random
from sklearn.metrics import classification_report
import pandas as pd

# add muliplication by nj charge to the potential matrix
def get_potential(x, b, gridsize):
    epbs_matrix = torch.load("/home/igardner/SiDBTransformer/electric_potential_matrix.pth") #size 42, 42, 1764
    x = x.squeeze(-1)

    epbs_out = torch.zeros(b, gridsize, gridsize, MAXDBS)
    for batch_index, batch in enumerate(x):
        batchnz = torch.nonzero(batch)
        for i, currenti in enumerate(batchnz):
            for j, comp_cord in enumerate(batchnz[i:]):
                loc = comp_cord[0] * gridsize + comp_cord[1]
                epbs_out[batch_index][currenti[0]][currenti[1]][j] = epbs_matrix[currenti[0]][currenti[1]][loc]
                loc2 = currenti[0] * gridsize + currenti[1]
                epbs_out[batch_index][comp_cord[0]][comp_cord[1]][i] = epbs_matrix[comp_cord[0]][comp_cord[1]][loc2]

    epbs_out = torch.sum(epbs_out, dim=-1)
    return epbs_out

def get_classification_report(y_test, y_pred):
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    df_classification_report = pd.DataFrame(report).transpose()
    df_classification_report = df_classification_report.sort_values(by=['f1-score'], ascending=False)
    return df_classification_report

