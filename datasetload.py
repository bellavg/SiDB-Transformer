from torch.utils.data import Dataset, DataLoader
import torch
from sklearn.model_selection import train_test_split
from hyperparameters import BATCHSIZE


class STDataset(Dataset):
    def __init__(self, t_input, target):
        self.input = t_input
        self.target = target

    def __len__(self):
        return self.input.shape[0]

    def __getitem__(self, idx):
        return self.input[idx], self.target[idx]


inputs = torch.load('/home/igardner/hpoSiDBTransformer/2235inputs.pth')
labels = torch.load('/home/igardner/hpoSiDBTransformer/2235labels.pth')

X_train, X_test, y_train,  y_test = train_test_split(inputs, labels, test_size=0.4, random_state=42)
train_dataset = STDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=BATCHSIZE, shuffle=True, pin_memory=True, num_workers=4)
X_test, X_val, y_test, y_val  = train_test_split(X_test, y_test, test_size=0.5, random_state=42)
test_dataset = STDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=BATCHSIZE, shuffle=False, pin_memory=True,  num_workers=4)
valid_dataset = STDataset(X_val, y_val)
valid_loader = DataLoader(valid_dataset, batch_size=BATCHSIZE, shuffle=False, pin_memory=True,  num_workers=4)
