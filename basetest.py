import pandas as pd
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
import pytorch_lightning as pl
from datasetload import train_loader, test_dataset, valid_loader, STDataset
from lightningwrapper import LitModel
from pytorch_lightning.loggers import CSVLogger
from hyperparameters import EPOCHS
from customloss import get_classification_report
import os 

trainer = pl.Trainer()
basemodel = LitModel(pe="base")
checkpoint_path = "/home/igardner/smalltest/baselogssmall/bestbase.ckpt"
basemodel = LitModel.load_from_checkpoint(checkpoint_path, pe="base")

#trainer.test(model=basemodel, dataloaders=test_loader, verbose=True)




times = []
dbs = []
accuracy_l = []
folder_path = '/home/igardner/finSiDBTransformer/'


#for layout in  os.listdir(folder_path):
#inputs = torch.load('home/igardner/finSiDBTransformer/inputs.pth')
#labels = torch.load('home/igardner/finSiDBTransformer/labels.pth')
#dbs.append(len(torch.nonzeros(inputs[0])))
#test_dataset = STDataset(inputs,labels)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=True,  num_workers=4)
#start_time = time.time()
trainer.test(model=basemodel, dataloaders=test_loader, verbose=True)


data = {'times': basemodel.runtime, 'dbs':basemodel.dbs, 'acc':basemodel.accuracylist}
df = pd.DataFrame.from_dict(data)

# Save the DataFrame to a CSV file
df.to_csv('/home/igardner/output.csv', index=False)
  
  
  
