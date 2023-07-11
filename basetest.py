from pytorch_lightning.callbacks import ModelCheckpoint
import torch
import pytorch_lightning as pl
from datasetload import train_loader, test_loader, valid_loader
from lightningwrapper import LitModel
from pytorch_lightning.loggers import CSVLogger
from hyperparameters import EPOCHS
from customloss import get_classification_report
test_loader = DataLoader(test_dataset, batch_size=BATCHSIZE, shuffle=False, pin_memory=True,  num_workers=4)

trainer = pl.Trainer()
basemodel = LitModel(pe="base")
checkpoint_path = "/home/igardner/smalltest/baselogssmall/bestbase.ckpt"
lightning_module = YourLightningModule.load_from_checkpoint(checkpoint_path)

trainer.test(model=basemodel, dataloaders=test_loader, verbose=True)

targets = torch.cat(basemodel.testtarget)
predictions = torch.cat(basemodel.testpred)

dataframe = get_classification_report(targets, predictions)

dataframe.to_csv("/home/igardner/smalltest/baselogssmall/testclassificationreport.csv")

times = []
dbs = []
accuracy_l = []

for layout in  os.listdir(folder_path):
  inputs, labels = torch.load(layout)
  dbs.append(len(torch.nonzeros(inputs[0]))
  test_dataset = STDataset(inputs[:50],labels[:50])
  test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=True,  num_workers=4)
  start_time = time.time()
  trainer.test(model=basemodel, dataloaders=test_loader, verbose=True)
  end_time = time.time()
  targets = torch.cat(basemodel.testtarget)
  predictions = torch.cat(basemodel.testpred)
  correct_predictions = (predictions == targets).sum().item()
  time.append(end_time - start_time)
  total_samples = targets.size(0)
  accuracy = correct_predictions / total_samples
  accuracy_l.append(accuracy)



  
  
  
