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
basemodel = LitModel(pe="potential")
checkpoint_path = "/home/igardner/currentresults/potentiallogs/best.ckpt"
lightning_module = YourLightningModule.load_from_checkpoint(checkpoint_path)

trainer.test(model=basemodel, dataloaders=test_loader, verbose=True)

targets = torch.cat(basemodel.testtarget)
predictions = torch.cat(basemodel.testpred)

dataframe = get_classification_report(targets, predictions)

dataframe.to_csv("/home/igardner/pottest/pottestclassificationreport.csv")


