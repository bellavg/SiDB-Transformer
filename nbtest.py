import pandas as pd
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
import pytorch_lightning as pl
from datasetload import test_loader
from lightningwrapper import LitModel
from pytorch_lightning.loggers import CSVLogger
from hyperparameters import EPOCHS
from customloss import get_classification_report
import os

logger = CSVLogger(save_dir='/home/igardner/newresults/', name='baselogs')

trainer = pl.Trainer(logger=logger)
basemodel = LitModel(pe="base")
checkpoint_path = "/home/igardner/newresults/baselogs/bestbase.ckpt"
basemodel = LitModel.load_from_checkpoint(checkpoint_path, pe="base")

trainer.test(model=basemodel, dataloaders=test_loader, verbose=True)

targets = torch.cat(basemodel.testtarget)
predictions = torch.cat(basemodel.testpred)

dataframe = get_classification_report(targets, predictions)

dataframe.to_csv("/home/igardner/newresults/baselogs/testclassificationreport.csv")



logger2 = CSVLogger(save_dir='/home/igardner/newresults/', name='nonelogs')
trainer2 = pl.Trainer(logger=logger2)
nonemodel = LitModel(pe="NA")
checkpoint_path = "/home/igardner/newresults/nonelogs/bb.ckpt"
nonemodel = LitModel.load_from_checkpoint(checkpoint_path, pe="NA")


trainer2.test(model=nonemodel, dataloaders=test_loader, verbose=True)

targets = torch.cat(nonemodel.testtarget)
predictions = torch.cat(nonemodel.testpred)

dataframe = get_classification_report(targets, predictions)

dataframe.to_csv("/home/igardner/newresults/nonelogs/testclassificationreport.csv")
