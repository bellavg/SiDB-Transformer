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

logger = CSVLogger(save_dir='/home/igardner/newresults/', name='potentiallogs')
# training loop

trainer = pl.Trainer(logger=logger)
ckpt_path = "/home/igardner/newresults/potentiallogs/best.ckpt"
potentialmodel =  LitModel.load_from_checkpoint(ckpt_path, pe="potential")

trainer.test(model=potentialmodel, dataloaders=test_loader, verbose=True,)

targets = torch.cat(potentialmodel.testtarget)
predictions = torch.cat(potentialmodel.testpred)

dataframe = get_classification_report(targets, predictions)

dataframe.to_csv("/home/igardner/newresults/potentiallogs/testclassificationreport.csv")




logger2 = CSVLogger(save_dir='/home/igardner/newresults/', name='physical2logs')

trainer2 = pl.Trainer(logger=logger2)
ckpt_path="/home/igardner/newresults/physical2logs/bb.ckpt"
physical1model = LitModel.load_from_checkpoint(ckpt_path, pe="physical2")

trainer2.test(model=physical1model, dataloaders=test_loader, verbose=True)

targets = torch.cat(physical1model.testtarget)
predictions = torch.cat(physical1model.testpred)

dataframe = get_classification_report(targets, predictions)

dataframe.to_csv("/home/igardner/newresults/physical2logs/testclassificationreport.csv")

