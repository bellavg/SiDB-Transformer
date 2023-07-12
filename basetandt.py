from pytorch_lightning.callbacks import ModelCheckpoint
import torch
import pytorch_lightning as pl
from datasetload import train_loader, test_loader, valid_loader
from lightningwrapper import LitModel
from pytorch_lightning.loggers import CSVLogger
from hyperparameters import EPOCHS
from customloss import get_classification_report

torch.cuda.empty_cache()

checkpoint_callback2 = ModelCheckpoint(
    save_top_k=1,
    monitor="val_acc",
    every_n_epochs=10,
    mode="max",
    dirpath="/home/igardner/newresults/baselogs/",
    filename="bestbase",
)

logger = CSVLogger(save_dir='/home/igardner/newresults/', name='baselogs')
# training loop

trainer = pl.Trainer(max_epochs=EPOCHS, logger=logger, check_val_every_n_epoch=10, strategy='ddp_find_unused_parameters_true',
                     callbacks=[checkpoint_callback2], devices=2, accelerator="gpu")

basemodel = LitModel(pe="base")
trainer.fit(model=basemodel, train_dataloaders=train_loader, val_dataloaders=valid_loader)

trainer.test(model=basemodel, dataloaders=test_loader, verbose=True, ckpt_path="/home/igardner/newresults/baselogs/bestbase.ckpt")

targets = torch.cat(basemodel.testtarget)
predictions = torch.cat(basemodel.testpred)

dataframe = get_classification_report(targets, predictions)

dataframe.to_csv("/home/igardner/newresults/baselogs/testclassificationreport.csv")



