from datasetload import train_loader, test_loader, valid_loader
from lightningwrapper import *
from pytorch_lightning.loggers import CSVLogger
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from hyperparameters import EPOCHS
from customloss import get_classification_report



logger = CSVLogger(save_dir='/home/igardner/newresults', name='nonelogs')

checkpoint_callback2 = ModelCheckpoint(
    save_top_k=1,
    monitor="val_acc",
    every_n_epochs=10,
    mode="max",
    dirpath="/home/igardner/newresults/nonelogs/",
    filename="bb",
)


# training loop

trainer = pl.Trainer(max_epochs=EPOCHS, logger=logger, check_val_every_n_epoch=10, strategy='ddp_find_unused_parameters_true',
                     callbacks=[checkpoint_callback2], devices=2, accelerator="gpu")


nonemodel = LitModel(pe="NA")
trainer.fit(model=nonemodel, train_dataloaders=train_loader, val_dataloaders=valid_loader)

filepath = '/home/igardner/nonelogs/none_model.pth'
torch.save(nonemodel.transformer.state_dict(), filepath)

trainer.test(model=nonemodel, dataloaders=test_loader, verbose=True, ckpt_path="/home/igardner/newresults/nonelogs/bb.ckpt")


targets = torch.cat(nonemodel.testtarget)
predictions = torch.cat(nonemodel.testpred)

dataframe = get_classification_report(targets, predictions)

dataframe.to_csv("/home/igardner/newresults/nonelogs/testclassificationreport.csv")

