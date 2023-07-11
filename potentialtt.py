from datasetload import train_loader, test_loader, valid_loader
from lightningwrapper import *
from pytorch_lightning.loggers import CSVLogger
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from hyperparameters import EPOCHS
from customloss import get_classification_report


torch.cuda.empty_cache()

checkpoint_callback2 = ModelCheckpoint(
    save_top_k=1,
    monitor="val_acc",
    every_n_epochs=10,
    mode="max",
    dirpath="/home/igardner/traintest/potentiallogs/",
    filename="best",
)


logger = CSVLogger(save_dir='/home/igardner/traintest/', name='potentiallogs')
# training loop

trainer = pl.Trainer(max_epochs=30, limit_train_batches=50, limit_val_batches=10, logger=logger, check_val_every_n_epoch=10, strategy='ddp_find_unused_parameters_true',
                     callbacks=[checkpoint_callback2], devices=2, accelerator="gpu")

potentialmodel = LitModel(pe="potential")
trainer.fit(model=potentialmodel, train_dataloaders=train_loader, val_dataloaders=valid_loader)

trainer.test(model=potentialmodel, dataloaders=test_loader, verbose=True, ckpt_path="/home/igardner/traintest/potentiallogs/best.ckpt")

targets = torch.cat(potentialmodel.testtarget)
predictions = torch.cat(potentialmodel.testpred)

dataframe = get_classification_report(targets, predictions)

dataframe.to_csv("/home/igardner/traintest/potentiallogs/testclassificationreport.csv")
