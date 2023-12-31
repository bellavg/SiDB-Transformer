from datasetload import train_loader, test_loader, valid_loader
from lightningwrapper import *
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from hyperparameters import EPOCHS
from customloss  import get_classification_report

torch.cuda.empty_cache()

logger = CSVLogger(save_dir='/home/igardner/newresults/', name='physical2logs')
# training loop
checkpoint_callback2 = ModelCheckpoint(
    save_top_k=1,
    monitor="val_acc",
    every_n_epochs=10,
    mode="max",
    dirpath="/home/igardner/newresults/physical2logs/",
    filename="bb",
)


trainer = pl.Trainer(max_epochs=EPOCHS, logger=logger, check_val_every_n_epoch=10, strategy='ddp_find_unused_parameters_true',
                     callbacks=[checkpoint_callback2], devices=2, accelerator="gpu")

physical1model = LitModel(pe="physical2")
trainer.fit(model=physical1model, train_dataloaders=train_loader, val_dataloaders=valid_loader)

filepath = '/home/igardner/physical2logs/physical1_model.pth'
torch.save(physical1model.transformer.state_dict(), filepath)

trainer.test(model=physical1model, dataloaders=test_loader, verbose=True, ckpt_path="/home/igardner/newresults/physical2logs/bb.ckpt")

targets = torch.cat(physical1model.testtarget)
predictions = torch.cat(physical1model.testpred)

dataframe = get_classification_report(targets, predictions)

dataframe.to_csv("/home/igardner/newresults/physical2logs/testclassificationreport.csv")
