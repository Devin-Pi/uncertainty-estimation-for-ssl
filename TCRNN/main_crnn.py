import numpy as np
from copy import deepcopy
from packaging.version import Version
from lightning.pytorch.cli import LightningCLI, LightningArgumentParser
import torch
import os
import lightning as l
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from lightning.pytorch.loggers import CometLogger
from torch.utils.data import DataLoader
from torch.functional import F
from dataloader.dataset_cls import TSSLDataSet

from model.module import PredDOA
from model.tcrnn import CRNN

class DataModule(l.LightningDataModule):
    def __init__(self, data_dir: str = "/TSSL/data/", batch_size: tuple = [2, 1], num_workers: int = 8):
        super().__init__()
        """this class is for the datamodule
        Args:
            data_dir (str): the path of the data
            batch_size (list): a list of batch size for train and test [2, 2]
            num_workers (int, optional): the value of the num_workers. Defaults to 0.
        """

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self) -> None:
        return super().prepare_data()

    def setup(self, stage: str):
        print(stage)
        if stage == "fit":
            self.dataset_train = TSSLDataSet(
                data_dir=os.path.join(self.data_dir, "train"),
                num_data= 10000,
            )
            self.dataset_val = TSSLDataSet(
                data_dir=os.path.join(self.data_dir, "dev"),
                num_data=998,
            )
        elif stage == "test":
            self.dataset_test = TSSLDataSet(
                data_dir=os.path.join(self.data_dir, "test"),
                num_data=5000,
            )

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            self.dataset_train,
            batch_size=self.batch_size[0],
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=False
            )
    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.dataset_val,
            batch_size=self.batch_size[1],
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False
        )
    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.dataset_test,
            batch_size=self.batch_size[1],
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False
        )

class TrustedRCNN(l.LightningModule):
    def __init__(
                self,
                input_dim: int=4,
                num_classes: int=180,
                dropout: float = 0.1,
                lr=0.0005,
                compile: bool = False,
                device: str = "cuda",
                lamdba_peochs: int = 10,
                ):
        """
        Inputs:
            input_dim - Hidden dimensionality of the input
            model_dim - Hidden dimensionality to use inside the Transformer
            num_classes - Number of classes to predict per sequence element
            num_heads - Number of heads to use in the Multi-Head Attention blocks
            num_layers - Number of encoder blocks to use.
            lr - Learning rate in the optimizer
            dropout - Dropout to apply inside the model
            input_dropout - Dropout to apply on the input features
        """
        super().__init__()
        # Model init
        self.model = CRNN(
            input_dim=input_dim,
            output_dim=num_classes,
            dropout_rate=dropout,
        )

        torch.set_float32_matmul_precision('medium')

        # if compile:
        #     print("Compiling the model!")
        #     assert Version(torch.__version__) >= Version(
        #         '2.0.0'), torch.__version__
        #     self.model = torch.compile(self.model)

        # save all the parameters to self.hparams
        self.save_hyperparameters(ignore=['model'])
        self.dev = device
        self.lamdba_epochs = lamdba_peochs
        self.get_metric = PredDOA()


    def forward(self, x):
        """
        Inputs:
            x - Input features of shape [Batch, SeqLen, input_dim]
            mask - Mask to apply on the attention outputs (optional)
            add_positional_encoding - If True, we add the positional encoding to the input.
            Might not be desired for some tasks.
        """
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams.lr)

        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8988, last_epoch=-1)

        # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        #     optimizer,
        #     T_max=10,
        #     eta_min=0,
        #     last_epoch=-1
        #     )


        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': lr_scheduler,
                # 'monitor': 'valid/loss',
            }
        }


    def training_step(self, batch, batch_idx: int):
        mic_sig_batch = batch[0] # [2, 4, 256, 299] bs, c, f, t
        gt_batch = batch[1]
        pred_batch = self.model(mic_sig_batch)

        loss, evidence, U = self.ce_loss_uncertainty(pred_batch=pred_batch, gt_batch=gt_batch, current_epoch=self.current_epoch)

        # NOTE: the loss function is the cross entropy loss
        # loss = self.ce_loss(pred_batch=pred_batch, gt_batch=gt_batch)

        self.log("train/loss", loss, prog_bar=True, on_epoch=True)

        return {"loss": loss}

    def validation_step(self, batch, batch_idx: int):
        mic_sig_batch = batch[0]
        gt_batch = batch[1]

        pred_batch = self(mic_sig_batch)
        loss, evidence, U = self.ce_loss_uncertainty(pred_batch=pred_batch, gt_batch=gt_batch, current_epoch=self.current_epoch)
        # NOTE: the loss function is the cross entropy loss
        # loss = self.ce_loss(pred_batch=pred_batch, gt_batch=gt_batch)

        self.log("valid/loss", loss, sync_dist=True, on_epoch=True)

        metric = self.get_metric(pred_batch=pred_batch, gt_batch=gt_batch)
        for m in metric:
            self.log('valid/'+m, metric[m].item(), sync_dist=True, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx: int):
        mic_sig_batch = batch[0]
        gt_batch = batch[1]

        pred_batch = self(mic_sig_batch) # [2, 24, 512]
        loss, evidence, U = self.ce_loss_uncertainty(pred_batch=pred_batch, gt_batch=gt_batch, current_epoch=self.current_epoch)
        U = U.detach().cpu().numpy()

        self.log("test/loss", loss, sync_dist=True)
        metric = self.get_metric(pred_batch=pred_batch, gt_batch=gt_batch)

        for m in metric:
            self.log('test/'+m, metric[m].item(), sync_dist=True, on_epoch=True)

    def predict_step(self, batch, batch_idx: int):

        mic_sig_batch = batch[0]
        pred_batch = self.forward(mic_sig_batch)

        return pred_batch

# loss function
    def KL(self, alpha, c):
        beta = torch.ones((1, c)).to(alpha.device)
        S_alpha = torch.sum(alpha, dim=1, keepdim=True)
        S_beta = torch.sum(beta, dim=1, keepdim=True)
        lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
        lnB_uni = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(S_beta)
        dg0 = torch.digamma(S_alpha)
        dg1 = torch.digamma(alpha)
        kl = torch.sum((alpha - beta) * (dg1 - dg0), dim=1, keepdim=True) + lnB + lnB_uni
        return kl

    def ecu_loss(self, alpha, uncertainty, S, gt_cls):
        eps = 1e-10
        # the expected probability is the mean of the corresponding dirichlet distribution
        pred_scores, pred_cls = torch.max(alpha / S, dim=1, keepdim= True)
        acc_match = torch.reshape(torch.eq(pred_cls, gt_cls.unsqueeze(1)).float(), (-1, 1))

        acc_uncertain = - pred_scores * torch.log(1 - uncertainty + eps)
        inacc_certain = - (1 - pred_scores) * torch.log(uncertainty + eps)
        annealing_start = torch.tensor(0.01, dtype=torch.float32).to(self.device)
        # as for the annealing_coef, the number before self.current_epoch should be revised.
        annealing_coef = annealing_start * torch.exp(-torch.log(annealing_start) / 100 * self.current_epoch)

        ecu_loss = annealing_coef * acc_match * acc_uncertain + (1 - annealing_coef) * (1 - acc_match) * inacc_certain

        return ecu_loss



    def ce_kl_loss(self, p, alpha, c, global_step, annealing_step):
        S = torch.sum(alpha, dim=1, keepdim=True)
        E = alpha - 1
        label = F.one_hot(p, num_classes=c) # [48, 180]
        A = torch.sum(label * (torch.digamma(S) - torch.digamma(alpha)), dim=1, keepdim=True)
        # annealing_coef = min(1, global_step / annealing_step)
        alp = E * (1 - label) + 1
        annealing_coef = torch.tensor(0.01, dtype=torch.float32).to(self.device)
        B = annealing_coef * self.KL(alp, c)

        # return (A + B)
        return A

    def ce_loss_uncertainty(self, pred_batch=None, gt_batch=None, current_epoch=None):
        """
		Function: ce loss for uncertainty
		Args:
			pred_batch: doa
			gt_batch: dict{'doa'}
		Returns:
			loss
        """
        # self.log("NO.CURRENT_EPOCH",current_epoch, sync_dist=True,on_epoch=True)
        nb, nt,_ = pred_batch.shape
        pred_batch = pred_batch.reshape(nb*nt, -1) # [48, 180]

        gt_doa = gt_batch['doa'] * 180 / np.pi
        gt_doa = gt_doa[:,:,1,:].type(torch.LongTensor).to(self.device) # [2, 24, 1]
        gt_doa = gt_doa.reshape(nb*nt) # [48]
        # NOTE: There are many ways to obtain the evidence!!! You can choose the one you like.

        # evidence = torch.nn.functional.relu(pred_batch) # obtain evidence
        # evidence = F.relu(pred_batch) # obtain evidence
        evidence = torch.exp(torch.clamp(pred_batch, -10, 10))
        # evidence = torch.exp(pred_batch)
        # evidence = F.relu(pred_batch) * 100
        # evidence = F.softplus(pred_batch) * 100
        alpha = evidence + 1
        S = torch.sum(alpha, dim=1, keepdim=True)
        U = 180 / S

        edl_loss = self.ce_kl_loss(
            gt_doa,
            alpha,
            180,
            current_epoch,
            self.lamdba_epochs,
                              )

        ecu_loss = self.ecu_loss(
            alpha=alpha,
            uncertainty=U,
            S=S,
            gt_cls=gt_doa,
        )

        # loss_all = torch.mean(edl_loss + 0.3 * ecu_loss)
        loss_all = torch.mean(edl_loss)

        # self.log("S", torch.mean(S), sync_dist=True, on_epoch=True)
        # self.log("uncertainty", torch.mean(U), sync_dist=True, on_epoch=True)
        # self.log("alpha", torch.mean(alpha), sync_dist=True, on_epoch=True)
        # self.log("edl_loss", torch.mean(edl_loss), sync_dist=True, on_epoch=True)
        # self.log("ecu_loss", torch.mean(ecu_loss), sync_dist=True, on_epoch=True)
        return loss_all, evidence, U

    def ce_loss(self, pred_batch=None, gt_batch=None):
        """
		Function: ce loss
		Args:
			pred_batch: doa
			gt_batch: dict{'doa'}
		Returns:
			loss
        """
        pred_doa = pred_batch
        gt_doa = gt_batch['doa'] * 180 / np.pi
        gt_doa = gt_doa[:,:,1,:].type(torch.LongTensor).to(self.device) # get the azimuth values [2, 24, 1]
        nb,nt,_ = pred_doa.shape
        pred_doa = pred_doa
        loss = torch.nn.functional.cross_entropy(pred_doa.reshape(nb*nt,-1),gt_doa.reshape(nb*nt))
        return loss


class MyCLI(LightningCLI):
    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor


        # parser.add_lightning_class_args(EarlyStopping, "early_stopping")
        # parser.set_defaults({
        #     "early_stopping.monitor": "valid/ACC",
        #     "early_stopping.min_delta": 0.0001,
        #     "early_stopping.patience": 15,
        #     "early_stopping.mode": "max",
        # })

        parser.add_lightning_class_args(ModelCheckpoint, "model_checkpoint")
        model_checkpoint_defaults = {
            "model_checkpoint.filename": "epoch{epoch}_valid_loss{valid/loss:.4f}",
            "model_checkpoint.monitor": "valid/loss",
            "model_checkpoint.mode": "min",
            "model_checkpoint.every_n_epochs": 1,
            "model_checkpoint.save_top_k": 5,
            "model_checkpoint.auto_insert_metric_name": False,
            "model_checkpoint.save_last": True,
            "model_checkpoint.dirpath": "./checkpoints/",
        }
        parser.set_defaults(model_checkpoint_defaults)

        parser.add_lightning_class_args(
            LearningRateMonitor, "learning_rate_monitor")
        learning_rate_monitor_defaults = {
            "learning_rate_monitor.logging_interval": "epoch",
        }
        parser.set_defaults(learning_rate_monitor_defaults)


if __name__ == '__main__':
    cli = MyCLI(
        TrustedRCNN,
        DataModule,
        seed_everything_default=1744,
        # save_config_callback=SaveConfigCallback,
        save_config_kwargs={'overwrite': True},
        # parser_kwargs={"parser_mode": "omegaconf"},

    )

    # debug for the model
    # model = TrustedRCNN()
    # trainer = l.Trainer(
    #     fast_dev_run=10,
    #     # devices="cpu",
    #     accelerator="cpu"
    # )


    # trainer.fit(model, datamodule=DataModule('/workspaces/tssl/data'))