import os
import warnings
import numpy as np
import torch
import scipy.io
import numpy as np
from scipy.signal import stft
import soundfile as sf
import torch.nn.functional as F
import matplotlib.pyplot as plt
from copy import deepcopy
from main_crnn import TrustedRCNN as CRNN

from dataloader.dataset_cls import TSSLDataSet
from torch.utils.data import DataLoader
from model.module import PredDOA

from loguru import logger



def angular_error( est, gt, ae_mode):
    """
    Function: return angular error in degrees
    """
    est = est.cpu()

    if ae_mode == 'azi':
        ae = torch.abs((est-gt+180)%360 - 180)
    elif ae_mode == 'ele':
        ae = torch.abs(est-gt)
    elif ae_mode == 'aziele':
        ele_gt = gt[0, ...].float() / 180 * np.pi
        azi_gt = gt[1, ...].float() / 180 * np.pi
        ele_est = est[0, ...].float() / 180 * np.pi
        azi_est = est[1, ...].float() / 180 * np.pi
        aux = torch.cos(ele_gt) * torch.cos(ele_est) + torch.sin(ele_gt) * torch.sin(ele_est) * torch.cos(azi_gt - azi_est)
        aux[aux.gt(0.99999)] = 0.99999
        aux[aux.lt(-0.99999)] = -0.99999
        ae = torch.abs(torch.acos(aux)) * 180 / np.pi
    else:
        raise Exception('Angle error mode unrecognized')
    return ae

def calu_metr(doa_gt,
              vad_gt,
              doa_est,
              vad_est,
              useVAD,
              vad_TH,
              ae_mode,
              ae_TH
              ):


    doa_est = doa_est[:, :doa_gt.shape[1], :]
    vad_est = vad_est[:, :vad_gt.shape[1], :]

    nbatch, nt, naziele, nsources = doa_est.shape
    if useVAD == False:
        vad_gt = torch.ones((nbatch, nt, nsources))
        vad_est = torch.ones((nbatch,nt, nsources))
    else:
        vad_gt = vad_gt > vad_TH[0] #  the VAD threshold
        vad_est = vad_est > vad_TH[1]

    vad_est = vad_est * vad_gt
    vad_gt_ = torch.from_numpy(vad_gt)
    azi_error = angular_error(doa_est[:,:,1,:], doa_gt[:,:,1,:], 'azi')
    ele_error = angular_error(doa_est[:,:,0,:], doa_gt[:,:,0,:], 'ele')
    # aziele_error = angular_error(doa_est.permute(2,0,1,3), doa_gt.permute(2,0,1,3), 'aziele')

    corr_flag = ((azi_error < ae_TH)+0.0) * vad_est # Accorrding to azimuth error
    act_flag = 1*vad_gt
    K_corr = torch.sum(corr_flag)
    # corr_flag_ = torch.from_numpy(corr_flag)
    act_flag_ = torch.from_numpy(act_flag)
    ACC = torch.sum(corr_flag) / torch.sum(act_flag_)
    MAE = []
    if 'ele' in ae_mode:
        MAE += [torch.sum(vad_gt_ * ele_error) / torch.sum(act_flag_)]
    if 'azi' in ae_mode:
        MAE += [torch.sum(vad_gt_ * azi_error) / torch.sum(act_flag_)]

    MAE = torch.tensor(MAE)
    metric = {}
    metric['ACC'] = torch.tensor([ACC])
    metric['MAE'] = MAE
    # metric = [ACC, MAE]

    return metric

def uncertainty_calu(pred_batch):
    nb, nt, _ = pred_batch.shape
    pred_batch = pred_batch.reshape(nb*nt, -1)
    evidence = F.softplus(pred_batch)
    # evidence = torch.exp(torch.clamp(pred_batch, -10, 10))
    alpha = evidence + 1
    S = torch.sum(alpha, dim=1, keepdim=True)
    U = 180 / S
    # evidence_scores, evidence_cls = torch.max(evidence, dim=1)
    evidence_cls = torch.argmax(evidence, dim=1)
    U = U.detach().cpu().numpy()
    with open("/workspaces/tssl/uncertainty_test/softplus.txt", "a+") as f: #!!! change the path of the txt file
        np.savetxt(f, U, delimiter="\n", fmt='%f')
    # with open("/TSSL/locata_result_uncertainty/pred_batch.txt", "w") as f:
    #     f.write(str(pred_batch.detach().cpu().numpy()) + "\n")
    # with open("/TSSL/locata_result_uncertainty/evidence.txt", "w") as f:
    #     f.write(str(evidence.detach().cpu().numpy()) + "\n")
    # # with open("/TSSL/locata_result_uncertainty/evidence_scores.txt", "w") as f:
    # #     f.write(str(evidence_scores.detach().cpu().numpy()) + "\n")
    # with open("/TSSL/locata_result_uncertainty/evidence_cls.txt", "w") as f:
    #     f.write(str(evidence_cls.detach().cpu().numpy()) + "\n")



def pred(idx, data):
    get_metric = PredDOA()
    mic_sig_batch = data[0]
    gt_batch = data[1]


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gt_batch["doa"] = gt_batch["doa"].to(device)
    gt_batch["vad_sources"] = gt_batch["vad_sources"].to(device)
    # load the pretrained model & eval
    # ckpt_path = '/workspaces/tlstm/ckpt/SPR-DNN.ckpt'
    ckpt_path = '/workspaces/tssl/ckpt/tcrnn/final_best.ckpt' #!!! change the path of the ckpt file
    net = CRNN.load_from_checkpoint(
        checkpoint_path=ckpt_path,
        map_location='cpu'
    )
    net.cuda()
    net.eval()
    # get the pred result of the LOCATA
    pred_batch = net(mic_sig_batch.to(device))
    uncertainty_calu(pred_batch)

    pred_batch = pred_batch.detach()
    metrics = get_metric(pred_batch, gt_batch, save_file=True, idx=idx)
    # for m in metrics:
    #     print(m, metrics[m].item())
        # logger.debug(f"{m}: {metrics[m].item()}")
        # self.log('test/'+m, metric[m].item(), sync_dist=True, on_epoch=True)
    return metrics
    # locata_plot(
    #     i,
    #     result_path='/workspaces/tssl/result_gt/', # gt & est save path
    #     save_fig_path='/workspaces/tlstm_1/pred_draw/', # audio file & figure save path
    #     gt_file = gt_file,
    #     vadgt_file = vadgt_file,
    #     )



    ############# Figure for Field Experiment #############

def main():
    data_paths = []
    acc = []
    mae = []
    # dataset_path = "/workspaces/tssl//snr_15/" #!!! change the path of the dataset
    dataset_path = "/workspaces/tssl/data/test/" #!!! change the path of the dataset
    print("dataset_path: ", dataset_path)
    dataset_test = TSSLDataSet(
        data_dir=dataset_path,
        num_data=5000,
        return_acoustic_scene=False,
    )

    data_loader = DataLoader(
        dataset=dataset_test,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    for idx, data in enumerate(data_loader):
        # logger.debug(f"idx: {idx}")
        print("idx: ", idx)
        metrics = pred(idx, data)
        acc.append(metrics["ACC"].item())
        mae.append(metrics["MAE"].item())
    print("ACC: ", np.mean(acc))
    print("MAE: ", np.mean(mae))

    # draw_overall()


if __name__ == "__main__":
    main()
