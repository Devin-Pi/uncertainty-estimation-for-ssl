import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import stft

import torch
import torch.nn.functional as F
from copy import deepcopy
from model.module import PredDOA
from loguru import logger
def locata_plot(result_path, save_fig_path, bias=4):
    plt.figure(figsize=(16,8),dpi=300)
    for k in range(12):
        doa_gt = np.load(result_path+str(k)+'_gt.npy')
        doa_est = np.load(result_path+str(k)+'_est.npy')-bias
        vad_gt = np.load(result_path+str(k)+'_vadgt.npy')
        vad_gt[vad_gt<2/3] = -1
        vad_gt[vad_gt>2/3] = 1
        for i in range(1):
            plt.subplot(3,4,k+1)
            plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                    wspace=0.3, hspace=0.3)
            x = [j*4096/16000 for j in range(doa_gt.shape[1])]
            plt.scatter(x,doa_gt[i,:,1,0],s=5,c='grey',linewidth=0.8,label='GT')
            plt.scatter(x,doa_est[i,:,1,0]*vad_gt[i,:,0],s=3,c='firebrick',linewidth=0.8,label='EST')
            #plt.scatter(x,doa_est[i,:,1,0],s=3,c='firebrick',linewidth=0.8,label='EST')
            plt.xlabel('Time [s]')
            plt.ylabel('DOA[°]')
            plt.ylim((0,180))
            plt.grid()
            plt.legend(loc=0,prop={'size': 4})
    plt.savefig(save_fig_path + 'locata_fig_new.jpg')

def pred_uncer(model, dataloader, metric_setting, save_file, return_predgt=False,dataset_name='locata'):

    data = []
    model.eval()
    get_metric = PredDOA()
    with torch.no_grad():
        idx = 0
        pred = []
        gt = []
        mic_sig = []
        if metric_setting is not None:
            metric = {}
        for mic_sig_batch, gt_batch in dataloader:
            # logger.debug(f"mic_sig_batch.shape: {mic_sig_batch.shape}")
            # logger.debug(gt_batch)
            print('Dataloading: ' + str(idx+1))
            if dataset_name == 'locata':
                # logger.debug("I am locata!")
                print(mic_sig_batch.shape) # [bs, 312197, 4]
                mic_sig_batch = torch.cat((mic_sig_batch[:,:,8:9], mic_sig_batch[:,:,5:6]), axis=-1)[0] # [1, 1061282, 2]
                # logger.debug(f"mic_sig_batch.shape: {mic_sig_batch.shape}")
                spectrogram = stft(
        	        mic_sig_batch,
        	        fs=16000,
        	        nperseg=512,
        	        nfft=512,
        	        padded=False,
        	        axis=0)[-1] # [f， c， t] f: the frequencies; t: the times; c: the channels as for locata: 257(f), 2(c), 1382(t)
                spectrogram = spectrogram[1:, :, :-1] # locata:[256, 2, 1381] simu:256, 2, 299
                spectrogram = spectrogram.transpose([1, 0, 2]) # [2, 256, 1381] c, f, t
                # convert the real and img part to the audio features
                spectrogram_real = np.real(spectrogram)
                spectrogram_img = np.imag(spectrogram)
                audio_feat_ = np.concatenate((spectrogram_real, spectrogram_img),axis=0) # 4, 256, 1381 c, f, t

                # transform the audio features to the 'batch_tensor'
                audio_feat_.astype(np.float32)
                mic_sig_batch = torch.from_numpy(audio_feat_)
                mic_sig_batch = torch.unsqueeze(mic_sig_batch, dim=0).to(torch.float).to('cuda')
                # [1, 4, 256, 1381]
                # get the pred result of the LOCATA
                pred_batch = model(mic_sig_batch) # [1, 121, 180]

                vad_batch = gt_batch["vad_sources"]
                vad_batch = vad_batch.mean(axis=2).float().to(pred_batch.device)
                gt_batch["doa"] = gt_batch["doa"].to(pred_batch.device)
                gt_batch["vad_sources"] = vad_batch
            else:
                logger.debug(dataset_name)
                pred_batch = model(mic_sig_batch.to('cuda'))
                vad_batch = gt_batch["vad_sources"]
                vad_batch = vad_batch.mean(axis=2).float().to(pred_batch.device)
                gt_batch["doa"] = gt_batch["doa"].to(pred_batch.device)
                gt_batch["vad_sources"] = vad_batch
                logger.debug(f"pred_batch.shape: {pred_batch.shape}")
            metrics = get_metric(pred_batch, gt_batch, save_file=True, idx=idx)
            # logger.debug(f"metrics: {metrics}")
            if metric_setting is not None:
                for m in metrics.keys():
                    if idx==0:
                        metric[m] = deepcopy(metrics[m])
                    else:
                        metric[m] = torch.cat((metric[m], metrics[m]), axis=0)
            idx = idx+1

        if return_predgt:
            data += [pred, gt]
            data += [mic_sig]
        if metric_setting is not None:
            data += [metric]
        return data