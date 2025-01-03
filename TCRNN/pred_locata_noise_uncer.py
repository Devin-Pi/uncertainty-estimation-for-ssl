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

def draw_overall():
    plt.figure(figsize=(16,8),dpi=300)
    for k in range(12):   
        doa_gt = np.load('/workspaces/tssl/locata_gt/'+str(k)+'_gt.npy')
        doa_est = np.load('/workspaces/tssl/locata_gt/'+str(k)+'_est.npy')
        vad_gt = np.load('/workspaces/tssl/locata_gt/'+str(k)+'_vadgt.npy')
        vad_gt[vad_gt<2/3] = -1
        vad_gt[vad_gt>2/3] = 1
        for i in range(1):
            plt.subplot(3,4,k+1)
            plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                    wspace=0.3, hspace=0.3)
            
            doa_est = doa_est[:,:doa_gt.shape[1],:,:]

            x_est = [j*4096/16000 for j in range(doa_est.shape[1])]
            x_gt = [j*4096/16000 for j in range(doa_gt.shape[1])]
            plt.scatter(x_gt,doa_gt[i,:,1,0],s=5,c='grey',linewidth=0.8,label='GT')
            # plt.scatter(x,doa_est[i,:,1,0]*vad_gt[i,:,0],s=3,c='firebrick',linewidth=0.8,label='EST')
            plt.scatter(x_est,doa_est[i,:,1,0]*vad_gt[i,:,0],s=3,c='firebrick',linewidth=0.8,label='EST')
            plt.xlabel('Time [s]')
            plt.ylabel('DOA[°]')
            plt.ylim((0,180))
            plt.grid()
            plt.legend(loc=0,prop={'size': 4})
    plt.savefig('/workspaces/tssl/locata_results/locata_fig_overall_snr_4.jpg') #!!! change it 


def locata_plot(i, 
                result_path,
                save_fig_path,
                gt_file,
                vadgt_file,
                bias=40
                ):
    print(i)
    plt.figure(figsize=(16,8),dpi=300)
    doa_est = np.load(result_path+str(i)+'_est.npy') # -bias
    doa_gt = np.load(gt_file)
    vad_gt = np.load(vadgt_file)
    vad_gt[vad_gt<2/3] = -1
    vad_gt[vad_gt>2/3] = 1    
    for p in range(1): 
        plt.subplot(1,1,1)
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                    wspace=0.3, hspace=0.3)

        doa_est = doa_est[:,:doa_gt.shape[1],:,:]

        x_est = [j*4096/16000 for j in range(doa_est.shape[1])]
        x_gt = [j*4096/16000 for j in range(doa_gt.shape[1])]
        plt.scatter(x_gt,doa_gt[p,:,1,0],s=5,c='grey',linewidth=0.8,label='GT')
        # plt.scatter(x,doa_est[i,:,1,0]*vad_gt[i,:,0],s=3,c='firebrick',linewidth=0.8,label='EST')
        plt.scatter(x_est,doa_est[p,:,1,0]*vad_gt[p,:,0],s=3,c='firebrick',linewidth=0.8,label='EST')
        plt.xlabel('Time [s]')
        plt.ylabel('DOA[°]')
        plt.ylim((0,180))
        plt.grid()
        plt.legend(loc=0,prop={'size': 4})
    plt.savefig(save_fig_path + str(i)+  '.jpg')
      
     
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

    # doa_gt = doa_gt * 180 / np.pi
    # doa_est = doa_est * 180 / np.pi
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
        # MAE += [torch.sum(corr_flag * azi_error) / torch.sum(act_flag)]
    # if 'aziele' in ae_mode:
    #     MAE += [torch.sum(vad_gt * aziele_error) / torch.sum(act_flag)]

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
    with open("/workspaces/tssl/locata_results_uncer/snr_-4.txt", "a+") as f: # !!! change the path of the txt file
        np.savetxt(f, U, delimiter="\n", fmt='%f')
      
    

def pred(i, audio_file):
    
    gt_file_path = "/workspaces/tssl/locata_gt/" #!!! change the path of the gt file path
    gt_file = gt_file_path + str(i) + '_gt.npy' # obtain gt file
    vadgt_file = gt_file_path + str(i) + '_vadgt.npy' # obtain vadgt file
    doa_gt = np.load(gt_file)
    vad_gt = np.load(vadgt_file)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load the pretrained model & eval
    # ckpt_path = '/workspaces/tlstm/ckpt/SPR-DNN.ckpt'
    ckpt_path = '/workspaces/tssl/ckpt/final_best.ckpt' #!!! change the path of the checkpoint file
    net = CRNN.load_from_checkpoint(
        checkpoint_path=ckpt_path,
        map_location='cpu'
    )
    net.cuda()
    net.eval()
    
    # load the audio data
    audio_data_, fs = sf.read(audio_file) #[1061282, 15]
    # For LOCATA DICIT ARRAY
    audio_data_ = np.concatenate((audio_data_[:,8:9], audio_data_[:,5:6]), axis=-1) # [ 1061282, 2]
    
    # chanege the sampling rate to 16000
    if fs > 16000:
        audio_data_ = scipy.signal.decimate(audio_data_, int(fs/16000), axis=0)
        new_fs = fs / int(fs/16000)
        if new_fs != 16000: warnings.warn('The actual fs is {}Hz'.format(new_fs))
        fs = new_fs
    elif fs < 16000:
        raise Exception('The sampling rate of the file ({}Hz) was lower than self.fs ({}Hz'.format(fs, 16000))

    # Compute multi-channel STFT and remove first coefficient and last frame
    spectrogram = stft(
        audio_data_,
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
    mic_sig_batch = torch.unsqueeze(mic_sig_batch, dim=0).to(torch.float).to(device) # [1, 4, 256, 1381]   

    # get the pred result of the LOCATA
    pred_batch = net(mic_sig_batch)  
    uncertainty_calu(pred_batch)
    pred_batch = pred_batch.detach()
    DOA_batch_pred = torch.argmax(
        pred_batch,
        dim = -1,
    )
    pred_batch = {}
    pred_batch["doa"] = DOA_batch_pred[:, :, np.newaxis, np.newaxis]
    doa_pred = pred_batch["doa"]
    doa_pred = torch.cat((doa_pred, doa_pred), dim=-2)
    nbatch, nt, naziele, nsources = pred_batch['doa'].shape
    pred_batch['vad_sources'] = torch.ones((nbatch, nt, nsources))
    vad_est = pred_batch["vad_sources"]
    # Draw the figure for the pred result
    save_path = '/workspaces/tssl/locata_gt/' #!!! change the path of the save path
    np.save(save_path+str(i)+'_est',doa_pred.cpu().numpy())

    # locata_plot(
    #     i,
    #     result_path='/workspaces/tssl/locata_gt/', #!!! gt & est save path
    #     save_fig_path='/workspaces/tssl/locata_results/', #!!! audio file & figure save path
    #     gt_file = gt_file,
    #     vadgt_file = vadgt_file,
    #     )
    
    metrics = calu_metr(
        doa_gt = doa_gt,
        vad_gt = vad_gt,
        doa_est = doa_pred,
        vad_est  = vad_est,
        useVAD = True,
        vad_TH = [2/3,2/3],
        ae_mode = 'azi',
        ae_TH = 15,
    )
    print(torch.mean(metrics['MAE']), torch.mean(metrics['ACC']))

    
    ############# Figure for Field Experiment #############
    
def main():
    data_paths = []
    dataset_path = "/workspaces/tssl/data/locata_uncer/snr_-4/" #!!! change the path of the dataset
    
    data_names = os.listdir(dataset_path)
    for fname in data_names:
        front, ext = os.path.splitext(fname)
        if ext == ".wav":
            data_paths.append((os.path.join(dataset_path, fname)))
    data_paths.sort()
    audio_file_directory = data_paths
    total_number = len(audio_file_directory)
    print(audio_file_directory)
    for i in range(total_number):
        pred(i, audio_file_directory[i])
    
    draw_overall()
    

if __name__ == "__main__":
    main()
	
