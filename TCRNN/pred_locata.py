# Description: This file is used to predict the DOA of the sound source and draw the result.
import numpy as np
import torch
import time

from torch.utils.data import DataLoader

from dataloader.Dataset import Segmenting_SRPDNN
from dataloader.Dataset import LocataDataset
from utils.utils_predict_draw import locata_plot, pred_uncer

from main_crnn import TrustedRCNN as CRNN


def main():
    speed = 343.0
    fs = 16000
    T = 4.79 # Trajectory length (s)
    array_locata_name = 'dicit'
    win_len = 512
    win_shift_ratio = 0.5

    seg_fra_ratio = 12 # one estimate per segment (namely seg_fra_ratio frames)
    seg_len = int(win_len*win_shift_ratio*(seg_fra_ratio+1))
    seg_shift = int(win_len*win_shift_ratio*seg_fra_ratio)

    segmenting = Segmenting_SRPDNN(K=seg_len, step=seg_shift, window=None)

	# load the ckpt
    # !!! change the path of the ckpt file
    ckpt_path = '/workspaces/tssl/TCRNN/checkpoints/final_best.ckpt'
    net = CRNN.load_from_checkpoint(
        checkpoint_path=ckpt_path,
        map_location='cpu'
    )
    net.cuda()
    net.eval()


    dirs = {}
    # !
    dirs["sensig_locata"] = '/workspaces/tssl/data/LOCATA'
    array_locata_name = 'dicit' # select the array mode
    tasks = ((3,5), ) # select the task
    path_locata = (dirs['sensig_locata'] + '/eval',dirs['sensig_locata'] + '/dev') # set the path

    metric_setting = {}
    save_file = True
    for task in tasks:
        task_idx = tasks.index(task)
        t_start = time.time()
        dataset_locata = LocataDataset(path_locata,array_locata_name, fs, dev=True, tasks=task, transforms=[segmenting])
        dataloader = DataLoader(dataset=dataset_locata, batch_size=1, shuffle=False)

        metric = pred_uncer(
            dataloader=dataloader,
            model=net,
            metric_setting=metric_setting,
            save_file=save_file,
        )

        print(torch.mean(metric[0]['MAE']),torch.mean(metric[0]['ACC']))
        # !!! change the path of the result & Fig
        locata_plot(result_path='/workspaces/tssl/locata_results/gt_/', save_fig_path='/workspaces/tssl/locata_results/')

if __name__ == "__main__":
    main()
