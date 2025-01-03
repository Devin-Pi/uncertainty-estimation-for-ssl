import pickle
import os

import torch
from torch.utils.data import Dataset

import numpy as np
from numpy import ndarray
from scipy.signal import stft

import soundfile as sf
from dataloader.Dataset import AcousticScene
from utils.simu import Segmenting_SRPDNN
from loguru import logger


class TSSLDataSet(Dataset):
    def __init__(
        self,
        data_dir,
        num_data,
        return_acoustic_scene=False,
                 ):
        super().__init__()

        self.data_paths = []
        data_names = os.listdir(data_dir)
        for fname in data_names:
            front, ext = os.path.splitext(fname)
            if ext == ".wav":
                self.data_paths.append((os.path.join(data_dir, fname)))
        self.num_data = len(self.data_paths) if num_data is None else num_data
        self.gt_segmentation = Segmenting_SRPDNN(
            K=3328, # int(win_len512*win_shift_ratio0.5*(seg_fra_ratio12+1))
            step=3072, # int(win_len*win_shift_ratio*(seg_fra_ratio))
            window=None
        )
        self.acoustic_scene = AcousticScene(
            room_sz = [],
            T60 = [],
            beta = [],
            noise_signal = [],
            SNR = [],
            source_signal = [],
            fs = [],
            array_setup = [],
            mic_pos = [],
            timestamps = [],
            traj_pts = [],
            trajectory = [],
            t = [],
            DOA = [],
            c = [],
        )
        self.return_acoustic_scene = return_acoustic_scene
    def _get_audio_features(self,
                            audio_file: str,
                            ) -> ndarray:
        """Computes spectrogram audio features for a given chunk from an audio file.

        Args:
            audio_file (str): Path to audio file in *.wav format.
            start_time (float): Chunk start time in seconds.
            end_time (float): Chunk end time in seconds.

        Returns:
            ndarray: Spectrogram audio features.
        """
        file_info = sf.info(audio_file)
        audio_data, samp_freq = sf.read(audio_file)
        # Compute multi-channel STFT and remove first coefficient and last frame
        spectrogram = stft(audio_data,
                           fs=file_info.samplerate,
                           nperseg=512,
                           nfft=512,
                           padded=False,
                           axis=0)[-1] # [1025， 4， 101] 1025: the frequencies; 101: the times; 4: the channels
        spectrogram = spectrogram[1:, :, :-1] # [1024, 4, 100]
        spectrogram = spectrogram.transpose([1, 0, 2]) # [4, 100, 1024]
        spectrogram_real = np.real(spectrogram)
        spectrogram_img = np.imag(spectrogram)
        audio_features = np.concatenate((spectrogram_real, spectrogram_img),axis=0) # 4, 299, 256

        return audio_features.astype(np.float32)

    def _gt_acoustic_scene(self,
                           acous_path,):

        file = open(acous_path,'rb')
        dataPickle = file.read()
        file.close()
        self.acoustic_scene.__dict__ = pickle.loads(dataPickle)
        return self.acoustic_scene

    def __len__(self):
        return self.num_data
    def __getitem__(self, idx):

        audio_path = self.data_paths[idx]
        acous_path = audio_path.replace("wav", "npz")

        audio_feat = self._get_audio_features(audio_path)
        acous_scene = self._gt_acoustic_scene(acous_path)

        audio_feat_, acous_scene_ = self.gt_segmentation(
            audio_feat,
            acous_scene
        )

        vad_gt = acous_scene_.mic_vad_sources.mean(axis=1) # [24, 1]

        gts = {}
        gts["doa"] = acous_scene_.DOAw.astype(np.float32)
        gts["vad_sources"] = vad_gt.astype(np.float32)
        # logger.debug(f"vad sources shape: {vad_gt.shape}")
        return audio_feat_, gts
if __name__ == "__main__":
    data_dir = "/home/data/DCASE2021-task3-dev/foa_dev"
    dataset = TSSLDataSet(data_dir, num_data=1)
    for i in range(len(dataset)):
        audio_feat, gts = dataset[i]
        print(audio_feat.shape)
        print(gts["doa"].shape)
        print(gts["vad_sources"].shape)
        break