import os
# from utils import set_seed,save_file,load_file
from utils.simu import save_file
from dataloader.Dataset import Parameter
import dataloader.Dataset as at_dataset
import tqdm
import hydra
from omegaconf import DictConfig
import lightning as l

@hydra.main(config_path="config/", config_name="data_simu", version_base=None)
def main(cfg: DictConfig) -> None:

    if cfg.DATA_SIMU.TRAIN:
        data_num = cfg.DATA_SIMU.TRAIN_NUM
        l.seed_everything(1848)
        sou_path = cfg.DATA_SIMU.SOU_PATH.TRAIN
        exp_path = cfg.DATA_SIMU.EXP_PATH.TRAIN
    elif cfg.DATA_SIMU.TEST:
        data_num = cfg.DATA_SIMU.TEST_NUM
        l.seed_everything(1858)
        sou_path = cfg.DATA_SIMU.SOU_PATH.TEST
        exp_path = cfg.DATA_SIMU.EXP_PATH.TEST
    else:
        data_num = cfg.DATA_SIMU.DEV_NUM
        l.seed_everything(1868)
        sou_path = cfg.DATA_SIMU.SOU_PATH.DEV
        exp_path = cfg.DATA_SIMU.EXP_PATH.DEV

    SPEED = 343.0
    FS = 16000
    T = 4.79 # Trajectory length (s)
    TRAJ_POINTS = 50 # number of RIRs per trajectory
    ARRAY_SETUP = at_dataset.dualch_array_setup # the settings of the microphone arrays
    # ARRAY_SETUP = at_dataset.MIC4_array_setup
    # ARRAY_SETUP = at_dataset.MIC8_array_setup
    # ARRAY_SETUP = at_dataset.MIC4_array_setup_small
    # SOURCE SIGNAL
    sourceDataset = at_dataset.LibriSpeechDataset(
        path = sou_path,
        T = T,
        fs=FS,
        num_source = max(cfg.BASIC_SETTING.NOS),
	    return_vad = True,
	    clean_silence = True
    )
    # NOISE SIGNAL
    noiseDataset = at_dataset.NoiseDataset(
        T = T,
        fs = FS,
        nmic = ARRAY_SETUP.mic_pos.shape[0],
        noise_type = Parameter(['diffuse'], discrete=True),
        noise_path = cfg.DATA_SIMU.NOISE_PATH,
        c = SPEED
    )

    # generate the data for the small room:
    dataset = at_dataset.RandomTrajectoryDataset(
        sourceDataset = sourceDataset,
        num_source = Parameter(cfg.BASIC_SETTING.NOS, discrete=True), # Random number of sources from list-args.sources
        source_state = cfg.BASIC_SETTING.S_STATE,
        room_sz = Parameter([6,6,2.5], [10,8,6]),   	# Random room sizes from 3x3x2.5 to 10x8x6 meters
        T60 = Parameter(0.2, 1.3),					# Random reverberation times from 0.2 to 1.3 seconds
        abs_weights = Parameter([0.1]*6, [0.2]*6),  # Random absorption weights ratios between walls
        array_setup = ARRAY_SETUP,
        array_pos = Parameter([0.1,0.1,0.3], [0.9,0.2,0.5]), # Ensure a minimum separation between the array and the walls
        noiseDataset = noiseDataset,
        SNR = Parameter(-5,15), 	# Start the simulation with a low level of omnidirectional noise
        nb_points = TRAJ_POINTS,	# Simulate RIRs per trajectory
        min_dis = Parameter(0.5, 2),
        c = SPEED,
        transforms = []
    )

    save_dir = exp_path
    exist_temp = os.path.exists(save_dir)
    if exist_temp==False:
        os.makedirs(save_dir)
        print('make dir: ' + save_dir)
    print(data_num)
    for idx in tqdm.tqdm(range(data_num)):
        mic_signals, acoustic_scene = dataset[idx]
        sig_path = save_dir + '/' + str(idx) + '.wav'
        acous_path = save_dir + '/' + str(idx) + '.npz'
        save_file(mic_signals, acoustic_scene, sig_path, acous_path)

if __name__ == "__main__":
    main()
