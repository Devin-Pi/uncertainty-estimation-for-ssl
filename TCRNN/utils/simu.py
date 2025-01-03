import torch
import numpy as np
import torch
import random
import pickle
import soundfile
import matplotlib.pyplot as plt
from loguru import logger

## for spherical coordinates
def forgetting_norm(input, sample_length=298):
        """
        Using the mean value of the near frames to normalization
        Args:
            input: feature
            sample_length: length of the training sample, used for calculating smooth factor
        Returns:
            normed feature
        Shapes:
            input: [B, C, F, T]
            sample_length_in_training: 192
        """
        assert input.ndim == 4
        batch_size, num_channels, num_freqs, num_frames = input.size()
        input = input.reshape(batch_size, num_channels * num_freqs, num_frames)

        eps = 1e-10
        mu = 0
        alpha = (sample_length - 1) / (sample_length + 1)

        mu_list = []
        for frame_idx in range(num_frames):
            if frame_idx < sample_length:
                alp = torch.min(torch.tensor([(frame_idx - 1) / (frame_idx + 1), alpha]))
                mu = alp * mu + (1 - alp) * torch.mean(
                    input[:, :, frame_idx], dim=1
                ).reshape(
                    batch_size, 1
                )  # [B, 1]
            else:
                current_frame_mu = torch.mean(input[:, :, frame_idx], dim=1).reshape(
                    batch_size, 1
                )  # [B, 1]
                mu = alpha * mu + (1 - alpha) * current_frame_mu

            mu_list.append(mu)

            # print("input", input[:, :, idx].min(), input[:, :, idx].max(), input[:, :, idx].mean())
            # print(f"alp {idx}: ", alp)
            # print(f"mu {idx}: {mu[128, 0]}")

        mu = torch.stack(mu_list, dim=-1)  # [B, 1, T298]
        #print(mu.shape)
        #output = input / (mu + eps)

        output = mu.reshape(batch_size, 1, 1, num_frames) # [1, 1, 1, 298]
        return output


def cart2sph(cart, include_r=False):
	""" Cartesian coordinates to spherical coordinates conversion.
	Each row contains one point in format (x, y, x) or (elevation, azimuth, radius),
	where the radius is optional according to the include_r argument.
	"""
	r = torch.sqrt(torch.sum(torch.pow(cart, 2), dim=-1))
	theta = torch.acos(cart[..., 2] / r)
	phi = torch.atan2(cart[..., 1], cart[..., 0])
	if include_r:
		sph = torch.stack((theta, phi, r), dim=-1)
	else:
		sph = torch.stack((theta, phi), dim=-1)
	return sph


def sph2cart(sph):
	""" Spherical coordinates to cartesian coordinates conversion.
	Each row contains one point in format (x, y, x) or (elevation, azimuth, radius),
	where the radius is supposed to be 1 if it is not included.
	"""
	if sph.shape[-1] == 2: sph = torch.cat((sph, torch.ones_like(sph[..., 0]).unsqueeze(-1)), dim=-1)
	x = sph[..., 2] * torch.sin(sph[..., 0]) * torch.cos(sph[..., 1])
	y = sph[..., 2] * torch.sin(sph[..., 0]) * torch.sin(sph[..., 1])
	z = sph[..., 2] * torch.cos(sph[..., 0])
	return torch.stack((x, y, z), dim=-1)


## for training process

def set_seed(seed):
	""" Function: fix random seed.
	"""
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.cuda.manual_seed(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.enabled = False # avoid-CUDNN_STATUS_NOT_SUPPORTED #(commont if use cpu??)

	np.random.seed(seed)
	random.seed(seed)

def set_random_seed(seed):

    np.random.seed(seed)
    random.seed(seed)

def get_learning_rate(optimizer):
    """ Function: get learning rates from optimizer
    """
    lr = []
    for param_group in optimizer.param_groups:
        lr += [param_group['lr']]
    return lr

def set_learning_rate(epoch, lr_init, step, gamma):
    """ Function: adjust learning rates
    """
    lr = lr_init*pow(gamma, int(epoch/step))
    return lr

## for data number

def detect_infnan(data, mode='torch'):
    """ Function: check whether there is inf/nan in the element of data or not
    """
    if mode == 'troch':
        inf_flag = torch.isinf(data)
        nan_flag = torch.isnan(data)
    elif mode == 'np':
        inf_flag = np.isinf(data)
        nan_flag = np.isnan(data)
    else:
        raise Exception('Detect infnan mode unrecognized')
    if (True in inf_flag):
        raise Exception('INF exists in data')
    if (True in nan_flag):
        raise Exception('NAN exists in data')


## for room acoustic data saving and reading

def save_file(mic_signal, acoustic_scene, sig_path, acous_path):

    if sig_path is not None:
        soundfile.write(sig_path, mic_signal, acoustic_scene.fs)

    if acous_path is not None:
        file = open(acous_path,'wb')
        file.write(pickle.dumps(acoustic_scene.__dict__))
        file.close()

def load_file(acoustic_scene, sig_path, acous_path):

    if sig_path is not None:
        mic_signal, fs = soundfile.read(sig_path)

    if acous_path is not None:
        file = open(acous_path,'rb')
        dataPickle = file.read()
        file.close()
        acoustic_scene.__dict__ = pickle.loads(dataPickle)

    if (sig_path is not None) & (acous_path is not None):
        return mic_signal, acoustic_scene
    elif (sig_path is not None) & (acous_path is None):
        return mic_signal
    elif (sig_path is None) & (acous_path is not None):
        return acoustic_scene

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
    plt.savefig(save_fig_path + 'locata_fig.jpg')


# def angular_error_2d(pred, true, doa_mode='azi'):
# 	""" 2D Angular distance between spherical coordinates.
# 	"""
# 	if doa_mode == 'azi':
# 		ae = torch.abs((pred-true+np.pi)%np.pi-np.pi)
# 	elif doa_mode == 'ele':
# 		ae = torch.abs(pred-true)

# 	return  ae

# def angular_error(the_pred, phi_pred, the_true, phi_true):
# 	""" Angular distance between spherical coordinates.
# 	"""
# 	aux = torch.cos(the_true) * torch.cos(the_pred) + \
# 		  torch.sin(the_true) * torch.sin(the_pred) * torch.cos(phi_true - phi_pred)

# 	return torch.acos(torch.clamp(aux, -0.99999, 0.99999))


# def mean_square_angular_error(y_pred, y_true):
# 	""" Mean square angular distance between spherical coordinates.
# 	Each row contains one point in format (elevation, azimuth).
# 	"""
# 	the_true = y_true[:, 0]
# 	phi_true = y_true[:, 1]
# 	the_pred = y_pred[:, 0]
# 	phi_pred = y_pred[:, 1]

# 	return torch.mean(torch.pow(angular_error(the_pred, phi_pred, the_true, phi_true), 2), -1)


# def rms_angular_error_deg(y_pred, y_true):
# 	""" Root mean square angular distance between spherical coordinates.
# 	Each input row contains one point in format (elevation, azimuth) in radians
# 	but the output is in degrees.
# 	"""

# 	return torch.sqrt(mean_square_angular_error(y_pred, y_true)) * 180 / pi
class Segmenting_SRPDNN(object):
	""" Segmenting transform.
	"""
	def __init__(self, K, step, window=None):
		self.K = K
		self.step = step
		if window is None:
			self.w = np.ones(K)
		elif callable(window):
			try: self.w = window(K)
			except: raise Exception('window must be a NumPy window function or a Numpy vector with length K')
		elif len(window) == K:
			self.w = window
		else:
			raise Exception('window must be a NumPy window function or a Numpy vector with length K')

	def __call__(self, x, acoustic_scene):
		# N_mics = x.shape[1]
		N_dims = acoustic_scene.DOA.shape[1]
		num_source = acoustic_scene.DOA.shape[2]
		L = acoustic_scene.source_signal.shape[0]
		N_w = np.floor(L/self.step - self.K/self.step + 1).astype(int)
		# logger.info('Number of segments: {}'.format(N_w))
		# logger.info('Number of samples: {}'.format(L))
		if self.K > L:
			raise Exception('The window size can not be larger than the signal length ({})'.format(L))
		elif self.step > L:
			raise Exception('The window step can not be larger than the signal length ({})'.format(L))

		DOA = []
		for source_idx in range(num_source):
			DOA += [np.append(acoustic_scene.DOA[:,:,source_idx], np.tile(acoustic_scene.DOA[-1,:,source_idx].reshape((1,2)),
				[N_w*self.step+self.K-L, 1]), axis=0)] # Replicate the last known DOA

		# logger.debug('DOA shape: {}'.format(DOA[0].shape))
		DOA = np.array(DOA).transpose(1,2,0)
		# logger.debug('DOA shape: {}'.format(DOA.shape))
		shape_DOAw = (N_w, self.K, N_dims) # (nwindow, win_len, naziele)
		# logger.debug('Shape DOAw: {}'.format(shape_DOAw))
		strides_DOAw = [self.step*N_dims, N_dims, 1]
		# logger.debug('Strides DOAw: {}'.format(strides_DOAw))
		strides_DOAw = [strides_DOAw[i] * DOA.itemsize for i in range(3)]
		# logger.debug('DOA itemsize: {}'.format(DOA.itemsize))
		# logger.debug('Strides DOAw: {}'.format(strides_DOAw))
		acoustic_scene.DOAw = []
		for source_idx in range(num_source):
			DOAw = np.lib.stride_tricks.as_strided(DOA[:,:,source_idx], shape=shape_DOAw, strides=strides_DOAw)
			DOAw = np.ascontiguousarray(DOAw)
			# logger.debug('DOAw shape: {}'.format(DOAw[..., 1].shape))
			for i in np.flatnonzero(np.abs(np.diff(DOAw[..., 1], axis=1)).max(axis=1) > np.pi):
				DOAw[i, DOAw[i,:,1]<0, 1] += 2*np.pi # Avoid jumping from -pi to pi in a window
			DOAw = np.mean(DOAw, axis=1)
			DOAw[DOAw[:,1]>np.pi, 1] -= 2*np.pi
			acoustic_scene.DOAw += [DOAw]
			# logger.debug('DOAw shape: {}'.format(DOAw.shape))
		acoustic_scene.DOAw = np.array(acoustic_scene.DOAw).transpose(1, 2, 0)# (nsegment,naziele,nsource)
		# logger.debug('DOAw shape: {}'.format(acoustic_scene.DOAw.shape))

		# Pad and window the VAD if it exists
		if hasattr(acoustic_scene, 'mic_vad'): # (nsample,1)
			vad = acoustic_scene.mic_vad[:, np.newaxis]
			vad = np.append(vad, np.zeros((L - vad.shape[0], 1)), axis=0)

			shape_vadw = (N_w, self.K, 1)
			strides_vadw = [self.step * 1, 1, 1]
			strides_vadw = [strides_vadw[i] * vad.itemsize for i in range(3)]

			acoustic_scene.mic_vad = np.lib.stride_tricks.as_strided(vad, shape=shape_vadw, strides=strides_vadw)[..., 0]

		# Pad and window the VAD if it exists
		if hasattr(acoustic_scene, 'mic_vad_sources'): # (nsample,nsource)
			shape_vadw = (N_w, self.K, 1)

			num_sources = acoustic_scene.mic_vad_sources.shape[1]
			vad_sources = []
			for source_idx in range(num_sources):
				vad = acoustic_scene.mic_vad_sources[:, source_idx:source_idx+1]
				vad = np.append(vad, np.zeros((L - vad.shape[0], 1)), axis=0)

				strides_vadw = [self.step * 1, 1, 1]
				strides_vadw = [strides_vadw[i] * vad.itemsize for i in range(3)]
				vad_sources += [np.lib.stride_tricks.as_strided(vad, shape=shape_vadw, strides=strides_vadw)[..., 0]]

			acoustic_scene.mic_vad_sources = np.array(vad_sources).transpose(1,2,0) # (nsegment, nsample, nsource)
			# logger.debug('VAD shape: {}'.format(acoustic_scene.mic_vad_sources.shape))
		# Timestamp for each window
		acoustic_scene.tw = np.arange(0, (L-self.K), self.step) / acoustic_scene.fs

		return x, acoustic_scene