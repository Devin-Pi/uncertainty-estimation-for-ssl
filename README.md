# Uncertainty Estimation for Sound Source Localization with Deep Learning
This repository contains the python implementation for the paper  "Uncertainty Estimation for Sound Source Localization with Deep Learning".
![arcchitecture](/asset/archi.png)

## Dataset
- Source signals: [LibriSpeech](https://www.openslr.org/12/)
- Noise signals: [Noise92X](http://spib.linse.ufsc.br/noise.html)
- The real-world dataset: [LOCATA](https://www.locata.lms.tf.fau.de/datasets/)

These datasets mentioned above can be downloaded from this [OneDrive link](https://connectpolyu-my.sharepoint.com/:f:/g/personal/22123553r_connect_polyu_hk/EgHVOLP0P8VMvVoZ5DNWYCkBCUWYkaa93QJGnw-Glx4Qlw?e=Zs8iOB).

The data directory structure is shown as follows:

```
.
|---data
    |---LibriSpeech
        |---dev-clean
        |---test-clean
        |---train-clean-100
    |---NoiSig
    |---test
    |---train
    |---dev
```
**Note**: The `data/` file does not have to be within your project, you can put it somewhere you want. Please remembet to fill the correct data path in `config/tcrnn.yaml`.

## Get Started
### Dependencies
We strongly recommend that you can use [VSCode](https://code.visualstudio.com/) and [Docker](https://www.docker.com/) for this project, it can save you much time😁! Note that the related configurations has already been within `.devcontainer`. The detail information can be found in this [Tutorial_for_Vscode&Dokcer](https://github.com/Devin-Pi/Tutorial_for_VScode_Docker).

The environment:
- cuda:11.8.0
- cudnn: 8
- python: 3.10
- pytorch: 2.2.0
- pytorch lightning: 2.2
### Configurations
The realted configurations are all saved in `config/`.
- The `data_simu.yaml` is used to configure the data generation.
- The `tcrnn.yaml` is used to configure the dataloader, model training & test.

You can change the value of these items based on your need.

Note: Do not forget to intall [gpuRIR](https://github.com/DavidDiazGuerra/gpuRIR) and [webrtcvad](https://github.com/wiseman/py-webrtcvad).

### 🚀 Quick Start
- **Inference**
We provide the checkpoint to help you reproduce the results represented in the paper. [ckpt download](https://drive.google.com/drive/folders/1Yn3kfe-RG_rK7jjXyd8BU5dMEddRTzW0?usp=sharing)

- **Data Generation**

Generate the training data:
```zsh
python data_simu.py DATA_SIMU.TRAIN=True DATA_SIMU.TRAIN_NUM=10000
```
In the same way, you can also generate the validation and test datasets by changing the `DATA_SIMU.TRAIN=True` to `DATA_SIMU.DEV=True` or `DATA_SIMU.TEST=True`.
- **Model Training**
```zsh
python main_crnn.py fit --config /workspaces/TCRNN/config/tcrnn.yaml
```
The parameter for `--config` should point to your config file path.
- **Model Evaluation**
1) Change the `ckpt_path` in the `config/tcrnn.yaml` to the trained model weight.
2) Use Multiple GPUs or Single GPU to test the model performance.
```zsh
python main_crnn.py test --config /workspaces/TCRNN/config/tcrnn.yaml
```
If you want to evaluate the model using the Single GPU, you can change the value of the `devices` from `"0,1"` to `"0,"` in the `config/tcrnn.yaml`.

## 🎓 Citation
If you find our work useful in your research, please consider citing:
```
@article{pi2025uncertainty,
  author={Pi, Rendong and Yu, Xiang},
  journal={IEEE Transactions on Instrumentation and Measurement}, 
  title={Uncertainty Estimation for Sound Source Localization With Deep Learning}, 
  year={2025},
  volume={74},
  number={},
  pages={1-12},
  doi={10.1109/TIM.2024.3522632}
}


@inproceedings{pi2024tssl,
  title={TSSL: Trusted Sound Source Localization},
  author={Pi, Rendong and Song, Yang and Li, Linfeng and Yu, Xiang and Cheng, Li},
  booktitle={INTER-NOISE and NOISE-CON Congress and Conference Proceedings},
  volume={270},
  number={11},
  pages={941--949},
  year={2024},
  organization={Institute of Noise Control Engineering}
}
```
## Acknowledge
This repository adapts and integrates from some wonderful works, shown as follows:

- [SRP-DNN](https://github.com/BingYang-20/SRP-DNN?tab=readme-ov-file)
- [FN-SSL](https://github.com/Audio-WestlakeU/FN-SSL)
- [Cross3D](https://github.com/DavidDiazGuerra/Cross3D)
