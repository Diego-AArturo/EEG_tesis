import torch
from audio.models import AudioClassifier
from eeg.model import CNNTransformer
from eeg.caueeg.datasets.caueeg_dataset import *
from eeg.caueeg.datasets.caueeg_script import *
from eeg.caueeg.datasets.pipeline import *


def load_audio_model(path,device): 
    model_audio = AudioClassifier()
    model_audio.load_state_dict(torch.load(path, map_location=device)) #CHECKPOINT DE VOZ
    model_audio.eval()
    return model_audio

def load_eeg_model(path,device,cfg):
    crop_length = 300 * 10
    n_fft, hop_length, seq_len_2d = calculate_stft_params(seq_length=crop_length, verbose=True)
    ckpt = torch.load(path, map_location=device, weights_only=False)
    state_dict = ckpt["model_state"]
    cfg_tr = ckpt["config"]  # ← esta es la config usada para entrenar
    # print('config: ',config)
    model_eeg = CNNTransformer(
        in_channels=cfg_tr["in_channels"],
        out_dims=cfg["out_dims"], 
        seq_length=3000,
        fc_stages=2,
        use_age="no",
        seq_len_2d=seq_len_2d  # ya que eso fue lo que se usó
    ).to(device)


    model_eeg.load_state_dict(state_dict)
    model_eeg.eval()
    return model_eeg


