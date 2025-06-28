# fusion_project/io_preprocess.py
from pathlib import Path
import torch, torchaudio
import numpy as np
import pyedflib

# ----- AUDIO --------------------------------------------------------- #
def load_audio_file(path_wav: str | Path,
                    target_sr: int = 16_000,
                    mean: float = 0.0,
                    std:  float = 1.0) -> torch.Tensor:
    """
    Devuelve espectrograma log-mel + delta: shape (2, 64, T).
    Aplica misma normalización que el dataset de entrenamiento.
    """
    path_wav = Path(path_wav)
    sig, sr_orig = torchaudio.load(path_wav)     # (1, N)

    if sr_orig != target_sr:
        sig = torchaudio.functional.resample(sig, sr_orig, target_sr)

    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=target_sr, n_fft=1024, hop_length=256,
        n_mels=64, power=2.0
    )(sig)                                       # (1, 64, T)

    mel_db = torchaudio.functional.amplitude_to_DB(mel)
    delta  = torchaudio.functional.compute_deltas(mel_db)

    spec = torch.cat([mel_db, delta], dim=0)     # (2, 64, T)
    spec = (spec - mean) / std                   # misma normalización
    return spec.float()


# ----- EEG ----------------------------------------------------------- #
def load_eeg_file_pyedf(edf_path: str | Path,
                        seq_length: int,
                        relevant_idx: list[int],
                        age_mean: float = 0.,
                        age_std:  float = 1.):
    """
    Lee un EDF con pyEDFlib, conserva solo los canales 'relevant_idx'
    (índices 0-based respecto al archivo), aplica el mismo transform y
    devuelve  (eeg_tensor, age_tensor).

    Parameters
    ----------
    edf_path      : ruta al archivo EDF.
    seq_length    : nº de muestras que usaste (e.g. 3000).
    relevant_idx  : lista de índices de canal que entrenaste (len = in_channels).
    age_mean/std  : para normalizar edad (si la usas; pon 0/1 si no).
    """
    edf_path = Path(edf_path)
    f = pyedflib.EdfReader(str(edf_path))

    # --- leer canales relevantes ------------------------------------------------
    sigs   = []
    for ch in relevant_idx:
        sig = f.readSignal(ch)             # numpy (N,)
        sigs.append(sig.astype(np.float32))
    data = np.vstack(sigs)                 # (C, N_total)
    f._close()                             # cerrar descriptor

    # --- recorte / padding a seq_length -----------------------------------------
    if data.shape[1] >= seq_length:
        data = data[:, :seq_length]
    else:
        pad = seq_length - data.shape[1]
        data = np.pad(data, ((0, 0), (0, pad)), mode="constant")

    # --- a tensor + mismo transform --------------------------------------------
    eeg_tensor = torch.from_numpy(data)      # (C, L) float32
    sample     = {'signal': eeg_tensor}
    sample     = transform_reduced(sample)   # aplica EegRandomCrop + DropChannels
    eeg_tensor = sample['signal']            # (len(relevant_idx), L_crop)

    # --- edad (si existe) -------------------------------------------------------
    try:
        h_age = int(f.getPatientAdditional())   # ejemplo: guardaron la edad aquí
    except Exception:
        h_age = 0
    age_norm = (h_age - age_mean) / age_std
    age_tensor = torch.tensor(age_norm, dtype=torch.float32)

    return eeg_tensor, age_tensor
