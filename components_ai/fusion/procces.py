# fusion_project/io_preprocess.py
from pathlib import Path
import torch, torchaudio
import numpy as np
import pyedflib
from components_ai.eeg.eeg_transforms import build_transform_reduced

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
                        relevant_idx: list[int],
                        transform=None,
                        age_mean: float = 71.2054,
                        age_std: float = 9.8287,
                        seq_length: int = 3000):
    if transform is None:
        transform = build_transform_reduced(relevant_idx)

    f = pyedflib.EdfReader(str(edf_path))
    sigs = [f.readSignal(i).astype(np.float32) for i in range(f.signals_in_file)]
    f._close()

    data = np.vstack(sigs)
    data = data[:, :seq_length] if data.shape[1] >= seq_length \
           else np.pad(data, ((0, 0), (0, seq_length - data.shape[1])), "constant")

    sample = transform({'signal': torch.tensor(data)})
    eeg_tensor = sample['signal']                       # (C_relev, L_crop)

    try:
        age = int(f.getPatientAdditional())
    except Exception:
        age = 0
    age_tensor = torch.tensor((age - age_mean) / age_std, dtype=torch.float32)
    return eeg_tensor, age_tensor
