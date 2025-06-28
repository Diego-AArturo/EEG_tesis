import joblib
from torch.utils.data import DataLoader

def load_audio_dataset(path):
    val_ds = joblib.load(path)
    val_dl = DataLoader(val_ds, batch_size=16, shuffle=False)
    dataset_audio = []
    for spec, label in val_dl:                           # ajusta si tu Dataset devuelve un dict
        for s, l in zip(spec, label):
            dataset_audio.append({'audio_tensor': s, 'label': int(l.item())})
    return dataset_audio

def load_eeg_dataset(path):
    dataset_eeg = joblib.load(path)
    return dataset_eeg

