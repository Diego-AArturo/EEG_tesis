from components_ai.eeg.caueeg.datasets.caueeg_dataset import *
from components_ai.eeg.caueeg.datasets.caueeg_script import *
from components_ai.eeg.caueeg.datasets.pipeline import *
from torchvision import transforms

def build_transform_reduced(relevant_ch: list[int], crop_len: int = 300 * 10):
    drop = [ch for ch in range(20) if ch not in relevant_ch]
    return transforms.Compose([
        EegRandomCrop(crop_length=crop_len, latency=200*10),
        EegDropChannels(drop),
        EegToTensor()
    ])