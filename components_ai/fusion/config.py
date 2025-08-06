import torch 
paths={
    "audio_ds":"components_ai/fusion/insumos_fusion/audio_val_ds.pkl",
    "eeg_ds":"components_ai/fusion/insumos_fusion/eeg_val_data.pkl",
    "checkpoint_audio":"components_ai/fusion/insumos_fusion/best_model.pt",
    "checkpoint_eeg":"components_ai/fusion/insumos_fusion/checkpoint_20250625_003848.pt",

}
parameters_eeg={
    "out_dims":2,
    "base_channel":128,
    "use_age":"no",
    
}
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")