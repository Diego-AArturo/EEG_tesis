import torch
from pathlib import Path
from .fusion import load_meta_model, fused_inference
from .models import load_audio_model, load_eeg_model
from .config import paths, get_device, parameters_eeg
from .procces import load_audio_file, load_eeg_file_pyedf

# Variables globales para los modelos
_meta_model = None
_audio_model = None
_eeg_model = None
_device = None

def _ensure_models_loaded():
    """Asegura que los modelos estén cargados"""
    global _meta_model, _audio_model, _eeg_model, _device
    
    if _meta_model is None:
        _device = get_device()
        _meta_model = load_meta_model(str(Path(__file__).parent / "insumos_fusion" / "meta_model.pkl"))
        _audio_model = load_audio_model(paths["checkpoint_audio"], _device)
        _eeg_model = load_eeg_model(paths["checkpoint_eeg"], _device, parameters_eeg)

def predict_multimodel(path_audio, path_eeg):
    """
    Realiza la predicción de fusión a partir de los archivos de audio y EEG.
    """
    _ensure_models_loaded()
    
    # Procesar audio
    audio_tensor = load_audio_file(path_audio)
    
    # Procesar EEG
    eeg_tensor, age_tensor = load_eeg_file_pyedf(
        edf_path=path_eeg,
        relevant_idx=[0,1,2,3],
        transform=None
    )

    
    # Obtener predicciones
    with torch.no_grad():
        _, p_audio = _audio_model(audio_tensor.unsqueeze(0).to(_device))
        _, p_eeg = _eeg_model(eeg_tensor.unsqueeze(0).to(_device))
    
    # Fusionar predicciones
    pred, prob = fused_inference(_meta_model, p_audio=p_audio, p_eeg=p_eeg)
    
    return float(prob[0]), float(prob[1])  # DCL, Normal