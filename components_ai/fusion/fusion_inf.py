from fusion import load_meta_model, fused_inference
from predict import predict_audio, predict_eeg
from config import paths, get_device, parameters_eeg
from models import load_audio_model, load_eeg_model

device=get_device()
meta_model=load_meta_model("modelo_fusion") #importar desde colab
audio_model=load_audio_model(paths["checkpoint_audio"],device)
eeg_model=load_eeg_model(paths["checkpoint_eeg"],device,parameters_eeg)
_,p_audio = predict_audio(audio_model, dataset_audio[10]['audio_tensor'], device) #El dataset_audio[0] hace referencia al registro que se quiere predecir
_,p_eeg = predict_eeg(eeg_model, dataset_eeg[10]['eeg_tensor'], dataset_eeg[0]['age'], device)

