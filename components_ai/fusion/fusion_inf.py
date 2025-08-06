from components_ai.fusion.fusion import load_meta_model, fused_inference
from components_ai.fusion.predict import predict_audio, predict_eeg
from components_ai.fusion.config import paths, get_device, parameters_eeg
from components_ai.fusion.models import load_audio_model, load_eeg_model
from components_ai.fusion.procces import load_audio_file, load_eeg_file_pyedf


device=get_device()
def predict_multimodel(path_audio, path_eeg):
    """
    Realiza la predicción de fusión a partir de los archivos de audio y EEG.
    """
    # Cargar modelos y realizar predicciones
    meta_model=load_meta_model("components_ai/fusion/insumos_fusion/meta_model.pkl") #importar desde colab
    audio_model=load_audio_model(paths["checkpoint_audio"],device)
    eeg_model=load_eeg_model(paths["checkpoint_eeg"],device,parameters_eeg)
    _,p_audio = predict_audio(audio_model, load_audio_file(path_audio), device) #El dataset_audio[0] hace referencia al registro que se quiere predecir
    _,p_eeg = predict_eeg(eeg_model, load_eeg_file_pyedf(path_eeg)[0], load_eeg_file_pyedf(path_eeg)[1], device)

    pred, prob = fused_inference(meta_model, p_audio=p_audio, p_eeg=p_eeg)
    return pred, prob