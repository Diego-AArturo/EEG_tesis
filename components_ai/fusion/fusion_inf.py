def predict_audio(model, audio_tensor, device):
    with torch.no_grad():
        return model(audio_tensor.unsqueeze(0).to(device)).item()

def predict_eeg(model, eeg_tensor, age_tensor, device):
    with torch.no_grad():
        return model(eeg_tensor.unsqueeze(0).to(device), age_tensor.unsqueeze(0).to(device)).item()

def fused_inference(meta_model, p_audio=None, p_eeg=None):
    p_audio = p_audio if p_audio is not None else 0.5
    p_eeg = p_eeg if p_eeg is not None else 0.5
    return meta_model.predict([[p_audio, p_eeg]])[0], meta_model.predict_proba([[p_audio, p_eeg]])[0][1]

def predict_multimodel():
    p_audio = predict_audio(model_audio, sample['audio_tensor'], device)
    p_eeg = predict_eeg(model_audio, sample['audio_tensor'], device)
    result = fused_inference(meta_model, p_audio, p_eeg)
    return result