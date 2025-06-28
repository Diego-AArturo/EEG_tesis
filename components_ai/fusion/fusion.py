import joblib

def fused_inference(meta_model, p_audio=None, p_eeg=None):
    p_audio = p_audio if p_audio is not None else 0.5
    p_eeg = p_eeg if p_eeg is not None else 0.5
    return meta_model.predict([[p_audio, p_eeg]])[0], meta_model.predict_proba([[p_audio, p_eeg]])[0][1]

# -------------------------------
# Guardar y cargar modelo
# -------------------------------

def save_meta_model(meta_model, path='meta_model.pkl'):
    joblib.dump(meta_model, path)

def load_meta_model(path='meta_model.pkl'):
    return joblib.load(path)
