import torch.nn.functional as F
import torch

def predict_audio(model, audio_tensor, device):
    """Recibe UN tensor (C,H,W) y devuelve (clase, probabilidad_clase1)."""
    with torch.no_grad():
        logits = model(audio_tensor.unsqueeze(0).to(device))
        probs  = F.softmax(logits, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()
        pred_prob  = probs[0, 1].item()
        return pred_class, pred_prob

def predict_eeg(model, eeg_tensor, age_tensor, device):
    """
    Devuelve (clase_predicha, probabilidad_clase1)
    - Si el modelo ya devuelve un escalar (sigmoid), lo usa tal cual.
    - Si devuelve 2 logits, aplica softmax.
    """
    with torch.no_grad():
        out = model(eeg_tensor.unsqueeze(0).to(device),
                    age_tensor.unsqueeze(0).to(device))

        # Caso 1: output shape = (1,)  -> probabilidad directa
        if out.ndim == 1 or out.shape[1] == 1:
            prob = out.squeeze().item()
            pred_class = int(prob >= 0.5)
            return pred_class, prob

        # Caso 2: output shape = (1, 2) -> logits de 2 clases
        probs = F.softmax(out, dim=1)          # (1,2)
        prob = probs[0, 1].item()              # p(clase 1)
        pred_class = torch.argmax(probs, dim=1).item()
        return pred_class, prob