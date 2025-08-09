import gradio as gr
import tempfile, os
import pandas as pd
import matplotlib.pyplot as plt

from components_ai.fusion.fusion_inf import predict_multimodel

def predict_fn(audio_file, eeg_file):
    if not audio_file or not eeg_file:
        return "Sube ambos archivos", None

    try:
        # Guardar temporalmente los archivos
        with tempfile.NamedTemporaryFile(delete=False, suffix='.edf') as tmp_eeg:
            eeg_path = tmp_eeg.name
            tmp_eeg.write(eeg_file.read())

        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(audio_file.name)[1]) as tmp_audio:
            audio_path = tmp_audio.name
            tmp_audio.write(audio_file.read())

        # Ejecutar predicción
        prob_dcl, prob_normal = predict_multimodel(audio_path, eeg_path)
        resultado = "DCL" if prob_dcl > prob_normal else "Normal"

        # Crear gráfico
        fig, ax = plt.subplots(figsize=(5, 2))
        ax.barh(["DCL", "Normal"], [prob_dcl, prob_normal], color=["#9BE1FF", "#418FDE"])
        ax.set_xlim(0, 1)
        ax.set_xlabel("Probabilidad")
        ax.set_title("Resultado del modelo")
        plt.tight_layout()

        # Limpiar archivos
        os.remove(eeg_path)
        os.remove(audio_path)

        return f"Diagnóstico: {resultado}", fig

    except Exception as e:
        return f"Error al procesar: {str(e)}", None


# Crear la interfaz
title = "<h1 style='text-align: center; color: black;'>DETECCIÓN DE DETERIORO COGNITIVO</h1>"
interface = gr.Interface(
    fn=predict_fn,
    inputs=[
        gr.File(label="Cargar EEG (.edf)", type="binary"),
        gr.File(label="Cargar Audio (.wav o .mp3)", type="binary"),
    ],
    outputs=[
        gr.Text(label="Resultado"),
        gr.Plot(label="Probabilidades")
    ],
    title=title,
    theme="default"
)

if __name__ == "__main__":
    interface.launch()
