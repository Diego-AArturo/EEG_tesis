import streamlit as st
import pandas as pd
import altair as alt
from components_ai.fusion.fusion_inf import predict_multimodel

# Título principal
st.markdown(
    "<h1 style='text-align: center; color: white; margin-bottom: 2rem;'>DETECCIÓN DE DETERIORO COGNITIVO</h1>",
    unsafe_allow_html=True
)

with st.form("input_form"):
    left, right = st.columns([3,2], gap="large")  # Mayor espacio entre columnas

    # COL 1: carga de archivos
    with left:
        st.markdown("<h3 style='text-align: center;'>Cargar archivos</h3>", unsafe_allow_html=True)
        uploaded_file_eeg = st.file_uploader("EEG (.edf)", type=["edf"])
        uploaded_file_audio = st.file_uploader("Audio (.wav o .mp3)", type=["wav", "mp3"])
        st.markdown("<div style='margin-top: 2rem;'></div>", unsafe_allow_html=True)
        procesar = st.form_submit_button("Procesar", type="primary")

    # COL 2: Diagnóstico y gráfico
    with right:
        contenedor = st.container()
        contenedor.markdown("<h3 style='text-align: center;'>Diagnóstico</h3>", unsafe_allow_html=True)
        contenedor.markdown("<div style='margin-bottom: 1rem;'></div>", unsafe_allow_html=True)

        resultado_placeholder = contenedor.empty()      # etiqueta
        resultado_placeholder_valor = contenedor.empty()  # valor
        contenedor.markdown("<div style='margin-bottom: 1rem;'></div>", unsafe_allow_html=True)

        if procesar:
            resultado = "DCL"
            
            # Guardar archivos temporalmente
            import tempfile
            import os
            
            if not uploaded_file_eeg or not uploaded_file_audio:
                st.error("Por favor, sube ambos archivos (EEG y Audio)")
                return
                
            try:
                # Crear archivos temporales
                with tempfile.NamedTemporaryFile(delete=False, suffix='.edf') as tmp_eeg:
                    tmp_eeg.write(uploaded_file_eeg.getvalue())
                    eeg_path = tmp_eeg.name
                    
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file_audio.name)[1]) as tmp_audio:
                    tmp_audio.write(uploaded_file_audio.getvalue())
                    audio_path = tmp_audio.name
                
                # Realizar predicción
                try:
                    prob_dcl, prob_normal = predict_multimodel(eeg_path, audio_path)
                finally:
                    # Limpiar archivos temporales
                    os.unlink(eeg_path)
                    os.unlink(audio_path)
            except Exception as e:
                st.error(f"Error al procesar los archivos: {str(e)}")
                return

            resultado_placeholder.markdown(
                "<div style='text-align: center; background-color: #262730;margin-top: 0.5rem; padding: 0.5rem; border-radius: 0.5rem;'>"
                "<span style='color: #AAAAAA;'>Resultado del modelo:</span></div>",
                unsafe_allow_html=True
            )

            resultado_placeholder_valor.markdown(
                f"<h4 style='text-align: center; color: #F95E5E; margin-top: 0.5rem;'>{resultado}</h4>",
                unsafe_allow_html=True
            )

            data = pd.DataFrame({
                "condición": ["DCL", "Normal"],
                "probabilidad": [prob_dcl, prob_normal]
            })

            chart = alt.Chart(data).mark_bar().encode(
                x=alt.X("probabilidad", title="Probabilidad (%)"),
                y=alt.Y("condición", sort='-x'),
                color=alt.Color("condición", scale=alt.Scale(range=["#9BE1FF", "#418FDE"]))
            ).properties(height=200)

            contenedor.altair_chart(chart, use_container_width=True)

        # Botón Limpiar (sin funcionalidad real aún)
        contenedor.markdown("<div style='margin-top: 0.5rem;'></div>", unsafe_allow_html=True)
        limpiar = st.form_submit_button("Limpiar")
        if limpiar:
            # Reiniciar el estado
            st.session_state.clear()
            st.experimental_rerun()





