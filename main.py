import os
import torch
from TTS.api import TTS
import gradio as gr

# Configuração para GPU T4
device = "cuda" if torch.cuda.is_available() else "cpu"

# Carregando modelo XTTS v2 (Multilingue)
print("📥 Carregando motor XTTS v2...")
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

def clonar_voz(texto, audio_ref):
    try:
        if audio_ref is None: return None
        output = "resultado.wav"
        
        # Forçando Português para evitar sotaque
        tts.tts_to_file(
            text=texto,
            speaker_wav=audio_ref,
            language="pt",
            file_path=output
        )
        return output
    except Exception as e:
        print(f"Erro: {e}")
        return None

app = gr.Interface(
    fn=clonar_voz,
    inputs=[gr.Textbox(label="Texto (PT-BR)"), gr.Audio(type="filepath", label="Voz de Referência")],
    outputs=gr.Audio(label="Áudio Clonado"),
    title="MIRAPLAY 2026"
)

if __name__ == "__main__":
    app.launch(share=True)
