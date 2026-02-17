import os
import torch
from TTS.api import TTS
import gradio as gr

# Aceita os termos automaticamente
os.environ["COQUI_TOS_AGREED"] = "1"

device = "cuda" if torch.cuda.is_available() else "cpu"

print("📥 Carregando motor XTTS v2...")
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

def clonar_voz(texto, audio_ref):
    try:
        if audio_ref is None: return None
        output = "resultado.wav"
        
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

# MUDANÇA AQUI: Adicionamos 'sources' e mudamos o componente de áudio
app = gr.Interface(
    fn=clonar_voz,
    inputs=[
        gr.Textbox(label="Texto (PT-BR)"),
        gr.Audio(
            label="Suba seu áudio (Qualquer formato)", 
            type="filepath", 
            sources=["upload", "microphone"] # Força a aceitar upload manual
        )
    ],
    outputs=gr.Audio(label="Áudio Gerado"),
    title="MIRAPLAY 2026"
)

if __name__ == "__main__":
    app.launch(share=True, debug=True)
