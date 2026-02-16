import os
import torch
from TTS.api import TTS
import gradio as gr

# 🚀 Carregando o motor XTTS v2 (O melhor para Português)
print("📥 Baixando motor de voz multilingue... Aguarde, isso pode levar 2 minutos.")
device = "cuda" if torch.cuda.is_available() else "cpu"
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
print("✅ Motor XTTS v2 pronto para o Brasil!")

def clonar_voz_definitivo(texto, audio_ref):
    try:
        if audio_ref is None:
            return None
        
        output_path = "resultado_br.wav"
        
        # O XTTS v2 tem o parâmetro 'language' nativo!
        # Isso força a IA a usar a fonética do Brasil
        tts.tts_to_file(
            text=texto,
            speaker_wav=audio_ref,
            language="pt",
            file_path=output_path
        )
        
        return output_path
    except Exception as e:
        print(f"💥 Erro: {e}")
        return None

# Interface simples e poderosa
app = gr.Interface(
    fn=clonar_voz_definitivo,
    inputs=[
        gr.Textbox(label="O que a IA deve falar (Em Português)", placeholder="Olá, tudo bem?"),
        gr.Audio(type="filepath", label="Voz de Referência (Suba seu áudio aqui)")
    ],
    outputs=gr.Audio(label="Voz Clonada em Português"),
    title="MIRAPLAY 2026 - MODO BRASIL 🇧🇷"
)

if __name__ == "__main__":
    app.launch(share=True, debug=True)
