import os
import torch
from TTS.api import TTS
import gradio as gr

# 🚀 ACEITA OS TERMOS AUTOMATICAMENTE (Não precisa mais digitar 'y')
os.environ["COQUI_TOS_AGREED"] = "1"

# Configuração para GPU T4 do Colab
device = "cuda" if torch.cuda.is_available() else "cpu"

# Carregando modelo XTTS v2 (Motor Brasileiro)
print("📥 Carregando motor XTTS v2 (Brasil)...")
try:
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
    print("✅ Motor carregado com sucesso!")
except Exception as e:
    print(f"❌ Erro ao carregar motor: {e}")

def clonar_voz_miraplay(texto, audio_ref):
    try:
        # Verifica se o usuário enviou o áudio
        if audio_ref is None:
            return None
        
        output_file = "resultado_miraplay.wav"
        
        # O XTTS v2 com language="pt" remove o sotaque estrangeiro
        tts.tts_to_file(
            text=texto,
            speaker_wav=audio_ref,
            language="pt",
            file_path=output_file
        )
        
        return output_file
    except Exception as e:
        print(f"💥 Erro na geração: {e}")
        return None

# Interface profissional ajustada para aceitar vários formatos
app = gr.Interface(
    fn=clonar_voz_miraplay,
    inputs=[
        gr.Textbox(
            label="Texto para a IA falar (Em Português)", 
            placeholder="Ex: Olá, bem-vindo ao meu canal!"
        ),
        gr.Audio(
            label="Sua Voz de Referência (Suba MP3 ou WAV)", 
            type="filepath" # 'filepath' resolve o erro de 'Invalid file type'
        )
    ],
    outputs=gr.Audio(label="Áudio Gerado pela IA"),
    title="MIRAPLAY 2026 - XTTS v2 BRASIL",
    description="Sistema de clonagem de voz otimizado para Português. Use áudios de 6 a 10 segundos para melhor fidelidade."
)

if __name__ == "__main__":
    # share=True gera o link para o seu site
    app.launch(share=True, debug=True)
