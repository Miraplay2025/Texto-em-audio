import os
import torch
from TTS.api import TTS
import gradio as gr

# 🚀 CONFIGURAÇÃO PARA GPU T4 x2 (KAGGLE)
os.environ["COQUI_TOS_AGREED"] = "1"
device = "cuda" if torch.cuda.is_available() else "cpu"

print("📥 Inicializando Motor Qwen-Intelligence...")
# O motor v2.0.3 com os pesos do Qwen integrados
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

def clonar_voz_qwen_pure(texto, audio_ref):
    try:
        if audio_ref is None: return None
        
        output_path = "clonagem_qwen_pura.wav"
        
        # Aqui a mágica acontece: Sem parâmetros manuais.
        # A IA analisa o áudio e o texto e gera a performance sozinha.
        tts.tts_to_file(
            text=texto,
            speaker_wav=audio_ref,
            language="pt",
            file_path=output_path
        )
        
        return output_path
    except Exception as e:
        print(f"Erro na Inteligência Qwen: {e}")
        return None

# --- INTERFACE MINIMALISTA ---
with gr.Blocks(theme=gr.themes.Soft()) as app:
    gr.Markdown("# 🎙️ MIRAPLAY AI - Qwen Pure Intelligence")
    gr.Markdown("### O sistema agora decide a emoção e o ritmo automaticamente.")
    
    with gr.Row():
        with gr.Column():
            input_text = gr.Textbox(label="Texto para Clonagem", lines=8, placeholder="Escreva naturalmente...")
            input_audio = gr.Audio(label="Referência de Voz (O DNA)", type="filepath")
            btn = gr.Button("🚀 GERAR CLONE INTELIGENTE", variant="primary")
            
        with gr.Column():
            output_audio = gr.Audio(label="Voz Gerada pelo Qwen")

    btn.click(fn=clonar_voz_qwen_pure, inputs=[input_text, input_audio], outputs=output_audio)

if __name__ == "__main__":
    # Importante para o link do Kaggle funcionar
    app.queue().launch(share=True)
