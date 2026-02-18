import os
import torch
from TTS.api import TTS
import gradio as gr
import shutil

# Aceita os termos automaticamente
os.environ["COQUI_TOS_AGREED"] = "1"

device = "cuda" if torch.cuda.is_available() else "cpu"

print("📥 Carregando motor XTTS v2...")
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

def clonar_voz_definitivo(texto, arquivo_referencia):
    try:
        if arquivo_referencia is None:
            return None
        
        # SOLUÇÃO DEFINITIVA: 
        # Criamos uma cópia do arquivo com extensão .wav para garantir que a IA reconheça,
        # não importa o que o Gradio pense que o arquivo seja.
        caminho_original = arquivo_referencia.name
        caminho_temporario = "temp_audio_ref.wav"
        shutil.copy(caminho_original, caminho_temporario)
        
        output = "resultado_miraplay.wav"
        
        tts.tts_to_file(
            text=texto,
            speaker_wav=caminho_temporario,
            language="pt",
            file_path=output
        )
        return output
    except Exception as e:
        print(f"Erro detectado: {e}")
        return None

# --- DESIGN MODERNO ---
meu_tema = gr.themes.Soft(primary_hue="blue").set(
    body_background_fill="*neutral_950",
    block_background_fill="*neutral_900",
)

with gr.Blocks(theme=meu_tema, title="MIRAPLAY AI 2026") as app:
    gr.Markdown("# 🎙️ MIRAPLAY AI - Clonagem de Voz")
    
    with gr.Row():
        with gr.Column():
            input_text = gr.Textbox(label="Texto para a IA falar", lines=4)
            # USANDO O COMPONENTE DE ARQUIVO SEM FILTROS (Solução para o erro de 'Invalid Type')
            input_file = gr.File(label="Suba seu áudio aqui (Qualquer formato)")
            btn = gr.Button("🚀 GERAR VOZ", variant="primary")
        
        with gr.Column():
            output_audio = gr.Audio(label="Resultado em Áudio")

    btn.click(fn=clonar_voz_definitivo, inputs=[input_text, input_file], outputs=output_audio)

if __name__ == "__main__":
    app.launch(share=True, debug=True)
