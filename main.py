import os
import torch
from TTS.api import TTS
import gradio as gr
import shutil

# 🚀 ACEITA OS TERMOS AUTOMATICAMENTE
os.environ["COQUI_TOS_AGREED"] = "1"

# Configuração para GPU T4
device = "cuda" if torch.cuda.is_available() else "cpu"

print("📥 Carregando motor XTTS v2 (Brasil)...")
try:
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
    print("✅ Motor carregado com sucesso!")
except Exception as e:
    print(f"❌ Erro ao carregar motor: {e}")

def clonar_voz_definitivo(texto, audio_ref):
    try:
        if audio_ref is None:
            return None
        
        # O audio_ref aqui já é o caminho do arquivo temporário
        output = "resultado_miraplay.wav"
        
        tts.tts_to_file(
            text=texto,
            speaker_wav=audio_ref,
            language="pt",
            file_path=output
        )
        return output
    except Exception as e:
        print(f"💥 Erro na geração: {e}")
        return None

# --- DESIGN MODERNO ---
meu_tema = gr.themes.Soft(primary_hue="blue", neutral_hue="slate")

with gr.Blocks(theme=meu_tema, title="MIRAPLAY AI 2026") as app:
    gr.Markdown("# 🎙️ MIRAPLAY AI - Clonagem de Voz")
    
    with gr.Row():
        with gr.Column(scale=1):
            input_text = gr.Textbox(
                label="Texto para a IA falar", 
                placeholder="Digite o texto aqui...",
                lines=5
            )
            # MUDANÇA AQUI: Voltamos para gr.Audio mas com 'filepath' e 'label'
            # Isso habilita a seleção de arquivos de áudio no celular e PC
            input_file = gr.Audio(
                label="Selecione ou Grave seu Áudio (MP3 ou WAV)",
                type="filepath",
                sources=["upload", "microphone"]
            )
            btn_gerar = gr.Button("🚀 GERAR CLONAGEM", variant="primary")
            
        with gr.Column(scale=1):
            gr.Markdown("### 🔊 Resultado")
            output_audio = gr.Audio(label="Áudio Gerado", interactive=False)

    btn_gerar.click(
        fn=clonar_voz_definitivo,
        inputs=[input_text, input_file],
        outputs=output_audio
    )

if __name__ == "__main__":
    app.launch(share=True, debug=True)
