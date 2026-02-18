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

def clonar_voz_definitivo(texto, arquivo_referencia):
    try:
        if arquivo_referencia is None:
            return None
        
        # Resolve o problema de reconhecimento de áudio no Colab
        caminho_original = arquivo_referencia.name
        caminho_temporario = "temp_audio_ref.wav"
        shutil.copy(caminho_original, caminho_temporario)
        
        output = "resultado_miraplay.wav"
        
        # Executa a síntese em Português
        tts.tts_to_file(
            text=texto,
            speaker_wav=caminho_temporario,
            language="pt",
            file_path=output
        )
        return output
    except Exception as e:
        print(f"💥 Erro na geração: {e}")
        return None

# --- DESIGN MODERNO (THEME) ---
meu_tema = gr.themes.Soft(
    primary_hue="blue",
    neutral_hue="slate",
).set(
    body_background_fill="*neutral_950",
    block_background_fill="*neutral_900",
    block_border_width="1px",
    button_primary_background_fill="*primary_600",
)

# --- CONSTRUÇÃO DA INTERFACE ---
with gr.Blocks(title="MIRAPLAY AI 2026") as app:
    gr.Markdown(
        """
        # 🎙️ MIRAPLAY AI - Clonagem de Voz
        ### Sistema otimizado para Português (Brasil) rodando em GPU T4.
        ---
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            input_text = gr.Textbox(
                label="Texto para a IA falar", 
                placeholder="Ex: Olá, eu sou a sua nova voz clonada!",
                lines=5
            )
            # gr.File evita o erro de "Invalid File Type" do navegador
            input_file = gr.File(
                label="Suba sua referência de voz (MP3 ou WAV)",
            )
            btn_gerar = gr.Button("🚀 GERAR CLONAGEM", variant="primary")
            
        with gr.Column(scale=1):
            gr.Markdown("### 🔊 Resultado")
            output_audio = gr.Audio(label="Áudio Gerado", interactive=False)
            gr.Markdown(
                """
                > **Atenção:** Se o link cair, reinicie a célula no Google Colab. 
                > O modelo leva cerca de 30 segundos para processar o áudio após clicar no botão.
                """
            )

    btn_gerar.click(
        fn=clonar_voz_definitivo,
        inputs=[input_text, input_file],
        outputs=output_audio
    )

# --- INICIALIZAÇÃO ---
if __name__ == "__main__":
    # Ajustado para evitar quedas e bugs de tema
    app.launch(
        share=True, 
        debug=True, 
        inline=False,
        theme=meu_tema # Tema movido para o launch para evitar Warning
    )
