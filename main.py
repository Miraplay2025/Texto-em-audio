import os
import torch
import shutil
import time
from TTS.api import TTS
import gradio as gr

# 🚀 ACEITA OS TERMOS AUTOMATICAMENTE
os.environ["COQUI_TOS_AGREED"] = "1"

# Configuração para GPU T4
device = "cuda" if torch.cuda.is_available() else "cpu"

print("📥 Carregando motor XTTS v2 (Brasil)...")
try:
    # O modelo é carregado uma única vez para economizar memória da T4
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
    print("✅ Motor carregado com sucesso!")
except Exception as e:
    print(f"❌ Erro ao carregar motor: {e}")

def clonar_voz_miraplay(texto, audio_ref):
    try:
        if audio_ref is None:
            return None
        
        # O arquivo de saída
        output_path = "resultado_miraplay.wav"
        
        # Geração da voz (XTTS v2 suporta MP3/WAV como referência via filepath)
        tts.tts_to_file(
            text=texto,
            speaker_wav=audio_ref,
            language="pt",
            file_path=output_path
        )
        
        return output_path
    except Exception as e:
        print(f"💥 Erro na geração: {e}")
        return None

# --- DESIGN MODERNO ---
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
with gr.Blocks(theme=meu_tema, title="MIRAPLAY AI 2026") as app:
    gr.Markdown("# 🎙️ MIRAPLAY AI - Clonagem de Voz")
    gr.Markdown("### Status: 🟢 Conectado à GPU T4")
    
    with gr.Row():
        with gr.Column(scale=1):
            input_text = gr.Textbox(
                label="Texto para a IA falar", 
                placeholder="Escreva aqui o que você quer que a IA diga...",
                lines=5
            )
            # Componente de Áudio configurado para aceitar arquivos e microfone
            input_audio = gr.Audio(
                label="Sua Voz de Referência (MP3 ou WAV)",
                type="filepath",
                sources=["upload", "microphone"]
            )
            btn_gerar = gr.Button("🚀 GERAR CLONAGEM", variant="primary")
            
        with gr.Column(scale=1):
            gr.Markdown("### 🔊 Áudio Gerado")
            output_audio = gr.Audio(label="Resultado Final", interactive=False)
            gr.Markdown(
                """
                > **Dica:** Mantenha a aba do Colab aberta para o link não expirar. 
                > Se houver erro de 'Invalid Type', tente gravar diretamente pelo microfone acima.
                """
            )

    btn_gerar.click(
        fn=clonar_voz_miraplay,
        inputs=[input_text, input_audio],
        outputs=output_audio
    )

# --- INICIALIZAÇÃO E MANUTENÇÃO DA CONEXÃO ---
if __name__ == "__main__":
    print("🚀 Servidor MIRAPLAY iniciando...")
    
    # launch com share=True cria o link público
    # debug=True ajuda a ver erros em tempo real
    app.launch(
        share=True, 
        debug=True, 
        show_error=True,
        inline=False
    )
    
    # Loop de segurança para o Colab não encerrar a execução da célula
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        print("Servidor parado pelo usuário.")
