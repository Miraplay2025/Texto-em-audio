import os
import torch
import shutil
import time
from TTS.api import TTS
import gradio as gr

# 🚀 CONFIGURAÇÃO DE AMBIENTE E TERMOS
os.environ["COQUI_TOS_AGREED"] = "1"
device = "cuda" if torch.cuda.is_available() else "cpu"

# 📥 CARREGAMENTO DO MOTOR
print("📥 Carregando motor XTTS v2 (Brasil)... Aguarde.")
try:
    # Carrega o modelo na GPU T4
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
    print("✅ Motor carregado e pronto para uso!")
except Exception as e:
    print(f"❌ Erro ao iniciar motor: {e}")

# 🎙️ FUNÇÃO DE CLONAGEM
def clonar_voz_miraplay(texto, audio_ref):
    try:
        if audio_ref is None:
            return None
        
        output_path = "resultado_miraplay.wav"
        
        # O parâmetro language="pt" é essencial para o sotaque brasileiro
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

# 🎨 DESIGN DA INTERFACE (THEME)
meu_tema = gr.themes.Soft(
    primary_hue="blue",
    neutral_hue="slate",
).set(
    body_background_fill="*neutral_950",
    block_background_fill="*neutral_900",
    block_border_width="1px",
    button_primary_background_fill="*primary_600",
)

# 🏗️ CONSTRUÇÃO DOS BLOCOS
with gr.Blocks(theme=meu_tema, title="MIRAPLAY AI 2026") as app:
    gr.Markdown("# 🎙️ MIRAPLAY AI - Clonagem de Voz")
    gr.Markdown("### Status: 🟢 Servidor Ativo na GPU T4")
    
    with gr.Row():
        with gr.Column(scale=1):
            input_text = gr.Textbox(
                label="Texto para a IA falar", 
                placeholder="Escreva aqui o que a IA deve dizer...",
                lines=5
            )
            # Componente de Áudio com suporte total a formatos
            input_audio = gr.Audio(
                label="Sua Voz de Referência",
                type="filepath",
                sources=["upload", "microphone"]
            )
            btn_gerar = gr.Button("🚀 GERAR CLONAGEM", variant="primary")
            
        with gr.Column(scale=1):
            gr.Markdown("### 🔊 Áudio Gerado")
            output_audio = gr.Audio(label="Resultado Final", interactive=False)
            gr.Markdown(
                """
                > **Dica Importante:** Se o link parar de responder, verifique se a célula do Colab ainda está com o ícone de 'Stop' girando. Se o ícone sumiu, a conexão caiu.
                """
            )

    btn_gerar.click(
        fn=clonar_voz_miraplay,
        inputs=[input_text, input_audio],
        outputs=output_audio
    )

# 🚀 INICIALIZAÇÃO COM FILA (QUEUE) PARA EVITAR QUE O COLAB FECHE
if __name__ == "__main__":
    print("🚀 Servidor MIRAPLAY em execução...")
    
    # O .queue() cria uma fila de espera que mantém o servidor ocupado e ativo
    app.queue().launch(
        share=True, 
        debug=True, 
        show_error=True,
        inline=False,
        prevent_thread_leak=True
    )
