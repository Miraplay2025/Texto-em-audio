import os
import torch
from TTS.api import TTS
import gradio as gr

# Aceita os termos automaticamente
os.environ["COQUI_TOS_AGREED"] = "1"

device = "cuda" if torch.cuda.is_available() else "cpu"

print("📥 Carregando motor XTTS v2...")
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

def clonar_voz_definitivo(texto, arquivo_referencia):
    try:
        if arquivo_referencia is None:
            return None
        
        caminho_audio = arquivo_referencia.name 
        output = "resultado_miraplay.wav"
        
        tts.tts_to_file(
            text=texto,
            speaker_wav=caminho_audio,
            language="pt",
            file_path=output
        )
        return output
    except Exception as e:
        print(f"Erro: {e}")
        return None

# --- CONFIGURAÇÃO DE DESIGN (THEME) ---
meu_tema = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="slate",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Poppins"), "ui-sans-serif", "system-ui", "sans-serif"],
).set(
    body_background_fill="*neutral_950",
    block_background_fill="*neutral_900",
    block_border_width="1px",
    button_primary_background_fill="*primary_600",
    button_primary_background_fill_hover="*primary_500",
)

# --- INTERFACE CUSTOMIZADA ---
with gr.Blocks(theme=meu_tema, title="MIRAPLAY AI 2026") as app:
    gr.Markdown(
        """
        # 🎙️ MIRAPLAY AI - Clonagem de Voz
        ### Transforme texto em áudio com sua própria voz em segundos.
        ---
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            input_text = gr.Textbox(
                label="Conteúdo da fala", 
                placeholder="Digite aqui o texto que a IA deve dizer...",
                lines=5
            )
            input_file = gr.File(
                label="Sua Voz de Referência",
                file_types=["audio"]
            )
            btn_gerar = gr.Button("🚀 GERAR CLONAGEM", variant="primary")
            
        with gr.Column(scale=1):
            gr.Markdown("### 🔊 Resultado")
            output_audio = gr.Audio(label="Áudio Gerado", interactive=False)
            gr.Markdown(
                """
                > **Dica de Ouro:** Para melhores resultados, use um áudio de referência limpo, sem música de fundo e com cerca de 10 segundos de fala.
                """
            )

    btn_gerar.click(
        fn=clonar_voz_definitivo,
        inputs=[input_text, input_file],
        outputs=output_audio
    )

if __name__ == "__main__":
    app.launch(share=True, debug=True)
