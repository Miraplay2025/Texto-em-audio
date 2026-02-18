import os
import torch
from TTS.api import TTS
import gradio as gr

# 🚀 CONFIGURAÇÃO DE ALTO DESEMPENHO
os.environ["COQUI_TOS_AGREED"] = "1"
device = "cuda" if torch.cuda.is_available() else "cpu"

print("📥 Carregando motor XTTS v2 Turbo...")
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

def clonar_voz_miraplay_v3(texto, audio_ref, estilo):
    try:
        if audio_ref is None: return None
        
        output_path = "resultado_miraplay_turbo.wav"
        
        # Configurações focadas em narração fluida e velocidade
        # Speed 1.1 ou 1.15 tira aquela sensação de 'lentidão'
        config = {
            "Narração Rápida (Estilo Youtube)": {"temp": 0.75, "top_p": 0.85, "speed": 1.15},
            "Padrão ElevenLabs (Equilibrado)": {"temp": 0.70, "top_p": 0.8, "speed": 1.05},
            "Entusiasta/Vendas": {"temp": 0.80, "top_p": 0.9, "speed": 1.10},
            "Sério/Noticiário": {"temp": 0.55, "top_p": 0.75, "speed": 1.0}
        }
        
        escolha = config.get(estilo, config["Padrão ElevenLabs (Equilibrado)"])

        tts.tts_to_file(
            text=texto,
            speaker_wav=audio_ref,
            language="pt",
            file_path=output_path,
            temperature=escolha["temp"],
            top_p=escolha["top_p"],
            speed=escolha["speed"],
            repetition_penalty=2.0, # Evita travamentos na fala
            length_penalty=1.0,      # Mantém a duração das pausas natural
        )
        return output_path
    except Exception as e:
        print(f"Erro na geração: {e}")
        return None

# --- DESIGN DO FORMULÁRIO ---
meu_tema = gr.themes.Soft(primary_hue="blue")

with gr.Blocks(theme=meu_tema, title="MIRAPLAY AI TURBO") as app:
    gr.Markdown("# 🎙️ MIRAPLAY AI - Narração Profissional")
    gr.Markdown("### Foco: Velocidade e Fluidez (PT-BR)")
    
    with gr.Row():
        with gr.Column():
            input_text = gr.Textbox(
                label="Texto para Narração", 
                placeholder="Dica: Use vírgulas para pausas breves e pontos para respiração.",
                lines=6
            )
            
            input_estilo = gr.Dropdown(
                choices=["Narração Rápida (Estilo Youtube)", "Padrão ElevenLabs (Equilibrado)", "Entusiasta/Vendas", "Sério/Noticiário"],
                value="Narração Rápida (Estilo Youtube)",
                label="Estilo e Velocidade"
            )
            
            input_audio = gr.Audio(label="Voz de Referência (Suba um áudio limpo)", type="filepath")
            btn = gr.Button("🚀 GERAR NARRAÇÃO AGORA", variant="primary")
            
        with gr.Column():
            output_audio = gr.Audio(label="Áudio Clonado (Narração)")
            gr.Markdown("""
            ### 💡 Como garantir a qualidade:
            1. **Evite frases gigantes sem vírgula:** A IA precisa saber onde respirar.
            2. **Qualidade do Microfone:** Se o áudio que você subir estiver abafado, a narração sairá abafada.
            3. **Velocidade:** O modo 'Estilo Youtube' é 15% mais rápido que o normal para manter o engajamento.
            """)

    btn.click(fn=clonar_voz_miraplay_v3, inputs=[input_text, input_audio, input_estilo], outputs=output_audio)

if __name__ == "__main__":
    app.queue().launch(share=True, debug=True)
