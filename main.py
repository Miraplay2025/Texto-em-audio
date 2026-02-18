import os
import torch
import shutil
from TTS.api import TTS
import gradio as gr

# 🚀 CONFIGURAÇÃO PRO
os.environ["COQUI_TOS_AGREED"] = "1"
device = "cuda" if torch.cuda.is_available() else "cpu"

print("📥 Carregando motor XTTS v2 High-Fidelity...")
try:
    # Carregando com configurações de precisão
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
    print("✅ Motor de Alta Qualidade pronto!")
except Exception as e:
    print(f"❌ Erro: {e}")

def clonar_voz_premium(texto, audio_ref):
    try:
        if audio_ref is None:
            return None
        
        output_path = "resultado_miraplay_hq.wav"
        
        # AJUSTES DE QUALIDADE ELEVENLABS STYLE
        tts.tts_to_file(
            text=texto,
            speaker_wav=audio_ref,
            language="pt",
            file_path=output_path,
            # Parâmetros que melhoram a fluidez:
            temperature=0.65,      # Menos "robótico", mais natural
            repetition_penalty=2.0, # Evita gagueira
            top_k=50,              # Melhora a clareza das palavras
            top_p=0.8,             # Melhora a entonação
            speed=1.0              # Velocidade natural
        )
        
        return output_path
    except Exception as e:
        print(f"💥 Erro: {e}")
        return None

# --- DESIGN PREMIUM ---
meu_tema = gr.themes.Soft(primary_hue="blue", neutral_hue="slate")

with gr.Blocks(theme=meu_tema, title="MIRAPLAY AI PRO") as app:
    gr.Markdown("# 🎙️ MIRAPLAY AI - Clonagem Ultra Realista")
    gr.Markdown("### Foco em Qualidade: Nível ElevenLabs (PT-BR)")
    
    with gr.Row():
        with gr.Column():
            input_text = gr.Textbox(label="Texto (Use vírgulas e pontos para pausas naturais)", lines=6)
            input_audio = gr.Audio(label="Referência (Suba um áudio de 15s com voz limpa)", type="filepath")
            btn = gr.Button("🚀 GERAR ÁUDIO PREMIUM", variant="primary")
            
        with gr.Column():
            output_audio = gr.Audio(label="Voz Clonada (Alta Fidelidade)")
            gr.Markdown("> **Segredo da Qualidade:** Se o áudio de referência tiver ruído de fundo ou música, a clonagem ficará ruim. Use um áudio seco e claro.")

    btn.click(fn=clonar_voz_premium, inputs=[input_text, input_audio], outputs=output_audio)

if __name__ == "__main__":
    app.queue().launch(share=True, debug=True, inline=False)
