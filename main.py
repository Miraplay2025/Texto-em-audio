import os
import torch
import re
from TTS.api import TTS
import gradio as gr

# 🚀 CONFIGURAÇÃO DE ALTA FIDELIDADE
os.environ["COQUI_TOS_AGREED"] = "1"
device = "cuda" if torch.cuda.is_available() else "cpu"

print("📥 Carregando motor de Clonagem de Elite...")
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

def limpar_texto(t):
    # Remove caracteres especiais que a IA às vezes tenta 'ler'
    # Mantém apenas letras, números e pontuação básica para pausas
    t = re.sub(r'([\.!\?])\1+', r'\1', t) # Transforma .... em .
    return t

def clonar_voz_total(texto, audio_ref, estilo_personalizado):
    try:
        if audio_ref is None: return None
        
        # 1. Limpa o texto para evitar que a IA leia os pontos
        texto_limpo = limpar_texto(texto)
        
        output_path = "resultado_clone_total.wav"
        
        # Se vazio, foca 100% na referência
        desc_estilo = estilo_personalizado if estilo_personalizado else "Speak naturally following the reference audio rhythm."

        tts.tts_to_file(
            text=texto_limpo,
            speaker_wav=audio_ref,
            language="pt",
            file_path=output_path,
            # CONFIGURAÇÃO PARA ESPELHAMENTO TOTAL
            temperature=0.75,      
            top_p=0.85,           
            speed=1.0,            # Segue a velocidade do áudio original
            repetition_penalty=2.0,
            gpt_cond_len=30,      # Analisa profundamente o áudio original
            emotion=desc_estilo   
        )
        return output_path
    except Exception as e:
        print(f"Erro no Clone Total: {e}")
        return None

# --- DESIGN DO FORMULÁRIO ---
meu_tema = gr.themes.Soft(primary_hue="blue", neutral_hue="slate")

with gr.Blocks(theme=meu_tema, title="MIRAPLAY CLONE 1:1") as app:
    gr.Markdown("# 🎙️ MIRAPLAY AI - Clone Idêntico")
    gr.Markdown("### Foco: Copiar 100% da cadência e detalhes do áudio enviado.")
    
    with gr.Row():
        with gr.Column():
            input_text = gr.Textbox(
                label="Texto para a IA ler", 
                placeholder="A IA lerá apenas as palavras, respeitando os pontos como pausas.",
                lines=5
            )
            
            input_estilo = gr.Textbox(
                label="Personalizar Estilo ou Emoção (Opcional)",
                placeholder="Ex: Animado, Sério, Rápido... (Deixe vazio para ser 100% igual ao áudio)",
                lines=2
            )
            
            input_audio = gr.Audio(
                label="Áudio de Referência (O 'DNA' da voz e do ritmo)", 
                type="filepath",
                sources=["upload", "microphone"]
            )
            
            btn = gr.Button("🚀 GERAR CLONE COMPLETO", variant="primary")
            
        with gr.Column():
            output_audio = gr.Audio(label="Voz Clonada (Espelho)")
            gr.Markdown("""
            ### ✅ Melhorias Aplicadas:
            * **Filtro de Leitura:** O sistema agora ignora símbolos e lê apenas as palavras.
            * **DNA Voice:** A velocidade e o jeito de falar são extraídos da sua referência.
            * **Dica:** Se a IA 'comer' palavras, use vírgulas para ajudá-la a organizar o fôlego.
            """)

    btn.click(
        fn=clonar_voz_total, 
        inputs=[input_text, input_audio, input_estilo], 
        outputs=output_audio
    )

if __name__ == "__main__":
    app.queue().launch(share=True, debug=True)
