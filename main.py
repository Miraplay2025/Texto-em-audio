import os
from f5_tts.api import F5TTS
import gradio as gr

# Carregamento da IA (Versão simplificada para estabilidade)
print("🚀 Iniciando Servidor de IA...")
tts = F5TTS() 

def processar_clonagem(texto, audio_ref):
    try:
        print(f"📥 Processando: {texto[:20]}...")
        
        if audio_ref is None:
            return None
        
        output_path = "output_audio.wav"
        
        # Inferência
        tts.infer(
            gen_text=texto,
            ref_file=audio_ref,
            output_file=output_path
        )
        
        print("✅ Sucesso!")
        return output_path
    except Exception as e:
        print(f"💥 Erro: {str(e)}")
        return None

# Interface Gradio - Mais estável para evitar quedas
demo = gr.Interface(
    fn=processar_clonagem,
    inputs=[
        gr.Textbox(label="Texto para converter"), 
        gr.Audio(type="filepath", label="Voz de referência (Áudio curto)")
    ],
    outputs=gr.Audio(label="Voz Clonada"),
    title="Miraplay AI - Clonagem de Voz"
)

# O segredo para não cair: usar apenas o launch do gradio com share=True
demo.launch(share=True, debug=True)
