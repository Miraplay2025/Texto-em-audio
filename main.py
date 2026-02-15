import os
import soundfile as sf
import whisper
import gradio as gr
from f5_tts.api import F5TTS

# Inicializando a IA
print("🚀 Miraplay AI: Otimizando para Português Brasileiro...")
tts = F5TTS()
modelo_transcritor = whisper.load_model("base")

def clonar_voz_miraplay(texto_para_gerar, audio_ref, idioma):
    try:
        if audio_ref is None:
            return None
        
        # 1. Transcrição automática com foco no idioma escolhido
        print(f"🎧 Analisando áudio de referência em {idioma}...")
        lang_code = "pt" if idioma == "Português" else "en"
        
        # O Whisper força a detecção da língua para não confundir a fonética
        resultado = modelo_transcritor.transcribe(audio_ref, language=lang_code)
        texto_detectado = resultado["text"].strip()
        
        print(f"📝 Texto da amostra: {texto_detectado}")

        output_file = "saida_miraplay.wav"

        # 2. O PULO DO GATO: Para o Português, o F5-TTS precisa que o texto 
        # seja processado sem caracteres que ele interprete como fonemas ingleses.
        # Vamos garantir que o modelo saiba que é um texto corrido.
        
        wav, sr, _ = tts.infer(
            gen_text=texto_para_gerar,
            ref_file=audio_ref,
            ref_text=texto_detectado
        )
        
        sf.write(output_file, wav, sr)
        print(f"✅ Clonagem concluída!")
        return output_file

    except Exception as e:
        print(f"💥 Erro: {str(e)}")
        return None

# Interface mais moderna
app = gr.Interface(
    fn=clonar_voz_miraplay,
    inputs=[
        gr.Textbox(label="Texto que a IA vai falar", placeholder="Ex: Olá, eu sou a nova voz da Miraplay."),
        gr.Audio(type="filepath", label="Áudio de Referência (Amostra da voz)"),
        gr.Radio(
            choices=["Português", "Inglês"], 
            value="Português", 
            label="Idioma do Processamento"
        )
    ],
    outputs=gr.Audio(label="Áudio Gerado"),
    title="MIRAPLAY 2026 - BRASIL",
    description="Sistema otimizado para reconhecimento de fonemas em Português."
)

if __name__ == "__main__":
    app.launch(share=True, debug=True)
