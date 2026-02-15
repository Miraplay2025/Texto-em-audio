import os
import soundfile as sf
import whisper  # Nova biblioteca para transcrição automática
from f5_tts.api import F5TTS
import gradio as gr

# Carregando os modelos
print("🚀 Miraplay AI: Carregando motores...")
tts = F5TTS()
modelo_transcritor = whisper.load_model("base") # Modelo leve e rápido
print("✅ Inteligência de áudio pronta!")

def clonar_voz_automatica(texto_para_gerar, audio_ref):
    try:
        if audio_ref is None:
            return None
        
        print("🎧 Ouvindo e transcrevendo o áudio de referência...")
        # O Whisper ouve o áudio e transforma em texto automaticamente
        resultado_transcricao = modelo_transcritor.transcribe(audio_ref)
        texto_detectado = resultado_transcricao["text"].strip()
        print(f"📝 Texto detectado: {texto_detectado}")

        output_file = "saida_miraplay.wav"

        # Agora usamos o texto que o Whisper acabou de gerar
        wav, sr, _ = tts.infer(
            gen_text=texto_para_gerar,
            ref_file=audio_ref,
            ref_text=texto_detectado
        )
        
        sf.write(output_file, wav, sr)
        print(f"✅ Clonagem concluída com sucesso!")
        return output_file

    except Exception as e:
        print(f"💥 Erro no processo: {str(e)}")
        return None

# Interface simplificada (O usuário só precisa de 2 coisas agora!)
app = gr.Interface(
    fn=clonar_voz_automatica,
    inputs=[
        gr.Textbox(label="1. O que a IA deve falar? (Seu Roteiro)"),
        gr.Audio(type="filepath", label="2. Áudio de Referência (A IA vai transcrever sozinha)")
    ],
    outputs=gr.Audio(label="Resultado da Clonagem"),
    title="MIRAPLAY 2026 - IA AUTOMÁTICA",
    description="Agora com transcrição automática via Whisper. Basta subir o áudio e o texto, a IA faz o resto."
)

if __name__ == "__main__":
    app.launch(share=True, debug=True)
