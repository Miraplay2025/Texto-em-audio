import os
import soundfile as sf
import whisper
import gradio as gr
from f5_tts.api import E2TTS  # Mudamos de F5TTS para E2TTS

# Inicializando a IA com o modelo E2 (Melhor para Português)
print("🚀 Miraplay AI: Ativando motor E2-TTS (Multilinguagem)...")
tts = E2TTS() 
modelo_transcritor = whisper.load_model("base")

def clonar_voz_miraplay(texto_para_gerar, audio_ref):
    try:
        if audio_ref is None:
            return None
        
        # 1. Transcrição automática (Dando a pista de que é Português)
        print(f"🎧 Analisando áudio...")
        resultado = modelo_transcritor.transcribe(audio_ref, language="pt")
        texto_detectado = resultado["text"].strip()
        print(f"📝 Texto da amostra: {texto_detectado}")

        output_file = "saida_miraplay.wav"

        # 2. Geração usando E2TTS
        # Este modelo lida melhor com a gramática e acentuação do Português
        wav, sr, _ = tts.infer(
            gen_text=texto_para_gerar,
            ref_file=audio_ref,
            ref_text=texto_detectado
        )
        
        sf.write(output_file, wav, sr)
        print(f"✅ Clonagem concluída com sucesso!")
        return output_file

    except Exception as e:
        print(f"💥 Erro: {str(e)}")
        return None

# Interface simplificada e direta
app = gr.Interface(
    fn=clonar_voz_miraplay,
    inputs=[
        gr.Textbox(label="O que a IA deve falar (Em Português)", placeholder="Escreva aqui..."),
        gr.Audio(type="filepath", label="Sua voz de referência")
    ],
    outputs=gr.Audio(label="Áudio Final (Voz Clonada)"),
    title="MIRAPLAY 2026 - MODO BRASIL",
    description="Utilizando motor E2-TTS para melhor suporte a idiomas latinos."
)

if __name__ == "__main__":
    app.launch(share=True, debug=True)
