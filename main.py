import os
import soundfile as sf
import whisper
from f5_tts.api import F5TTS
import gradio as gr

# Carregando os modelos
print("🚀 Miraplay AI: Ajustando para múltiplos idiomas...")
tts = F5TTS()
modelo_transcritor = whisper.load_model("base")

def clonar_voz_miraplay(texto_para_gerar, audio_ref, idioma):
    try:
        if audio_ref is None:
            return None
        
        # 1. Transcrição automática com dica de idioma para o Whisper
        print(f"🎧 Transcrevendo áudio em {idioma}...")
        # Traduzindo a escolha para o código que o Whisper entende
        lang_code = "pt" if idioma == "Português" else "en"
        
        resultado = modelo_transcritor.transcribe(audio_ref, language=lang_code)
        texto_detectado = resultado["text"].strip()
        print(f"📝 Texto da amostra: {texto_detectado}")

        output_file = "saida_miraplay.wav"

        # 2. Inferência da IA
        # O F5-TTS usa o ref_text para captar a cadência do idioma escolhido
        wav, sr, _ = tts.infer(
            gen_text=texto_para_gerar,
            ref_file=audio_ref,
            ref_text=texto_detectado
        )
        
        sf.write(output_file, wav, sr)
        print(f"✅ Clonagem em {idioma} concluída!")
        return output_file

    except Exception as e:
        print(f"💥 Erro: {str(e)}")
        return None

# Interface com Seleção de Idioma
app = gr.Interface(
    fn=clonar_voz_miraplay,
    inputs=[
        gr.Textbox(label="1. Texto que a IA vai falar"),
        gr.Audio(type="filepath", label="2. Áudio de Referência"),
        gr.Dropdown(
            choices=["Português", "Inglês"], 
            value="Português", 
            label="3. Selecione o Idioma"
        )
    ],
    outputs=gr.Audio(label="Resultado da Clonagem"),
    title="MIRAPLAY 2026 - MULTI-IDIOMAS",
    description="Agora você pode forçar a IA a falar no idioma correto selecionando acima."
)

if __name__ == "__main__":
    app.launch(share=True, debug=True)
