import os
import soundfile as sf
import whisper
import gradio as gr
from f5_tts.api import F5TTS # Usaremos a classe base que o sistema reconhece

# Inicializando a IA forçando o modelo E2 (Mais multilingue)
print("🚀 Miraplay AI: Ativando motor E2-TTS...")
try:
    # Mudamos o model_type para 'e2' aqui dentro
    tts = F5TTS(model_type="e2") 
    print("✅ Motor E2 carregado!")
except:
    # Caso a versão seja muito antiga e não aceite o parâmetro, ele usa o padrão
    tts = F5TTS()
    print("✅ Motor F5 carregado (Modo Compatibilidade)")

modelo_transcritor = whisper.load_model("base")

def clonar_voz_miraplay(texto_para_gerar, audio_ref):
    try:
        if audio_ref is None:
            return None
        
        print(f"🎧 Analisando áudio com Whisper...")
        # Forçamos o Whisper a entender que o áudio de referência é PT
        resultado = modelo_transcritor.transcribe(audio_ref, language="pt")
        texto_detectado = resultado["text"].strip()
        print(f"📝 Texto detectado: {texto_detectado}")

        output_file = "saida_miraplay.wav"

        # Geração do áudio
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

app = gr.Interface(
    fn=clonar_voz_miraplay,
    inputs=[
        gr.Textbox(label="O que a IA deve falar (Use acentos: á, é, í, õ)"),
        gr.Audio(type="filepath", label="Áudio de Referência (Voz da pessoa)")
    ],
    outputs=gr.Audio(label="Áudio Final"),
    title="MIRAPLAY 2026 - MODO BRASIL",
    description="Sistema atualizado para evitar sotaque estrangeiro."
)

if __name__ == "__main__":
    app.launch(share=True, debug=True)
