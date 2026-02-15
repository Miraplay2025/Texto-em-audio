import os
import soundfile as sf
from f5_tts.api import F5TTS
import gradio as gr

# Inicializando a Inteligência Artificial
print("🚀 Miraplay AI: Carregando motores neurais...")
tts = F5TTS()

def clonar_voz_miraplay(texto, audio_ref):
    try:
        print(f"🎤 Nova solicitação de narração recebida...")
        
        if audio_ref is None:
            return None
        
        # Nome do arquivo temporário
        output_file = "voz_final.wav"
        
        # A IA gera os dados do áudio (wav) e a frequência (sr)
        wav, sr, _ = tts.infer(
            gen_text=texto,
            ref_file=audio_ref
        )
        
        # Salvando o arquivo de áudio no disco do Colab
        sf.write(output_file, wav, sr)
        
        print(f"✅ Áudio gerado com sucesso!")
        return output_file

    except Exception as e:
        print(f"💥 Erro na geração: {str(e)}")
        return None

# Configuração da Interface (Visual que aparece no seu Iframe)
app = gr.Interface(
    fn=clonar_voz_miraplay,
    inputs=[
        gr.Textbox(label="Roteiro da Narração", placeholder="Escreva o que a voz deve dizer aqui..."),
        gr.Audio(type="filepath", label="Voz de Referência (Amostra)")
    ],
    outputs=gr.Audio(label="Resultado da Clonagem"),
    title="MIRAPLAY 2026",
    theme=gr.themes.Soft()
)

if __name__ == "__main__":
    # O share=True cria o link .gradio.live que você cola no InfinityFree
    app.launch(share=True, debug=True)
