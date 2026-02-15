import os
import soundfile as sf  # Biblioteca para salvar o áudio corretamente
from f5_tts.api import F5TTS
import gradio as gr

# Carregamento da IA
print("🚀 Iniciando Servidor de IA Miraplay...")
tts = F5TTS()

def processar_clonagem(texto, audio_ref):
    try:
        print(f"📥 Processando texto: {texto[:30]}...")
        
        if audio_ref is None:
            return None
        
        # O NOME DO ARQUIVO FINAL
        caminho_saida = "resultado_miraplay.wav"
        
        # A nova forma de chamar: a IA devolve 3 coisas (o audio, a frequencia e os detalhes)
        # Removi o 'output_path' daqui porque a nova versão não aceita mais
        wav, sr, _ = tts.infer(
            gen_text=texto,
            ref_file=audio_ref
        )
        
        # Agora nós salvamos o arquivo manualmente usando soundfile
        sf.write(caminho_saida, wav, sr)
        
        print(f"✅ Áudio gerado com sucesso!")
        return caminho_saida

    except Exception as e:
        print(f"💥 Erro técnico: {str(e)}")
        return None

# Interface do site
demo = gr.Interface(
    fn=processar_clonagem,
    inputs=[
        gr.Textbox(label="Texto para a IA falar"), 
        gr.Audio(type="filepath", label="Sua voz de referência")
    ],
    outputs=gr.Audio(label="Áudio Clonado"),
    title="MIRAPLAY 2026",
    description="Sistema profissional de clonagem de voz via API."
)

if __name__ == "__main__":
    demo.launch(share=True, debug=True)
