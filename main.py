import os
from f5_tts.api import F5TTS
import gradio as gr

# Carregamento da IA (Versão 2026 estável)
print("🚀 Iniciando Servidor de IA...")
try:
    # Inicialização sem parâmetros extras para evitar erro de TypeError
    tts = F5TTS() 
    print("✅ Modelo carregado com sucesso!")
except Exception as e:
    print(f"❌ Erro ao carregar o modelo: {str(e)}")

def processar_clonagem(texto, audio_ref):
    try:
        print(f"📥 Recebido texto: {texto[:30]}...")
        
        if audio_ref is None:
            print("⚠️ Aviso: Nenhum áudio de referência foi enviado.")
            return None
        
        # Nome do arquivo de saída
        caminho_saida = "voz_clonada_miraplay.wav"
        
        print("🧠 IA processando a voz... Aguarde.")
        
        # CHAMADA CORRIGIDA: Usando 'output_path' em vez de 'output_file'
        tts.infer(
            gen_text=texto,
            ref_file=audio_ref,
            output_path=caminho_saida
        )
        
        print(f"✅ Sucesso! Áudio gerado em: {caminho_saida}")
        return caminho_saida

    except Exception as e:
        print(f"💥 Erro durante a inferência: {str(e)}")
        return None

# Interface Gradio (A moldura que aparece no seu site)
demo = gr.Interface(
    fn=processar_clonagem,
    inputs=[
        gr.Textbox(label="Texto para a IA falar", placeholder="Digite aqui o roteiro..."), 
        gr.Audio(type="filepath", label="Voz de referência (Upload do áudio)")
    ],
    outputs=gr.Audio(label="Áudio Final Clonado"),
    title="MIRAPLAY 2026 - Sistema de Clonagem Neural",
    description="Interface conectada ao Google Colab T4 GPU."
)

# Lança o servidor com link público
# O link .gradio.live que aparecer no Colab deve ser colocado no seu index.html
if __name__ == "__main__":
    demo.launch(share=True, debug=True)
