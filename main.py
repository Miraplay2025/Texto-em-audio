import os
import torch
import re
import shutil
from TTS.api import TTS
import gradio as gr

# 🚀 CONFIGURAÇÃO DE ALTA FIDELIDADE
os.environ["COQUI_TOS_AGREED"] = "1"
device = "cuda" if torch.cuda.is_available() else "cpu"

print("📥 Carregando motor de Clonagem de Elite (Aguarde o link)...")
try:
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
    print("✅ Motor carregado com sucesso!")
except Exception as e:
    print(f"❌ Erro ao carregar motor: {e}")

def limpar_texto_para_ia(texto):
    # Remove excesso de pontos e símbolos que a IA tenta 'falar'
    # Converte múltiplos pontos em apenas um para pausa de respiração
    texto = re.sub(r'\.{2,}', '.', texto) 
    # Remove caracteres especiais desnecessários
    texto = re.sub(r'[#\*_~]', '', texto)
    return texto.strip()

def clonar_voz_total(texto, audio_ref, estilo_personalizado):
    try:
        if audio_ref is None: return None
        
        # Limpeza para evitar que a IA diga "ponto" ou faça ruídos nos símbolos
        texto_final = limpar_texto_para_ia(texto)
        
        output_path = "resultado_miraplay_clone.wav"
        
        # Estilo: Se o usuário não digitar, foca em ser um ESPELHO do áudio
        prompt_estilo = estilo_personalizado if estilo_personalizado else "Extremely natural speech, mirror the reference audio pace and tone perfectly."

        tts.tts_to_file(
            text=texto_final,
            speaker_wav=audio_ref,
            language="pt",
            file_path=output_path,
            # CONFIGURAÇÃO DE ESPELHAMENTO (ELEVEN LABS LEVEL)
            temperature=0.70,      
            top_p=0.85,           
            speed=1.0,            # 1.0 = Segue o ritmo original da pessoa
            repetition_penalty=2.5, # Segurança extra contra robótica
            gpt_cond_len=30,      # Analisa 30 segundos de 'jeito de falar'
            emotion=prompt_estilo
        )
        return output_path
    except Exception as e:
        print(f"💥 Erro: {e}")
        return None

# --- INTERFACE MODERNA ---
meu_tema = gr.themes.Soft(primary_hue="blue", neutral_hue="slate")

with gr.Blocks(theme=meu_tema, title="MIRAPLAY CLONE 1:1") as app:
    gr.Markdown("# 🎙️ MIRAPLAY AI - Clone Idêntico")
    gr.Markdown("### 💎 Foco: Copiar 100% da cadência e detalhes do áudio original.")
    
    with gr.Row():
        with gr.Column():
            input_text = gr.Textbox(
                label="Texto para a IA ler", 
                placeholder="Escreva o texto aqui. A IA usará os pontos apenas para respirar.",
                lines=6
            )
            input_estilo = gr.Textbox(
                label="Estilo Opcional (Ex: Alegre, Rápido, Triste)",
                placeholder="Deixe vazio para seguir 100% o áudio de referência.",
            )
            input_audio = gr.Audio(
                label="Áudio de Referência (O 'DNA' da voz)", 
                type="filepath",
                sources=["upload", "microphone"]
            )
            btn = gr.Button("🚀 GERAR CLONE COMPLETO", variant="primary")
            
        with gr.Column():
            output_audio = gr.Audio(label="Resultado (Voz + Ritmo Copiados)")
            gr.Markdown("""
            ### 💡 Dicas para o Clone Perfeito:
            1. **Referência:** Use um áudio de 15 segundos onde a pessoa fala com clareza.
            2. **Pausas:** Se a IA falar muito corrido, coloque uma vírgula.
            3. **Sem Robótica:** O sistema agora ignora símbolos repetidos para não gaguejar.
            """)

    btn.click(
        fn=clonar_voz_total, 
        inputs=[input_text, input_audio, input_estilo], 
        outputs=output_audio
    )

if __name__ == "__main__":
    # app.queue() ajuda a manter a conexão estável no Colab
    app.queue().launch(share=True, debug=True, show_error=True)
