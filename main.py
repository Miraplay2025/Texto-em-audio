import os
import eventlet
eventlet.monkey_patch()

from flask import Flask
from flask_socketio import SocketIO, emit
from f5_tts.api import F5TTS
import gradio as gr
import threading

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# Carregamento da IA
print("🚀 Iniciando Servidor de IA...")
tts = F5TTS(model_type="F5-TTS")

def send_log(message, type="info"):
    """Envia um log em tempo real para o HTML via Socket"""
    socketio.emit('log_message', {'msg': message, 'type': type})
    print(f"LOG: {message}")

def processar_clonagem(texto, audio_ref):
    try:
        send_log("📥 Pedido recebido pelo servidor.", "success")
        
        send_log("⚙️ Analisando áudio de referência...", "info")
        if audio_ref is None:
            send_log("❌ Erro: Áudio de referência não enviado.", "error")
            return None
        
        send_log(f"📝 Texto detectado: '{texto[:30]}...'", "info")
        
        send_log("🧠 IA processando clonagem (isso usa 100% da GPU)...", "warning")
        output_path = "output_realtime.wav"
        
        # Inferência Real
        tts.infer(
            gen_text=texto,
            ref_file=audio_ref,
            output_file=output_path
        )
        
        send_log("✅ Áudio gerado com sucesso! Enviando para o cliente...", "success")
        return output_path
    except Exception as e:
        send_log(f"💥 Erro interno: {str(e)}", "error")
        return None

# Interface Gradio para servir como túnel e API
demo = gr.Interface(
    fn=processar_clonagem,
    inputs=[gr.Textbox(), gr.Audio(type="filepath")],
    outputs=gr.Audio(),
)

def run_socket():
    socketio.run(app, port=5001, host='0.0.0.0')

# Inicia o Socket em uma thread separada
threading.Thread(target=run_socket).start()

# Lança o Gradio (o link público que você usará)
demo.launch(share=True)
