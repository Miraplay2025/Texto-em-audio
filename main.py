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

# AJUSTE: Removido model_type para evitar erro de inicialização
print("🚀 Iniciando Servidor de IA...")
tts = F5TTS() 

def send_log(message, type="info"):
    """Envia um log em tempo real para o HTML via Socket"""
    try:
        socketio.emit('log_message', {'msg': message, 'type': type})
    except:
        pass
    print(f"LOG: {message}")

def processar_clonagem(texto, audio_ref):
    try:
        send_log("📥 Pedido recebido pelo servidor.", "success")
        
        if audio_ref is None:
            send_log("❌ Erro: Áudio de referência não enviado.", "error")
            return None
        
        send_log(f"📝 Texto processando: '{texto[:30]}...'", "info")
        send_log("🧠 IA processando clonagem (GPU Ativa)...", "warning")
        
        output_path = "output_realtime.wav"
        
        # Inferência com os parâmetros corretos da versão atual
        tts.infer(
            gen_text=texto,
            ref_file=audio_ref,
            output_file=output_path
        )
        
        send_log("✅ Áudio gerado com sucesso!", "success")
        return output_path
    except Exception as e:
        send_log(f"💥 Erro interno: {str(e)}", "error")
        return None

# Interface Gradio
demo = gr.Interface(
    fn=processar_clonagem,
    inputs=[gr.Textbox(label="Texto"), gr.Audio(type="filepath", label="Voz de Referência")],
    outputs=gr.Audio(label="Resultado"),
)

def run_socket():
    # Rodando em porta diferente para não conflitar com o túnel do Gradio
    socketio.run(app, port=5001, host='0.0.0.0', debug=False, use_reloader=False)

# Thread para o SocketIO
threading.Thread(target=run_socket, daemon=True).start()

# Lança o Gradio com link público
demo.launch(share=True)
