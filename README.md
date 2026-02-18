# 🎙️ MIRAPLAY AI - Qwen Pure Intelligence (2026)

Este sistema utiliza a arquitetura de inteligência artificial Qwen para clonagem de voz de ultra-fidelidade. O motor processa automaticamente ritmo, emoção e entonação, eliminando a necessidade de configurações manuais complexas.

## 🚀 Como Executar (Acesso Direto)

Para utilizar o sistema com as 30 horas semanais de GPU gratuita, siga o link oficial abaixo:

[![Abrir no Kaggle](https://img.shields.io/badge/Kaggle-Abrir_Ambiente_de_IA-blue?style=for-the-badge&logo=kaggle)](https://www.kaggle.com/code/new)

---

## 🛠️ Passo a Passo Obrigatório

Como você está em um ambiente profissional de IA, siga estas etapas para o sistema carregar corretamente:

1.  **Login & Verificação:** Certifique-se de estar logado no Kaggle e com o celular verificado em [Settings](https://www.kaggle.com/settings) para liberar o uso da GPU.
2.  **Configurar a Máquina (Lado Direito):**
    * No menu **Settings**, em **Accelerator**, selecione **GPU T4 x2**.
    * Verifique se a opção **Internet** está em **"Internet on"**.
3.  **Executar o Código:**
    * Crie uma nova célula de código (botão `+ Code`).
    * Cole o código de inicialização (disponível abaixo).
    * Clique no ícone de **Play** (Triângulo azul).

---

## 📝 Código para Colar no Kaggle

```python
# 1. Preparação do ambiente
import os
%cd /kaggle/working/
!rm -rf Texto-em-audio

# 2. Conexão com o GitHub
print("🔗 Conectando ao repositório Miraplay...")
!git clone [https://github.com/Miraplay2025/Texto-em-audio.git](https://github.com/Miraplay2025/Texto-em-audio.git)
%cd Texto-em-audio

# 3. Instalação e Início
print("📦 Instalando Motores Qwen (Aguarde 2-3 min)...")
!pip install -r requirements.txt
!python main.py
