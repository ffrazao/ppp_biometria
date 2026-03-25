# Usa uma imagem oficial do Python travada na versão Bookworm (Debian 12 Estável) para evitar quebra de pacotes
FROM python:3.10-slim-bookworm

# Instala as bibliotecas de sistema C++ necessárias para o OpenCV compilar a IA (Pacotes atualizados)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Adicionar para reduzir logs do TensorFlow
ENV TF_CPP_MIN_LOG_LEVEL=2
ENV CUDA_VISIBLE_DEVICES=-1

# Copia e instala as dependências
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia o código fonte
COPY main.py .

# Força o DeepFace a baixar os pesos da rede neural na build para inicializar instantaneamente depois
RUN python -c "from deepface import DeepFace; DeepFace.build_model('Facenet')"

# Expõe a porta do FastAPI
EXPOSE 8000

# Inicia o servidor
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]