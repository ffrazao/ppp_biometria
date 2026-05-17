# Usa uma imagem oficial do Python travada na versão Bookworm (Debian 12 Estável) para evitar quebra de pacotes
FROM python:3.11-slim-bookworm

# Define o nome do usuário da aplicação como argumento configurável
ARG BIOMETRIA_USUARIO=appbiometria
ARG APP_HOME=/home/${BIOMETRIA_USUARIO:-appbiometria}

# Instala as bibliotecas de sistema C++ necessárias para o OpenCV compilar a IA (Pacotes atualizados)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# não trabalhar com o usuário root por segurança, cria um usuário não privilegiado
RUN useradd -m -u 1000 ${BIOMETRIA_USUARIO:-appbiometria}

# Cria o diretório de pesos no home do usuario local
RUN mkdir -p ${APP_HOME}/.deepface/weights
COPY weights/ ${APP_HOME}/.deepface/weights/
RUN chown -R ${BIOMETRIA_USUARIO:-appbiometria}:${BIOMETRIA_USUARIO:-appbiometria} ${APP_HOME}/.deepface

# Prepara o ambiente de trabalho e instala as dependências do Python
WORKDIR ${APP_HOME}/app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Garante que o diretório da aplicação (incluindo logs) pertença ao usuário correto
RUN chown -R ${BIOMETRIA_USUARIO:-appbiometria}:${BIOMETRIA_USUARIO:-appbiometria} ${APP_HOME}/app

# Copia o código fonte
COPY main.py .

# Troca para o usuário não privilegiado para rodar a aplicação
USER ${BIOMETRIA_USUARIO:-appbiometria}

# Expõe a porta do FastAPI
EXPOSE 8000

# Inicia o servidor
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
