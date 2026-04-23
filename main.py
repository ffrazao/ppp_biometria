from contextlib import asynccontextmanager
import logging
from logging.handlers import RotatingFileHandler
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from deepface import DeepFace
import base64
import cv2
import numpy as np
import uuid
import os

# O modelo Facenet é open source, leve e altamente preciso. Perfeito para a RFC-0003.
MODEL_NAME = "Facenet"
MODEL = None

# Preparação do log do sistema
LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "app.log")
os.makedirs(LOG_DIR, exist_ok=True)

logger = logging.getLogger("motor_biometrico")
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    formatter = logging.Formatter(
        f"%(asctime)s | PID {os.getpid()} | %(levelname)s | %(name)s | %(message)s"
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    file_handler = RotatingFileHandler(
        LOG_FILE, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)


# Carregamento dos modelos biométricos
@asynccontextmanager
async def lifespan(app: FastAPI):

    global MODEL

    logger.info("🔄 Inicializando modelos de IA...")

    try:
        MODEL = DeepFace.build_model(MODEL_NAME)
        logger.info(f"Modelo carregado: {MODEL_NAME}")

        dummy = np.zeros((100, 100, 3), dtype=np.uint8)

        DeepFace.extract_faces(
            dummy,
            detector_backend="retinaface",
            enforce_detection=False
        )

        logger.info("Detector facial inicializado (retinaface)")

    except Exception as e:
        logger.exception(f"Erro ao inicializar modelos: {str(e)}")

    yield

## criação do app
app = FastAPI(
    title="Motor Biométrico SEAGRI",
    version="1.0",
    lifespan=lifespan
)

logger.info("Aplicação iniciando...")
logger.info(f"Modelo biométrico configurado: {MODEL_NAME}")

# --- DTOs de Entrada ---
class ExtractRequest(BaseModel):
    image_base64: str


class VerifyRequest(BaseModel):
    image_base64_1: str  # Foto do cadastro (Referência)
    image_base64_2: str  # Foto tirada no momento do ponto (Captura)


# --- Funções Auxiliares ---
def base64_to_cv2(base64_str: str):
    """Converte a string Base64 do React para uma imagem legível pelo OpenCV"""
    try:
        if "," in base64_str:
            base64_str = base64_str.split(",", 1)[1]
        img_data = base64.b64decode(base64_str)
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Imagem decodificada resultou em None")
        return img
    except Exception as e:
        logger.exception("Erro ao converter imagem Base64 para OpenCV")
        raise HTTPException(
            status_code=400, detail=f"Erro ao processar imagem Base64: {str(e)}"
        )

def get_embedding(img):
    result = DeepFace.represent(
        img_path=img,
        model=MODEL,
        enforce_detection=True,
        detector_backend="retinaface"
    )
    
    if not result:
        raise ValueError("Nenhum rosto detectado")

    # Dependendo da versão do DeepFace, pode vir lista
    if isinstance(result, list):
        return result[0]["embedding"]

    return result["embedding"]


# --- Endpoints da API ---
@app.post("/api/v1/biometria/extract")
async def extract_template(request: ExtractRequest):
    """Extrai o vetor matemático da face (Embedding) para salvar no banco (RFC-0003)"""

    logger.info("Requisição recebida: /api/v1/biometria/extract")

    if MODEL is None:
        logger.error("Modelo não carregado")
        raise HTTPException(status_code=500, detail="Modelo de IA não inicializado")

    img = base64_to_cv2(request.image_base64)

    try:
        # Extrai o template usando a IA
        result = get_embedding(img)

        logger.info("Template facial extraído com sucesso")

        return {"status": "success", "template_vector": result}
    except ValueError as e:
        # Aqui capturamos a mensagem exata da biblioteca
        logger.warning(f"Falha na detecção facial: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Erro de detecção: {str(e)}")
    except Exception:
        logger.exception("Erro inesperado ao extrair template facial")
        raise HTTPException(
            status_code=500, detail="Erro interno ao extrair template facial."
        )

@app.post("/api/v1/biometria/verify")
async def verify_face(request: VerifyRequest):
    """Compara duas imagens e retorna se é a mesma pessoa e o score de risco"""

    logger.info("Requisição recebida: /api/v1/biometria/verify")

    if MODEL is None:
        logger.error("Modelo não carregado")
        raise HTTPException(status_code=500, detail="Modelo de IA não inicializado")

    img1 = base64_to_cv2(request.image_base64_1)
    img2 = base64_to_cv2(request.image_base64_2)

    try:
        # 1. executa a comparação facial
        emb1 = get_embedding(img1)
        emb2 = get_embedding(img2)

        # 2. calcula distância manual (cosine)
        from numpy import dot
        from numpy.linalg import norm

        # A distância é o quão diferentes os rostos são. Menor distância = Maior similaridade.
        # Nós invertemos para virar um "Score de Similaridade" (0 a 100%)
        distance = 1 - (dot(emb1, emb2) / (norm(emb1) * norm(emb2)))

        is_match = distance < 0.4  # threshold típico facenet

        biometric_score = max(0.0, (1.0 - distance) * 100)

        logger.info(
            f"Verificação concluída | match={is_match} | "
            f"score={round(biometric_score, 2)} | distance={round(distance, 4)}"
        )

        return {
            "status": "success",
            "is_match": is_match,
            "biometric_score": round(biometric_score, 2),
            "distance": round(distance, 4),
            "tecnico_falha": False,
        }

    except ValueError as e:
        logger.warning(f"Rosto não detectado: {str(e)}")
        # Se cair aqui, é porque o DeepFace não achou rosto (enforce_detection=True)
        return {
            "status": "warning",
            "is_match": False,
            "biometric_score": 0.0,
            "detail": "Rosto não identificado",
            "tecnico_falha": True,
        }

    except Exception as e:
        logger.error(f"Erro técnico na verificação: {str(e)}")
        return {
            "status": "warning",
            "is_match": False,
            "biometric_score": 0.0,
            "detail": f"Erro no processamento da imagem: {str(e)}",
            "tecnico_falha": True,
        }

# Código para rodar o servidor se executar o script diretamente
if __name__ == "__main__":
    import uvicorn

    logger.info("Subindo servidor Uvicorn...")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
