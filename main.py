from contextlib import asynccontextmanager
import logging
from logging.handlers import RotatingFileHandler
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
import numpy as np
from deepface import DeepFace
import cv2
import os

# ---------------------------------------------------------------------------
# Configuração do modelo
# ---------------------------------------------------------------------------
MODEL_NAME = "Facenet"
MODEL = None

# ---------------------------------------------------------------------------
# Preparação do log
# ---------------------------------------------------------------------------
LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "app.log")
os.makedirs(LOG_DIR, exist_ok=True)

logger = logging.getLogger("motor_biometrico")
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    formatter = logging.Formatter(
        "%(asctime)s | PID %(process)d | %(levelname)s | %(name)s | %(message)s"
    )
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    file_handler = RotatingFileHandler(
        LOG_FILE, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

# ---------------------------------------------------------------------------
# Lifespan — carrega os modelos uma única vez na inicialização
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global MODEL

    logger.info("🔄 Inicializando modelos de IA...")
    try:
        MODEL = DeepFace.build_model(MODEL_NAME)
        logger.info(f"Modelo carregado: {MODEL_NAME}")

        dummy = np.zeros((100, 100, 3), dtype=np.uint8)
        DeepFace.extract_faces(dummy, detector_backend="retinaface", enforce_detection=False)
        logger.info("Detector facial inicializado (retinaface)")

    except Exception as e:
        logger.exception(f"Erro ao inicializar modelos: {e}")

    yield

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="Motor Biométrico SEAGRI", version="1.0", lifespan=lifespan)

logger.info("Iniciando a Aplicação...")
logger.info(f"Modelo biométrico configurado: {MODEL_NAME}")

# ---------------------------------------------------------------------------
# DTOs
# ---------------------------------------------------------------------------
class CompareTemplatesRequest(BaseModel):
    """Corpo da requisição para comparação de dois vetores já extraídos."""
    template_1: list[float]
    template_2: list[float]

class CompareTemplateFileRequest(BaseModel):
    """Vetor já extraído enviado junto com um arquivo de imagem."""
    template_1: list[float]

# ---------------------------------------------------------------------------
# Funções auxiliares internas (não são endpoints — sem acoplamento ao FastAPI)
# ---------------------------------------------------------------------------
def _bytes_to_cv2(raw_bytes: bytes) -> np.ndarray:
    """Converte bytes brutos em imagem OpenCV."""
    nparr = np.frombuffer(raw_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Imagem inválida ou corrompida")
    return img

def _extract_embedding(img: np.ndarray) -> list[float]:
    """Extrai o vetor biométrico de uma imagem OpenCV."""
    if MODEL is None:
        raise HTTPException(status_code=500, detail="Modelo de IA não inicializado")

    result = DeepFace.represent(
        img_path=img,
        model_name=MODEL_NAME,
        detector_backend="retinaface",
        enforce_detection=True,
    )

    if not result:
        raise ValueError("Nenhum rosto detectado!")

    if isinstance(result, list):
        return result[0]["embedding"]

    return result["embedding"]

def _cosine_similarity(emb1: list[float], emb2: list[float]) -> dict:
    """Calcula similaridade por distância cosseno entre dois vetores."""
    v1 = np.array(emb1)
    v2 = np.array(emb2)

    distance = float(1 - (np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))))
    is_match = distance < 0.4
    biometric_score = float(round(max(0.0, (1.0 - distance) * 100), 2))

    logger.info(
        f"Comparação concluída | match={is_match} | "
        f"score={biometric_score} | distance={round(distance, 4)}"
    )

    return {
        "status": "success",
        "is_match": is_match,
        "biometric_score": biometric_score,
        "distance": round(distance, 4),
        "tecnico_falha": False,
    }

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/api/v1/biometria/extract")
async def extract_template(image: UploadFile = File(...)):
    """Recebe um arquivo de imagem e devolve o vetor biométrico (embedding)."""
    logger.info("Requisição recebida: /api/v1/biometria/extract")

    try:
        img = _bytes_to_cv2(await image.read())
        embedding = _extract_embedding(img)
        logger.info("Template facial extraído com sucesso")
        return {"status": "success", "template_vector": embedding}

    except HTTPException:
        raise

    except ValueError as e:
        logger.warning(f"Falha na detecção facial: {e}")
        raise HTTPException(status_code=400, detail=f"Falha na detecção facial: {e}")

    except Exception as e:
        logger.exception(f"Erro técnico ao extrair template: {e}")
        raise HTTPException(status_code=500, detail=f"Erro técnico ao extrair template: {e}")


@app.post("/api/v1/biometria/compare_file")
async def compare_file(
    image_1: UploadFile = File(...),
    image_2: UploadFile = File(...),
):
    """Recebe dois arquivos de imagem e devolve o score de similaridade."""
    logger.info("Requisição recebida: /api/v1/biometria/compare_file")

    try:
        img1 = _bytes_to_cv2(await image_1.read())
        img2 = _bytes_to_cv2(await image_2.read())

        emb1 = _extract_embedding(img1)
        emb2 = _extract_embedding(img2)

        return _cosine_similarity(emb1, emb2)

    except HTTPException:
        raise

    except ValueError as e:
        logger.warning(f"Rosto não detectado: {e}")
        return {
            "status": "warning",
            "is_match": False,
            "biometric_score": 0.0,
            "detail": f"Falha na detecção facial: {e}",
            "tecnico_falha": True,
        }

    except Exception as e:
        logger.exception(f"Erro técnico na comparação: {e}")
        return {
            "status": "warning",
            "is_match": False,
            "biometric_score": 0.0,
            "detail": f"Erro técnico na comparação: {e}",
            "tecnico_falha": True,
        }


@app.post("/api/v1/biometria/compare_template")
async def compare_template(
    template_1: str = Form(...),   # vetor JSON enviado como campo de formulário
    image_2: UploadFile = File(...),
):
    """Recebe um vetor biométrico já extraído e um arquivo de imagem e devolve o score."""
    logger.info("Requisição recebida: /api/v1/biometria/compare_template")

    import json
    try:
        emb1 = json.loads(template_1)
        if not isinstance(emb1, list):
            raise HTTPException(status_code=400, detail="template_1 deve ser uma lista JSON de floats")

        img2 = _bytes_to_cv2(await image_2.read())
        emb2 = _extract_embedding(img2)

        return _cosine_similarity(emb1, emb2)

    except HTTPException:
        raise

    except ValueError as e:
        logger.warning(f"Rosto não detectado: {e}")
        return {
            "status": "warning",
            "is_match": False,
            "biometric_score": 0.0,
            "detail": f"Falha na detecção facial: {e}",
            "tecnico_falha": True,
        }

    except Exception as e:
        logger.exception(f"Erro técnico no compare_template: {e}")
        return {
            "status": "warning",
            "is_match": False,
            "biometric_score": 0.0,
            "detail": f"Erro técnico: {e}",
            "tecnico_falha": True,
        }


@app.post("/api/v1/biometria/compare")
async def compare(body: CompareTemplatesRequest):
    """Recebe dois vetores biométricos já extraídos e devolve o score de similaridade."""
    logger.info("Requisição recebida: /api/v1/biometria/compare")

    try:
        return _cosine_similarity(body.template_1, body.template_2)

    except Exception as e:
        logger.exception(f"Erro técnico na comparação de vetores: {e}")
        return {
            "status": "warning",
            "is_match": False,
            "biometric_score": 0.0,
            "detail": f"Erro técnico: {e}",
            "tecnico_falha": True,
        }


# ---------------------------------------------------------------------------
# Execução direta
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    logger.info("Subindo servidor Uvicorn...")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
