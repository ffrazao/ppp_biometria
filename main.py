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

# =========================
# CONFIGURAÇÃO DE LOG
# =========================
LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "app.log")

os.makedirs(LOG_DIR, exist_ok=True)

logger = logging.getLogger("motor_biometrico")
logger.setLevel(logging.INFO)

# Evita handlers duplicados em reload/restart
if not logger.handlers:
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    # Console (aparece no docker logs)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # Arquivo com rotação
    file_handler = RotatingFileHandler(
        LOG_FILE,
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5,
        encoding="utf-8"
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

app = FastAPI(title="Motor Biométrico SEAGRI", version="1.0")

# O modelo Facenet é open source, leve e altamente preciso. Perfeito para a RFC-0003.
MODEL_NAME = "Facenet"

logger.info("Aplicação iniciando...")
logger.info(f"Modelo biométrico configurado: {MODEL_NAME}")

# --- DTOs de Entrada ---
class ExtractRequest(BaseModel):
    image_base64: str

class VerifyRequest(BaseModel):
    image_base64_1: str # Foto do cadastro (Referência)
    image_base64_2: str # Foto tirada no momento do ponto (Captura)

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
        raise HTTPException(status_code=400, detail=f"Erro ao processar imagem Base64: {str(e)}")

def save_temp_image(img) -> str:
    """DeepFace trabalha melhor com caminhos de arquivo no disco para otimização em C++"""
    filename = f"/tmp/{uuid.uuid4()}.jpg"
    cv2.imwrite(filename, img)
    logger.debug(f"Imagem temporária salva: {filename}")
    return filename

# --- Endpoints da API ---

@app.post("/api/v1/biometria/extract")
async def extract_template(request: ExtractRequest):
    """ Extrai o vetor matemático da face (Embedding) para salvar no banco (RFC-0003) """
    logger.info("Requisição recebida: /api/v1/biometria/extract")

    img = base64_to_cv2(request.image_base64)
    img_path = save_temp_image(img)
    
    try:
        # Extrai o template usando a IA
        embedding_objs = DeepFace.represent(
            img_path=img_path,
            model_name=MODEL_NAME,
            enforce_detection=True
        )
        # Dependendo da versão do DeepFace, pode vir lista
        if isinstance(embedding_objs, list):
            embedding = embedding_objs[0]["embedding"]
        else:
            embedding = embedding_objs["embedding"]
        logger.info("Template facial extraído com sucesso")
        return {"status": "success", "template_vector": embedding}
    except ValueError:
        logger.warning("Nenhum rosto detectado no endpoint extract")
        raise HTTPException(status_code=400, detail="Nenhum rosto humano detectado na imagem.")
    except Exception:
        logger.exception("Erro inesperado ao extrair template facial")
        raise HTTPException(status_code=500, detail="Erro interno ao extrair template facial.")
    finally:
        if os.path.exists(img_path):
            os.remove(img_path)
            logger.debug(f"Arquivo temporário removido: {img_path}")

@app.post("/api/v1/biometria/verify")
async def verify_face(request: VerifyRequest):
    """ Compara duas imagens e retorna se é a mesma pessoa e o score de risco """
    
    logger.info("Requisição recebida: /api/v1/biometria/verify")

    img1 = base64_to_cv2(request.image_base64_1)
    img2 = base64_to_cv2(request.image_base64_2)
    
    path1 = save_temp_image(img1)
    path2 = save_temp_image(img2)
    
    try:
        # Executa a comparação facial
        result = DeepFace.verify(
            img1_path=path1, 
            img2_path=path2, 
            model_name=MODEL_NAME, 
            enforce_detection=True
        )
        
        # A distância é o quão diferentes os rostos são. Menor distância = Maior similaridade.
        # Nós invertemos para virar um "Score de Similaridade" (0 a 100%)
        distance = result["distance"]
        biometric_score = max(0.0, (1.0 - distance) * 100)

        logger.info(
            f"Verificação concluída | match={result['verified']} | "
            f"score={round(biometric_score, 2)} | distance={round(distance, 4)}"
        )
        
        return {
            "status": "success",
            "is_match": result["verified"], # Booleano: É a mesma pessoa?
            "biometric_score": round(biometric_score, 2), # O Score exigido pela RFC-0003
            "distance": round(distance, 4)
        }
    except ValueError:
        logger.warning("Rosto não detectado em uma das imagens no endpoint verify")
        raise HTTPException(status_code=400, detail="Rosto não detectado em uma das imagens.")
    except Exception:
        logger.exception("Erro inesperado na verificação facial")
        raise HTTPException(status_code=500, detail="Erro interno ao verificar biometria.")
    finally:
        if os.path.exists(path1):
            os.remove(path1)
            logger.debug(f"Arquivo temporário removido: {path1}")
        if os.path.exists(path2):
            os.remove(path2)
            logger.debug(f"Arquivo temporário removido: {path2}")

# Código para rodar o servidor se executar o script diretamente
if __name__ == "__main__":
    import uvicorn
    logger.info("Subindo servidor Uvicorn...")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
