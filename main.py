from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from deepface import DeepFace
import base64
import cv2
import numpy as np
import uuid
import os

app = FastAPI(title="Motor Biométrico SEAGRI", version="1.0")

# O modelo Facenet é open source, leve e altamente preciso. Perfeito para a RFC-0003.
MODEL_NAME = "Facenet"

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
            base64_str = base64_str.split(",")[3]
        img_data = base64.b64decode(base64_str)
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Imagem decodificada resultou em None")
        return img
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erro ao processar imagem Base64: {str(e)}")

def save_temp_image(img) -> str:
    """DeepFace trabalha melhor com caminhos de arquivo no disco para otimização em C++"""
    filename = f"/tmp/{uuid.uuid4()}.jpg"
    cv2.imwrite(filename, img)
    return filename

# --- Endpoints da API ---

@app.post("/api/v1/biometria/extract")
async def extract_template(request: ExtractRequest):
    """ Extrai o vetor matemático da face (Embedding) para salvar no banco (RFC-0003) """
    img = base64_to_cv2(request.image_base64)
    img_path = save_temp_image(img)
    
    try:
        # Extrai o template usando a IA
        embedding_objs = DeepFace.represent(img_path=img_path, model_name=MODEL_NAME, enforce_detection=True)
        # O vetor é uma lista de números flutuantes
        embedding = embedding_objs["embedding"] 
        return {"status": "success", "template_vector": embedding}
    except ValueError:
        raise HTTPException(status_code=400, detail="Nenhum rosto humano detectado na imagem.")
    finally:
        if os.path.exists(img_path): os.remove(img_path)

@app.post("/api/v1/biometria/verify")
async def verify_face(request: VerifyRequest):
    """ Compara duas imagens e retorna se é a mesma pessoa e o score de risco """
    img1 = base64_to_cv2(request.image_base64_1)
    img2 = base64_to_cv2(request.image_base64_2)
    
    path1 = save_temp_image(img1)
    path2 = save_temp_image(img2)
    
    try:
        # Executa a comparação facial
        result = DeepFace.verify(img1_path=path1, img2_path=path2, model_name=MODEL_NAME, enforce_detection=True)
        
        # A distância é o quão diferentes os rostos são. Menor distância = Maior similaridade.
        # Nós invertemos para virar um "Score de Similaridade" (0 a 100%)
        distance = result["distance"]
        biometric_score = max(0.0, (1.0 - distance) * 100)
        
        return {
            "status": "success",
            "is_match": result["verified"], # Booleano: É a mesma pessoa?
            "biometric_score": round(biometric_score, 2), # O Score exigido pela RFC-0003
            "distance": round(distance, 4)
        }
    except ValueError:
        raise HTTPException(status_code=400, detail="Rosto não detectado em uma das imagens.")
    finally:
        if os.path.exists(path1): os.remove(path1)
        if os.path.exists(path2): os.remove(path2)

# Código para rodar o servidor se executar o script diretamente
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)