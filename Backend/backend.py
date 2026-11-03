import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, HTTPException

# Import our custom ML functions from preprocessing.py
from preprocessing import load_artifacts, preprocess_image_bytes, generate_caption

# Global variables to hold the loaded model and tokenizer
Transformer_Model = None
tokenizer = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Modern FastAPI lifespan handler: runs before server starts and cleans up on shutdown."""
    global Transformer_Model, tokenizer
    # Startup: Load model and tokenizer once into memory
    Transformer_Model, tokenizer = load_artifacts()
    yield
    # Shutdown: Clean up resources when stopping the server
    print("[INFO] Shutting down LiveSight API...")


app = FastAPI(title="LiveSight Image Captioning API", lifespan=lifespan)


@app.get("/health")
def health_check():
    """Simple check to verify the server is running."""
    return {"status": "online", "model_loaded": Transformer_Model is not None}

@app.get("/")
def welcome():
    return "Hello yor are on right track:)"

@app.post("/predict")
async def predict_caption(file: UploadFile = File(...)):
    """Receives jpeg_bytes from imageCapturing.py and returns the generated caption."""
    try:
        start_time = time.perf_counter()
        raw_bytes = await file.read()

        # 1. Preprocess JPEG bytes into (224, 224, 3) tensor
        img_tensor = preprocess_image_bytes(raw_bytes)

        # 2. Generate caption using the Transformer model
        caption = generate_caption(Transformer_Model, img_tensor, tokenizer)
        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

        return {
            "status": "success",
            "caption": caption,
            "inference_time_ms": latency_ms
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))