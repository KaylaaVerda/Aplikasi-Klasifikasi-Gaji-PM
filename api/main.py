from fastapi import FastAPI, HTTPException

from api.inference import predict
from api.schemas import PredictRequest, PredictResponse

app = FastAPI(title="Klasifikasi Gaji API")


@app.get("/")
def root() -> dict:
    return {"message": "Klasifikasi Gaji API. Gunakan POST /predict untuk prediksi."}


@app.post("/predict", response_model=PredictResponse)
def predict_salary(request: PredictRequest) -> PredictResponse:
    try:
        result = predict(request.model_dump())
        return PredictResponse(**result)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediksi gagal: {exc}") from exc
