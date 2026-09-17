from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from pathlib import Path
import pickle
import numpy as np

app = FastAPI(
    title="Прогноз стоимости жилья",
    version="3000",
)

MODEL = None
SCALER = None
FEATURE_NAMES = None
CONTINUOUS_COLS = None


@app.on_event("startup")
def load_model():
    global MODEL, SCALER, FEATURE_NAMES, CONTINUOUS_COLS

    try:
        with open("models/model_stub.pkl", "rb") as f:
            MODEL = pickle.load(f)
        with open("models/scaler.pkl", "rb") as f:
            SCALER = pickle.load(f)
        with open("models/feature_names.pkl", "rb") as f:
            FEATURE_NAMES = pickle.load(f)
        with open("models/continuous_cols.pkl", "rb") as f:
            CONTINUOUS_COLS = pickle.load(f)
        print("Модель загружена")
    except FileNotFoundError as e:
        print(f"Не удалось загрузить модель: {e}")

class PredictionInput(BaseModel):
    size: float = Field(..., ge=10, le=500, description="Площадь, м²")
    room_total: int = Field(..., ge=1, le=15, description="Комнат всего")
    building_age: int = Field(..., ge=0, le=50, description="Возраст здания, лет")
    floor_no: int = Field(..., ge=-4, le=30, description="Этаж")
    total_floor_count: int = Field(..., ge=1, le=30, description="Этажей в доме")
    listing_type: int = Field(..., ge=1, le=3, description="Тип объявления (1=Продажа, 2=Аренда)")
    sub_type: str = Field(..., description="Тип недвижимости")
    heating_type: str = Field(..., description="Тип отопления")
    tom: int = Field(..., ge=0, le=180, description="Срок размещения, дней")

    class Config:
        json_schema_extra = {
            "example": {
                "size": 100.0,
                "room_total": 3,
                "building_age": 5,
                "floor_no": 5,
                "total_floor_count": 10,
                "listing_type": 1,
                "sub_type": "Daire",
                "heating_type": "Kombi (Doğalgaz)",
                "tom": 30,
            }
        }


class PredictionOutput(BaseModel):
    price: float = Field(..., description="Прогнозируемая цена, TRY")
    range_low: float = Field(..., description="Нижняя граница диапазона")
    range_high: float = Field(..., description="Верхняя граница диапазона")
    price_per_m2: float = Field(..., description="Цена за м², TRY")
    model_type: str = Field(..., description="Тип использованной модели")




@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": MODEL is not None,
    }


@app.get("/model_info")
def model_info():
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Модель не загружена")
    return {
        "type": type(MODEL).__name__,
        "n_features": len(FEATURE_NAMES) if FEATURE_NAMES else None,
        "features": FEATURE_NAMES,
        "continuous_cols": CONTINUOUS_COLS,
    }


@app.post("/predict", response_model=PredictionOutput)
def predict(data: PredictionInput):
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Модель не загружена")

    # Валидация
    if data.floor_no > data.total_floor_count and data.floor_no > 0:
        raise HTTPException(
            status_code=400,
            detail="Этаж не может быть выше этажности дома"
        )


