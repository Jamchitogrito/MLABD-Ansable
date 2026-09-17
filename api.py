from contextlib import asynccontextmanager
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL = None
SCALER = None
FEATURE_NAMES = None
CONTINUOUS_COLS = None

MODEL_PATH = Path("final_real_estate_model.pkl")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global MODEL, SCALER, FEATURE_NAMES, CONTINUOUS_COLS
    with open(MODEL_PATH, "rb") as f:
        bundle = pickle.load(f)
    MODEL = bundle["model"]
    FEATURE_NAMES = bundle["feature_names"]
    SCALER = bundle.get("scaler")
    CONTINUOUS_COLS = bundle.get("continuous_cols", [])
    print(f"Модель загружена: {type(MODEL).__name__}, фич: {len(FEATURE_NAMES)}")
    yield


app = FastAPI(title="Прогноз стоимости жилья", version="1.0", lifespan=lifespan)


class PredictionInput(BaseModel):
    size: float = Field(..., ge=10, le=1000)
    room_total: int = Field(..., ge=1, le=15)
    building_age: int = Field(..., ge=0, le=100)
    floor_no: int = Field(..., ge=-4, le=50)
    total_floor_count: int = Field(..., ge=1, le=50)
    listing_type: int = Field(..., ge=1, le=3)
    subtype: str
    heating: str
    tom: int = Field(..., ge=0, le=365)

    model_config = {
        "json_schema_extra": {
            "example": {
                "size": 100.0, "room_total": 3, "building_age": 5,
                "floor_no": 5, "total_floor_count": 10, "listing_type": 1,
                "subtype": "Daire", "heating": "Kombi (Doğalgaz)", "tom": 30,
            }
        }
    }


class PredictionOutput(BaseModel):
    price: float
    range_low: float
    range_high: float
    price_per_m2: float
    model_type: str


def build_features(data: PredictionInput) -> pd.DataFrame:
    row = {
        "listing_type": data.listing_type,
        "tom": data.tom,
        "building_age": data.building_age,
        "total_floor_count": data.total_floor_count,
        "floor_no": data.floor_no,
        "size_log": float(np.log1p(data.size)),
        "room_total": data.room_total,
        "floor_ratio": data.floor_no / data.total_floor_count if data.total_floor_count else 0.0,
        "is_ground_floor": int(data.floor_no == 0),
        "is_top_floor": int(data.floor_no == data.total_floor_count),
        "size_per_room": data.size / data.room_total if data.room_total else 0.0,
        "age_x_size": data.building_age * data.size,
        "rooms_per_100m2": data.room_total / (data.size / 100) if data.size else 0.0,
    }
    df = pd.DataFrame([row])

    for col in FEATURE_NAMES:
        if col.startswith("subtype_"):
            df[col] = int(data.subtype == col[len("subtype_"):])
        elif col.startswith("heating_"):
            df[col] = int(data.heating == col[len("heating_"):])

    df = df.reindex(columns=FEATURE_NAMES, fill_value=0)

    if SCALER is not None and CONTINUOUS_COLS:
        df[CONTINUOUS_COLS] = SCALER.transform(df[CONTINUOUS_COLS])

    return df


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": MODEL is not None}


@app.get("/model_info")
def model_info():
    if MODEL is None:
        raise HTTPException(503, "Модель не загружена")
    return {
        "type": type(MODEL).__name__,
        "n_features": len(FEATURE_NAMES),
        "has_scaler": SCALER is not None,
    }


@app.post("/predict", response_model=PredictionOutput)
def predict(data: PredictionInput):
    if MODEL is None:
        raise HTTPException(503, "Модель не загружена")

    if data.floor_no > data.total_floor_count and data.floor_no > 0:
        raise HTTPException(400, "Этаж не может быть выше этажности дома")

    X = build_features(data)
    raw_log = float(MODEL.predict(X)[0])
    price = float(np.expm1(raw_log))          # ← обратно из лога

    if hasattr(MODEL, "estimators_"):
        tree_logs = np.array([t.predict(X)[0] for t in MODEL.estimators_])
        low = float(np.expm1(np.percentile(tree_logs, 10)))
        high = float(np.expm1(np.percentile(tree_logs, 90)))
    else:
        low = high = price

    return PredictionOutput(
        price=price,
        range_low=low,
        range_high=high,
        price_per_m2=price / data.size if data.size else 0.0,
        model_type=type(MODEL).__name__,
    )