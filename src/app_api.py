"""
FastAPI REST API for Motor Failure Prediction
Provides HTTP endpoints for SCADA systems to request predictions.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import uvicorn
import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from notebooks.predict_factory import MotorFailurePredictor
from notebooks.generate_report import generate_report

app = FastAPI(title="M'Sila Factory - Motor Predictive Maintenance API")

# Load model globally
predictor = None

class SensorData(BaseModel):
    machine_id: str
    motor_type: str = "VIDA"
    sensors: Dict[str, float]

class BatchSensorData(BaseModel):
    records: List[SensorData]

@app.on_event("startup")
def load_predictor():
    global predictor
    config_path = ROOT / "models" / "production_config.json"
    if config_path.exists():
        predictor = MotorFailurePredictor.from_config(config_path)
        print("Predictor loaded successfully.")
    else:
        print("Warning: production_config.json not found. Model not loaded.")

@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": predictor is not None}

@app.post("/predict")
def predict(data: SensorData):
    if not predictor:
        raise HTTPException(status_code=503, detail="Model not loaded")
        
    try:
        result = predictor.predict(
            raw_input=data.sensors,
            machine_id=data.machine_id,
            motor_type=data.motor_type
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict_batch")
def predict_batch(data: BatchSensorData):
    if not predictor:
        raise HTTPException(status_code=503, detail="Model not loaded")
        
    results = []
    for record in data.records:
        try:
            res = predictor.predict(
                raw_input=record.sensors,
                machine_id=record.machine_id,
                motor_type=record.motor_type
            )
            results.append(res)
        except Exception as e:
            results.append({"machine_id": record.machine_id, "error": str(e)})
            
    return {"results": results}


# BUG 10 fix: /report endpoint — wires predictor + generate_report together
class ReportRequest(BaseModel):
    machine_id: str
    motor_type: str = "VIDA"
    manufacturer: str = "Siemens"   # "Siemens" or "Schneider"
    sensors: Dict[str, float]

@app.post("/report")
def get_report(data: ReportRequest):
    if not predictor:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        prediction = predictor.predict(
            raw_input=data.sensors,
            machine_id=data.machine_id,
            motor_type=data.motor_type
        )
        report = generate_report(prediction, manufacturer=data.manufacturer)
        return {
            "prediction" : prediction,
            "report_text": report["text"],
            "report_lines": report["lines"],
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
