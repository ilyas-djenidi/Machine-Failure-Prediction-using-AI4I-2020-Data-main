"""
FastAPI REST API for Motor Failure Prediction
Provides HTTP endpoints for SCADA systems to request predictions.
"""

from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import uvicorn
import sys, io
# Fix Windows console Unicode encoding
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from notebooks.predict_factory import MotorFailurePredictor
from notebooks.generate_report import generate_report

app = FastAPI(title="M'Sila Factory - Motor Predictive Maintenance API")

# Load model globally on startup (FIX BUG 10 / CHECK 3)
CONFIG_PATH = ROOT / "models" / "production_config.json"
try:
    predictor = MotorFailurePredictor.from_config(CONFIG_PATH)
    print(f"[API] Predictor loaded successfully from {CONFIG_PATH}")
except Exception as e:
    predictor = None
    print(f"[API] WARNING: Could not load predictor: {e}")

class SensorData(BaseModel):
    machine_id: str
    motor_type: str = "VIDA"
    sensors: Dict[str, float]

class BatchSensorData(BaseModel):
    records: List[SensorData]

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
@app.post("/report_pdf")
def get_pdf_report(data: ReportRequest):
    if not predictor:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        prediction = predictor.predict(
            raw_input=data.sensors,
            machine_id=data.machine_id,
            motor_type=data.motor_type
        )
        report = generate_report(prediction, manufacturer=data.manufacturer)
        
        # Save temporary PDF
        pdf_path = ROOT / "reports" / f"temp_report_{data.machine_id}.pdf"
        from notebooks.generate_report import save_pdf
        save_pdf(str(pdf_path), report)
        
        return FileResponse(
            path=str(pdf_path),
            filename=f"Rapport_{data.machine_id}.pdf",
            media_type="application/pdf"
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
