"""
Test scenarios for MotorFailurePredictor v2
Run this after training the new model and downloading artifacts to the models/ directory.
"""

from predict_factory import MotorFailurePredictor
from pathlib import Path
import json

def test_scenarios():
    config_path = Path(__file__).parent.parent / "models" / "production_config.json"
    
    if not config_path.exists():
        print("Model configuration not found. Please run train_final.py first.")
        return

    predictor = MotorFailurePredictor.from_config(config_path)

    scenarios = [
        {
            "name": "Scenario 1: NORMAL",
            "sensors": {
                "Speed_RPM": 1500,
                "Torque_Nm": 35,
                "AmbientTemp": 25.0,  # 298.15K
                "MotorTemp": 37.0,    # 310.15K -> temp_diff = 12
                "RunHours": 50 / 60.0 # wear = 50
            }
        },
        {
            "name": "Scenario 2: HDF (Heat Dissipation Failure)",
            "sensors": {
                "Speed_RPM": 1350,
                "Torque_Nm": 42,
                "AmbientTemp": 35.0,
                "MotorTemp": 41.5,    # temp_diff = 6.5 (< 8.6)
                "RunHours": 120 / 60.0 # wear = 120
            }
        },
        {
            "name": "Scenario 3: OSF (Overstrain Failure)",
            "sensors": {
                "Speed_RPM": 1480,
                "Torque_Nm": 72,
                "AmbientTemp": 30.0,
                "MotorTemp": 40.0,    # temp_diff = 10
                "RunHours": 215 / 60.0 # wear = 215 -> wear*torque = 15480 (> 13000)
            }
        }
    ]

    for s in scenarios:
        print(f"\n{'='*50}\n{s['name']}\n{'='*50}")
        result = predictor.predict(raw_input=s["sensors"], machine_id="m_c_test", motor_type="VIDA")
        print(json.dumps(result, indent=2, ensure_ascii=True))

if __name__ == "__main__":
    test_scenarios()
