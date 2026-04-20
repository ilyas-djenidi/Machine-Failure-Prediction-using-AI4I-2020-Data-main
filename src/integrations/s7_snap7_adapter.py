"""
Siemens S7 PLC Adapter (Direct read)
Uses python-snap7 to read Data Blocks (DBs) directly from S7-1200/1500 or S7-300 PLCs.
"""

import snap7
from snap7.util import get_real, get_bool
import logging

logging.basicConfig(level=logging.INFO)

class S7Adapter:
    def __init__(self, ip: str, rack: int = 0, slot: int = 1):
        self.ip = ip
        self.rack = rack
        self.slot = slot
        self.client = snap7.client.Client()
        
    def connect(self):
        try:
            self.client.connect(self.ip, self.rack, self.slot)
            logging.info(f"Connected to S7 PLC at {self.ip}")
        except Exception as e:
            logging.error(f"Failed to connect to S7 PLC: {e}")
            raise
            
    def disconnect(self):
        if self.client.get_connected():
            self.client.disconnect()
            logging.info("Disconnected from S7 PLC.")
            
    def read_motor_db(self, db_number: int, start_offset: int = 0) -> dict:
        """
        Reads a standard Motor DB.
        Assumes the following DB structure (example):
        Offset 0.0: AmbientTemp (REAL)
        Offset 4.0: MotorTemp (REAL)
        Offset 8.0: Speed_RPM (REAL)
        Offset 12.0: Torque_Nm (REAL)
        Offset 16.0: RunHours (REAL)
        Offset 20.0: FaultBit (BOOL)
        """
        if not self.client.get_connected():
            logging.warning("Not connected. Attempting connection...")
            self.connect()
            
        try:
            # Read 21 bytes to cover offsets 0 to 20
            buffer = self.client.db_read(db_number, start_offset, 21)
            
            data = {
                "AmbientTemp": get_real(buffer, 0),
                "MotorTemp": get_real(buffer, 4),
                "Speed_RPM": get_real(buffer, 8),
                "Torque_Nm": get_real(buffer, 12),
                "RunHours": get_real(buffer, 16),
                "FaultBit": get_bool(buffer, 20, 0),
            }
            return data
        except Exception as e:
            logging.error(f"Failed to read DB {db_number}: {e}")
            return {}

if __name__ == "__main__":
    # Example usage
    # adapter = S7Adapter("192.168.0.1")
    # adapter.connect()
    # data = adapter.read_motor_db(db_number=5)
    # print(data)
    # adapter.disconnect()
    print("S7 Adapter ready.")
