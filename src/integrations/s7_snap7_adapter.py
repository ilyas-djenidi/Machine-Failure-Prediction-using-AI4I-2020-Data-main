"""
Siemens S7 PLC Adapter (Direct read)
Uses python-snap7 to read Data Blocks (DBs) directly from S7-1200/1500 or S7-300 PLCs.
"""

import snap7
from snap7.util import get_real, get_bool
import logging
import time
from datetime import datetime, timezone

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
            
    def read_motor_db(self, db_number: int, start_offset: int = 0, retries: int = 3) -> dict:
        """
        Reads a standard Motor DB with exponential-backoff retry (BUG 11 fix).
        Assumes the following DB structure:
          Offset  0.0: AmbientTemp (REAL)
          Offset  4.0: MotorTemp   (REAL)
          Offset  8.0: Speed_RPM   (REAL)
          Offset 12.0: Torque_Nm   (REAL)
          Offset 16.0: RunHours    (REAL)
          Offset 20.0: FaultBit    (BOOL bit 0)
        """
        for attempt in range(retries):
            if not self.client.get_connected():
                try:
                    self.connect()
                except Exception as e:
                    wait = 2 ** attempt
                    logging.warning(f"Reconnect attempt {attempt+1} failed: {e}. Retrying in {wait}s")
                    time.sleep(wait)
                    continue
            try:
                buffer = self.client.db_read(db_number, start_offset, 21)
                return {
                    "AmbientTemp": round(get_real(buffer, 0),  2),
                    "MotorTemp"  : round(get_real(buffer, 4),  2),
                    "Speed_RPM"  : round(get_real(buffer, 8),  1),
                    "Torque_Nm"  : round(get_real(buffer, 12), 2),
                    "RunHours"   : round(get_real(buffer, 16), 4),
                    "FaultBit"   : get_bool(buffer, 20, 0),
                    "timestamp"  : datetime.now(timezone.utc).isoformat(),
                }
            except Exception as e:
                wait = 2 ** attempt
                logging.error(f"Read attempt {attempt+1}/{retries} failed for DB{db_number}: {e}. Retrying in {wait}s")
                time.sleep(wait)
        logging.error(f"All {retries} read attempts failed for DB{db_number}. Returning empty dict.")
        return {}

if __name__ == "__main__":
    # Example usage
    # adapter = S7Adapter("192.168.0.1")
    # adapter.connect()
    # data = adapter.read_motor_db(db_number=5)
    # print(data)
    # adapter.disconnect()
    print("S7 Adapter ready.")
