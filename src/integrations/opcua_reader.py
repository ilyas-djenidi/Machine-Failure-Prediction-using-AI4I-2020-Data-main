"""
OPC-UA Reader for Siemens WinCC V16 Integration
Connects to OPC-UA Server and reads live motor tags.
"""

import asyncio
from asyncua import Client
import logging

logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger('asyncua')

class WinCCOPCUAReader:
    def __init__(self, endpoint="opc.tcp://DESKTOP-JK55OMP:4861"):
        self.endpoint = endpoint
        self.client = Client(url=self.endpoint)

    async def connect(self):
        try:
            await self.client.connect()
            _logger.info(f"Connected to OPC UA Server at {self.endpoint}")
        except Exception as e:
            _logger.error(f"Connection failed: {e}")
            raise

    async def disconnect(self):
        await self.client.disconnect()
        _logger.info("Disconnected from OPC UA Server")

    async def read_motor_data(self, motor_id: str) -> dict:
        """
        Reads tags for a specific motor instance (e.g., 'm_c_206').
        Adjust namespaces (ns=2) based on your specific WinCC UA Server configuration.
        """
        tags_to_read = {
            'AmbientTemp': f'ns=2;s={motor_id}.AirTemp',
            'MotorTemp'  : f'ns=2;s={motor_id}.ProcessTemp',
            'Speed_RPM'  : f'ns=2;s={motor_id}.Speed',
            'Torque_Nm'  : f'ns=2;s={motor_id}.Torque',
            'RunHours'   : f'ns=2;s={motor_id}.RunHours',
            'FaultBit'   : f'ns=2;s={motor_id}_status',
        }
        
        results = {}
        for tag_key, node_id in tags_to_read.items():
            try:
                node = self.client.get_node(node_id)
                value = await node.read_value()
                results[tag_key] = value
            except Exception as e:
                _logger.warning(f"Failed to read tag {node_id}: {e}")
                results[tag_key] = None
                
        return results

async def main():
    reader = WinCCOPCUAReader()
    await reader.connect()
    
    # Example: Read pilot motor data
    data = await reader.read_motor_data("m_c_206")
    print(f"Data for m_c_206: {data}")
    
    await reader.disconnect()

if __name__ == "__main__":
    asyncio.run(main())
