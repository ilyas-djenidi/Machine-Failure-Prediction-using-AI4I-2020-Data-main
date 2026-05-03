"""
OPC-UA Reader for Siemens WinCC V16 Integration
Connects to OPC-UA Server and reads live motor tags.

FIX (opcua_reader.py — 1 bug):
  - read_motor_data() now retries each tag read with exponential backoff
    instead of silently returning None on first failure.
  - Added async context-manager connect/disconnect so the client is always
    cleaned up even if an exception occurs mid-read.
  - Added 'timestamp' key to returned dict.
  - Added configurable connect_timeout parameter.
"""

import asyncio
import logging
from datetime import datetime, timezone

_logger = logging.getLogger("asyncua")
logging.basicConfig(level=logging.INFO)

try:
    from asyncua import Client
except ImportError:
    Client = None  # graceful degradation if asyncua not installed


class WinCCOPCUAReader:
    def __init__(self,
                 endpoint: str = "opc.tcp://DESKTOP-JK55OMP:4861",
                 connect_timeout: float = 4.0):
        if Client is None:
            raise ImportError(
                "asyncua is required: pip install asyncua"
            )
        self.endpoint = endpoint
        self.connect_timeout = connect_timeout
        self.client = Client(url=self.endpoint, timeout=self.connect_timeout)

    async def connect(self):
        try:
            await self.client.connect()
            _logger.info(f"Connected to OPC UA Server at {self.endpoint}")
        except Exception as e:
            _logger.error(f"Connection failed: {e}")
            raise

    async def disconnect(self):
        try:
            await self.client.disconnect()
            _logger.info("Disconnected from OPC UA Server")
        except Exception:
            pass  # best-effort disconnect

    async def read_motor_data(self, motor_id: str, retries: int = 3) -> dict:
        """
        Reads tags for a specific motor instance (e.g., 'm_c_206').
        Adjust namespaces (ns=2) based on your WinCC UA Server configuration.

        FIX: Each tag read is retried up to `retries` times with exponential
        backoff (1s, 2s, 4s) instead of silently returning None on first error.
        Returns 'timestamp' key in UTC ISO format.
        """
        tags_to_read = {
            "AmbientTemp": f"ns=2;s={motor_id}.AirTemp",
            "MotorTemp"  : f"ns=2;s={motor_id}.ProcessTemp",
            "Speed_RPM"  : f"ns=2;s={motor_id}.Speed",
            "Torque_Nm"  : f"ns=2;s={motor_id}.Torque",
            "RunHours"   : f"ns=2;s={motor_id}.RunHours",
            "FaultBit"   : f"ns=2;s={motor_id}_status",
        }

        results = {}
        for tag_key, node_id in tags_to_read.items():
            value = None
            for attempt in range(retries):
                try:
                    node  = self.client.get_node(node_id)
                    value = await node.read_value()
                    break  # success — stop retrying this tag
                except Exception as e:
                    wait = 2 ** attempt
                    _logger.warning(
                        f"Tag {node_id} read attempt {attempt+1}/{retries} failed: {e}. "
                        f"Retrying in {wait}s"
                    )
                    if attempt < retries - 1:
                        await asyncio.sleep(wait)
                    else:
                        _logger.error(f"All {retries} attempts failed for tag {node_id}")
            results[tag_key] = value

        results["timestamp"] = datetime.now(timezone.utc).isoformat()
        return results

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.disconnect()
        return False  # do not suppress exceptions


async def main():
    """Example usage with async context manager (ensures clean disconnect)."""
    async with WinCCOPCUAReader() as reader:
        data = await reader.read_motor_data("m_c_206")
        print(f"Data for m_c_206: {data}")


if __name__ == "__main__":
    asyncio.run(main())
