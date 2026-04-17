"""
PLC Integration Module for Industrial Equipment
Supports Schneider (Modbus TCP) and Siemens (S7) PLCs
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional
import logging
from datetime import datetime

# Try to import PLC libraries
try:
    from pymodbus.client import ModbusTcpClient
    MODBUS_AVAILABLE = True
except ImportError:
    MODBUS_AVAILABLE = False
    logging.warning("pymodbus not installed - Schneider Modbus support disabled")

try:
    import snap7
    from snap7.util import get_real, set_real
    SNAP7_AVAILABLE = True
except ImportError:
    SNAP7_AVAILABLE = False
    logging.warning("python-snap7 not installed - Siemens S7 support disabled")

try:
    from opcua import Client as OPCUAClient
    OPCUA_AVAILABLE = True
except ImportError:
    OPCUA_AVAILABLE = False
    logging.warning("opcua not installed - OPC UA support disabled")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PLCConnector(ABC):
    """Base class for PLC connectors"""
    
    def __init__(self, ip_address: str, port: int = None):
        self.ip_address = ip_address
        self.port = port
        self.connected = False
        self.last_error = None
    
    @abstractmethod
    def connect(self) -> bool:
        """Establish connection to PLC"""
        pass
    
    @abstractmethod
    def disconnect(self):
        """Close connection to PLC"""
        pass
    
    @abstractmethod
    def read_sensor(self, address: str) -> Optional[float]:
        """Read a single sensor value"""
        pass
    
    @abstractmethod
    def read_sensors_batch(self, addresses: List[str]) -> Dict[str, float]:
        """Read multiple sensors in one operation"""
        pass
    
    def is_connected(self) -> bool:
        """Check if connection is alive"""
        return self.connected
    
    def get_status(self) -> Dict:
        """Get connection status"""
        return {
            'connected': self.connected,
            'ip_address': self.ip_address,
            'port': self.port,
            'last_error': self.last_error
        }


class SchneiderModbusConnector(PLCConnector):
    """Connector for Schneider Electric PLCs via Modbus TCP"""
    
    def __init__(self, ip_address: str, port: int = 502):
        super().__init__(ip_address, port)
        self.client = None
        if not MODBUS_AVAILABLE:
            raise ImportError("pymodbus is not installed. Install with: pip install pymodbus")
    
    def connect(self) -> bool:
        """Connect to Schneider PLC via Modbus TCP"""
        try:
            logger.info(f"Connecting to Schneider PLC at {self.ip_address}:{self.port}")
            self.client = ModbusTcpClient(self.ip_address, port=self.port)
            self.connected = self.client.connect()
            
            if self.connected:
                logger.info("✓ Connected to Schneider PLC")
            else:
                self.last_error = "Connection failed"
                logger.error(f"✗ Failed to connect to {self.ip_address}:{self.port}")
            
            return self.connected
        except Exception as e:
            self.last_error = str(e)
            logger.error(f"Connection error: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from PLC"""
        if self.client:
            self.client.close()
            self.connected = False
            logger.info("Disconnected from Schneider PLC")
    
    def read_sensor(self, address: str) -> Optional[float]:
        """
        Read sensor from holding register
        
        Args:
            address: Register address (e.g., "40001" for holding register 1)
        """
        if not self.connected:
            self.last_error = "Not connected"
            return None
        
        try:
            # Parse address (format: 4xxxx for holding registers)
            register_num = int(address) - 40001  # Modbus addressing offset
            
            # Read holding register
            result = self.client.read_holding_registers(register_num, 1)
            
            if result.isError():
                self.last_error = f"Read error for register {register_num}"
                logger.error(self.last_error)
                return None
            
            # Convert to float (assuming register stores scaled integer)
            value = result.registers[0] / 10.0  # Adjust scaling as needed
            logger.debug(f"Read {address}: {value}")
            return value
            
        except Exception as e:
            self.last_error = str(e)
            logger.error(f"Read error: {e}")
            return None
    
    def read_sensors_batch(self, addresses: List[str]) -> Dict[str, float]:
        """Read multiple sensors efficiently"""
        results = {}
        
        for addr in addresses:
            value = self.read_sensor(addr)
            if value is not None:
                results[addr] = value
        
        return results


class SiemensS7Connector(PLCConnector):
    """Connector for Siemens S7 PLCs"""
    
    def __init__(self, ip_address: str, rack: int = 0, slot: int = 1):
        super().__init__(ip_address)
        self.rack = rack
        self.slot = slot
        self.client = None
        if not SNAP7_AVAILABLE:
            raise ImportError("python-snap7 is not installed. Install with: pip install python-snap7")
    
    def connect(self) -> bool:
        """Connect to Siemens S7 PLC"""
        try:
            logger.info(f"Connecting to Siemens S7 PLC at {self.ip_address} (rack={self.rack}, slot={self.slot})")
            self.client = snap7.client.Client()
            self.client.connect(self.ip_address, self.rack, self.slot)
            
            self.connected = True
            logger.info("✓ Connected to Siemens S7 PLC")
            return True
            
        except Exception as e:
            self.last_error = str(e)
            self.connected = False
            logger.error(f"Connection error: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from PLC"""
        if self.client:
            self.client.disconnect()
            self.connected = False
            logger.info("Disconnected from Siemens S7 PLC")
    
    def read_sensor(self, address: str) -> Optional[float]:
        """
        Read sensor from data block
        
        Args:
            address: DB address (e.g., "DB1.0" for DB1, byte 0)
        """
        if not self.connected:
            self.last_error = "Not connected"
            return None
        
        try:
            # Parse address (format: DBx.y where x=DB number, y=byte offset)
            parts = address.split('.')
            db_number = int(parts[0].replace('DB', ''))
            byte_offset = int(parts[1])
            
            # Read 4 bytes for a REAL (float)
            data = self.client.db_read(db_number, byte_offset, 4)
            
            # Convert bytes to float
            value = get_real(data, 0)
            logger.debug(f"Read {address}: {value}")
            return value
            
        except Exception as e:
            self.last_error = str(e)
            logger.error(f"Read error: {e}")
            return None
    
    def read_sensors_batch(self, addresses: List[str]) -> Dict[str, float]:
        """Read multiple sensors"""
        results = {}
        
        for addr in addresses:
            value = self.read_sensor(addr)
            if value is not None:
                results[addr] = value
        
        return results


class OPCUAConnector(PLCConnector):
    """Connector for OPC UA servers"""
    
    def __init__(self, endpoint_url: str):
        super().__init__(endpoint_url)
        self.endpoint_url = endpoint_url
        self.client = None
        if not OPCUA_AVAILABLE:
            raise ImportError("opcua is not installed. Install with: pip install opcua")
    
    def connect(self) -> bool:
        """Connect to OPC UA server"""
        try:
            logger.info(f"Connecting to OPC UA server: {self.endpoint_url}")
            self.client = OPCUAClient(self.endpoint_url)
            self.client.connect()
            
            self.connected = True
            logger.info("✓ Connected to OPC UA server")
            return True
            
        except Exception as e:
            self.last_error = str(e)
            self.connected = False
            logger.error(f"Connection error: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from OPC UA server"""
        if self.client:
            self.client.disconnect()
            self.connected = False
            logger.info("Disconnected from OPC UA server")
    
    def read_sensor(self, node_id: str) -> Optional[float]:
        """
        Read sensor value from OPC UA node
        
        Args:
            node_id: Node identifier (e.g., "ns=2;i=1001")
        """
        if not self.connected:
            self.last_error = "Not connected"
            return None
        
        try:
            node = self.client.get_node(node_id)
            value = node.get_value()
            logger.debug(f"Read {node_id}: {value}")
            return float(value)
            
        except Exception as e:
            self.last_error = str(e)
            logger.error(f"Read error: {e}")
            return None
    
    def read_sensors_batch(self, node_ids: List[str]) -> Dict[str, float]:
        """Read multiple sensors"""
        results = {}
        
        for node_id in node_ids:
            value = self.read_sensor(node_id)
            if value is not None:
                results[node_id] = value
        
        return results


def create_plc_connector(plc_type: str, **kwargs) -> Optional[PLCConnector]:
    """
    Factory function to create appropriate PLC connector
    
    Args:
        plc_type: Type of PLC ('schneider', 'siemens', 'opcua')
        **kwargs: Connection parameters
    
    Returns:
        Configured PLC connector or None
    """
    try:
        if plc_type.lower() == 'schneider':
            return SchneiderModbusConnector(
                ip_address=kwargs['ip_address'],
                port=kwargs.get('port', 502)
            )
        elif plc_type.lower() == 'siemens':
            return SiemensS7Connector(
                ip_address=kwargs['ip_address'],
                rack=kwargs.get('rack', 0),
                slot=kwargs.get('slot', 1)
            )
        elif plc_type.lower() == 'opcua':
            return OPCUAConnector(
                endpoint_url=kwargs['endpoint_url']
            )
        else:
            logger.error(f"Unknown PLC type: {plc_type}")
            return None
    except KeyError as e:
        logger.error(f"Missing required parameter: {e}")
        return None
    except Exception as e:
        logger.error(f"Error creating connector: {e}")
        return None


# Example usage
if __name__ == "__main__":
    # Example: Connect to Schneider PLC
    schneider = create_plc_connector(
        'schneider',
        ip_address='192.168.1.100',
        port=502
    )
    
    if schneider:
        if schneider.connect():
            # Read temperature sensor from register 40001
            temp = schneider.read_sensor('40001')
            print(f"Temperature: {temp}°C")
            
            # Read multiple sensors
            sensors = schneider.read_sensors_batch(['40001', '40002', '40003'])
            print(f"Batch read: {sensors}")
            
            schneider.disconnect()
