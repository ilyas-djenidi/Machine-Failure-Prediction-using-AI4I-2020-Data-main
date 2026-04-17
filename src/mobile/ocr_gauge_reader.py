"""
OCR Gauge Reader for Analog and Digital Gauges
Supports reading values from photos of industrial equipment
"""

import cv2
import numpy as np
from PIL import Image
import easyocr
import logging
from typing import Dict, Optional, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GaugeReader:
    """Read numeric values from gauge photos using OCR"""
    
    def __init__(self, languages=['en', 'ar']):
        """
        Initialize OCR reader
        
        Args:
            languages: List of languages for OCR (default: English and Arabic)
        """
        try:
            self.reader = easyocr.Reader(languages, gpu=False)
            logger.info("✓ OCR reader initialized")
        except Exception as e:
            logger.error(f"Failed to initialize OCR: {e}")
            self.reader = None
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for better OCR results
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Preprocessed image
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Apply denoising
        denoised = cv2.fastNlMeansDenoising(gray)
        
        # Increase contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(denoised)
        
        # Apply adaptive thresholding
        binary = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        return binary
    
    def extract_numbers(self, text: str) -> Optional[float]:
        """
        Extract numeric value from OCR text
        
        Args:
            text: OCR result text
            
        Returns:
            Extracted number or None
        """
        # Remove common non-numeric characters
        cleaned = text.strip().replace(' ', '').replace(',', '.')
        
        # Try to find numbers in the text
        import re
        
        # Pattern for decimal numbers
        patterns = [
            r'[-+]?\d*\.?\d+',  # Standard decimal
            r'[-+]?\d+',         # Integer only
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, cleaned)
            if matches:
                try:
                    return float(matches[0])
                except ValueError:
                    continue
        
        return None
    
    def detect_gauge_type(self, image: np.ndarray) -> str:
        """
        Detect type of gauge (analog dial, digital display, thermometer)
        
        Args:
            image: Input image
            
        Returns:
            Gauge type: 'digital', 'analog_dial', 'thermometer', 'unknown'
        """
        # Simple heuristic based on edge detection
        edges = cv2.Canny(image, 50, 150)
        
        # Count circular features (Hough circles)
        circles = cv2.HoughCircles(
            cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image,
            cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=10, maxRadius=200
        )
        
        if circles is not None and len(circles[0]) > 0:
            return 'analog_dial'
        
        # Check for vertical lines (thermometer)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, 
                                minLineLength=50, maxLineGap=10)
        
        if lines is not None:
            vertical_lines = sum(1 for line in lines 
                               if abs(line[0][0] - line[0][2]) < 10)
            if vertical_lines > 2:
                return 'thermometer'
        
        return 'digital'
    
    def read_gauge(self, image_path: str, gauge_type: str = 'auto') -> Dict:
        """
        Read value from gauge image
        
        Args:
            image_path: Path to gauge image
            gauge_type: Type of gauge ('auto', 'digital', 'analog_dial', 'thermometer')
            
        Returns:
            Dictionary with detected value, confidence, and metadata
        """
        if self.reader is None:
            return {
                'success': False,
                'error': 'OCR reader not initialized',
                'value': None,
                'confidence': 0.0
            }
        
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                return {
                    'success': False,
                    'error': 'Failed to load image',
                    'value': None,
                    'confidence': 0.0
                }
            
            # Auto-detect gauge type if needed
            if gauge_type == 'auto':
                gauge_type = self.detect_gauge_type(image)
                logger.info(f"Detected gauge type: {gauge_type}")
            
            # Preprocess image
            processed = self.preprocess_image(image)
            
            # Perform OCR
            results = self.reader.readtext(processed)
            
            if not results:
                return {
                    'success': False,
                    'error': 'No text detected in image',
                    'value': None,
                    'confidence': 0.0,
                    'gauge_type': gauge_type
                }
            
            # Extract numbers from OCR results
            best_value = None
            best_confidence = 0.0
            all_detections = []
            
            for (bbox, text, confidence) in results:
                number = self.extract_numbers(text)
                if number is not None:
                    all_detections.append({
                        'text': text,
                        'value': number,
                        'confidence': confidence
                    })
                    
                    if confidence > best_confidence:
                        best_value = number
                        best_confidence = confidence
            
            if best_value is None:
                return {
                    'success': False,
                    'error': 'No numeric values detected',
                    'value': None,
                    'confidence': 0.0,
                    'gauge_type': gauge_type,
                    'raw_detections': all_detections
                }
            
            logger.info(f"✓ Read value: {best_value} (confidence: {best_confidence:.2f})")
            
            return {
                'success': True,
                'value': best_value,
                'confidence': best_confidence,
                'gauge_type': gauge_type,
                'all_detections': all_detections,
                'warning': 'Please verify' if best_confidence < 0.7 else None
            }
            
        except Exception as e:
            logger.error(f"Error reading gauge: {e}")
            return {
                'success': False,
                'error': str(e),
                'value': None,
                'confidence': 0.0
            }
    
    def validate_reading(self, value: float, expected_range: Tuple[float, float]) -> bool:
        """
        Validate if reading is within expected range
        
        Args:
            value: Read value
            expected_range: (min, max) tuple
            
        Returns:
            True if valid, False otherwise
        """
        min_val, max_val = expected_range
        return min_val <= value <= max_val


if __name__ == "__main__":
    # Test the gauge reader
    reader = GaugeReader()
    logger.info("GaugeReader initialized for testing")
