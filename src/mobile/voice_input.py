"""
Arabic Voice Input for Hands-Free Data Entry
Supports Algerian Arabic dialect
"""

import speech_recognition as sr
import logging
import re
from typing import Optional, Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ArabicVoiceInput:
    """Voice-to-text for Arabic speech recognition"""
    
    def __init__(self):
        """Initialize speech recognizer"""
        self.recognizer = sr.Recognizer()
        # Adjust for ambient noise
        self.recognizer.energy_threshold = 4000
        self.recognizer.dynamic_energy_threshold = True
        logger.info("✓ Voice input initialized")
    
    def listen_for_value(self, timeout: int = 5, language: str = 'ar-DZ') -> Dict:
        """
        Listen for spoken value
        
        Args:
            timeout: Maximum time to wait for speech (seconds)
            language: Language code (ar-DZ for Algerian Arabic, fr-FR for French)
            
        Returns:
            Dictionary with transcribed text and extracted value
        """
        try:
            with sr.Microphone() as source:
                logger.info("🎤 Listening... Speak now")
                
                # Adjust for ambient noise
                logger.info("Adjusting for ambient noise...")
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
                
                # Listen for audio
                audio = self.recognizer.listen(source, timeout=timeout)
                
                logger.info("Processing speech...")
                
                # Try Google Speech Recognition (supports Arabic)
                try:
                    text = self.recognizer.recognize_google(audio, language=language)
                    logger.info(f"Recognized: {text}")
                    
                    # Extract numeric value
                    value = self.extract_value_from_speech(text, language)
                    
                    return {
                        'success': True,
                        'text': text,
                        'value': value,
                        'language': language
                    }
                    
                except sr.UnknownValueError:
                    logger.warning("Could not understand audio")
                    return {
                        'success': False,
                        'error': 'Could not understand speech',
                        'text': None,
                        'value': None
                    }
                    
                except sr.RequestError as e:
                    logger.error(f"API error: {e}")
                    return {
                        'success': False,
                        'error': f'Speech recognition service error: {e}',
                        'text': None,
                        'value': None
                    }
                    
        except Exception as e:
            logger.error(f"Error capturing audio: {e}")
            return {
                'success': False,
                'error': str(e),
                'text': None,
                'value': None
            }
    
    def extract_value_from_speech(self, text: str, language: str = 'ar') -> Optional[float]:
        """
        Extract numeric value from spoken text
        
        Args:
            text: Transcribed speech text
            language: Language of speech
            
        Returns:
            Extracted numeric value or None
        """
        if language.startswith('ar'):
            return self._extract_arabic_numbers(text)
        elif language.startswith('fr'):
            return self._extract_french_numbers(text)
        else:
            return self._extract_english_numbers(text)
    
    def _extract_arabic_numbers(self, text: str) -> Optional[float]:
        """Extract numbers from Arabic text"""
        # Arabic number words mapping
        arabic_numbers = {
            'صفر': 0, 'واحد': 1, 'اثنين': 2, 'ثلاثة': 3, 'أربعة': 4,
            'خمسة': 5, 'ستة': 6, 'سبعة': 7, 'ثمانية': 8, 'تسعة': 9,
            'عشرة': 10, 'عشرين': 20, 'ثلاثين': 30, 'أربعين': 40,
            'خمسين': 50, 'ستين': 60, 'سبعين': 70, 'ثمانين': 80, 'تسعين': 90,
            'مئة': 100, 'مائة': 100
        }
        
        # Try to find direct numeric digits first
        digits = re.findall(r'\d+\.?\d*', text)
        if digits:
            try:
                return float(digits[0])
            except ValueError:
                pass
        
        # Try to parse Arabic number words
        words = text.split()
        for word in words:
            if word in arabic_numbers:
                return float(arabic_numbers[word])
        
        return None
    
    def _extract_french_numbers(self, text: str) -> Optional[float]:
        """Extract numbers from French text"""
        french_numbers = {
            'zéro': 0, 'un': 1, 'deux': 2, 'trois': 3, 'quatre': 4,
            'cinq': 5, 'six': 6, 'sept': 7, 'huit': 8, 'neuf': 9,
            'dix': 10, 'vingt': 20, 'trente': 30, 'quarante': 40,
            'cinquante': 50, 'soixante': 60, 'soixante-dix': 70,
            'quatre-vingt': 80, 'quatre-vingt-dix': 90, 'cent': 100
        }
        
        # Try digits first
        digits = re.findall(r'\d+\.?\d*', text.lower())
        if digits:
            try:
                return float(digits[0])
            except ValueError:
                pass
        
        # Try French number words
        words = text.lower().split()
        for word in words:
            if word in french_numbers:
                return float(french_numbers[word])
        
        return None
    
    def _extract_english_numbers(self, text: str) -> Optional[float]:
        """Extract numbers from English text"""
        # Try digits
        digits = re.findall(r'\d+\.?\d*', text)
        if digits:
            try:
                return float(digits[0])
            except ValueError:
                pass
        
        # Basic English number words
        english_numbers = {
            'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4,
            'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9,
            'ten': 10, 'twenty': 20, 'thirty': 30, 'forty': 40,
            'fifty': 50, 'sixty': 60, 'seventy': 70, 'eighty': 80,
            'ninety': 90, 'hundred': 100
        }
        
        words = text.lower().split()
        for word in words:
            if word in english_numbers:
                return float(english_numbers[word])
        
        return None
    
    def test_microphone(self) -> bool:
        """
        Test if microphone is working
        
        Returns:
            True if microphone is accessible
        """
        try:
            with sr.Microphone() as source:
                logger.info("✓ Microphone is working")
                return True
        except Exception as e:
            logger.error(f"✗ Microphone error: {e}")
            return False


if __name__ == "__main__":
    # Test voice input
    voice = ArabicVoiceInput()
    
    if voice.test_microphone():
        logger.info("Voice input ready for use")
        # Uncomment to test:
        # result = voice.listen_for_value(language='ar-DZ')
        # print(f"Result: {result}")
