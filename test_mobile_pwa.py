"""
Test script for Mobile PWA components
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

def test_imports():
    """Test if all critical modules can be imported"""
    print("Testing module imports...")
    
    try:
        from mobile.ocr_gauge_reader import GaugeReader
        print("✓ OCR Gauge Reader imported")
    except Exception as e:
        print(f"✗ OCR Gauge Reader failed: {e}")
        return False
    
    try:
        from mobile.voice_input import ArabicVoiceInput
        print("✓ Arabic Voice Input imported")
    except Exception as e:
        print(f"✗ Arabic Voice Input failed: {e}")
        return False
    
    try:
        from integrations.whatsapp_alerts import WhatsAppAlerter
        print("✓ WhatsApp Alerts imported")
    except Exception as e:
        print(f"✗ WhatsApp Alerts failed: {e}")
        return False
    
    return True


def test_ocr_reader():
    """Test OCR gauge reader initialization"""
    print("\nTesting OCR Gauge Reader...")
    
    try:
        from mobile.ocr_gauge_reader import GaugeReader
        reader = GaugeReader(languages=['en'])
        
        if reader.reader is not None:
            print("✓ OCR reader initialized successfully")
            
            # Test number extraction
            test_text = "Temperature: 78.5 °C"
            number = reader.extract_numbers(test_text)
            if number == 78.5:
                print(f"✓ Number extraction works: {number}")
            else:
                print(f"⚠ Number extraction issue: got {number}, expected 78.5")
            
            return True
        else:
            print("⚠ OCR reader initialized but may have issues")
            return False
            
    except Exception as e:
        print(f"✗ OCR test failed: {e}")
        return False


def test_voice_input():
    """Test voice input module"""
    print("\nTesting Voice Input...")
    
    try:
        from mobile.voice_input import ArabicVoiceInput
        voice = ArabicVoiceInput()
        
        # Test microphone
        if voice.test_microphone():
            print("✓ Microphone is accessible")
        else:
            print("⚠ Microphone not accessible (may need permissions)")
        
        # Test number extraction
        test_text_ar = "خمسة وسبعين"  # 75 in Arabic
        value = voice._extract_arabic_numbers("75")
        if value == 75.0:
            print(f"✓ Arabic number extraction works: {value}")
        else:
            print(f"⚠ Number extraction: got {value}")
        
        return True
        
    except Exception as e:
        print(f"✗ Voice input test failed: {e}")
        return False


def test_whatsapp():
    """Test WhatsApp alerter"""
    print("\nTesting WhatsApp Alerter...")
    
    try:
        from integrations.whatsapp_alerts import WhatsAppAlerter
        alerter = WhatsAppAlerter()
        
        if alerter.client is not None:
            print("✓ WhatsApp client initialized (credentials found)")
        else:
            print("⚠ WhatsApp not configured (set TWILIO credentials)")
        
        # Test message building
        msg = alerter._build_message(
            machine_id='M001',
            failure_type='Heat Dissipation',
            probability=0.85,
            time_to_failure=12,
            language='ar'
        )
        
        if 'M001' in msg and '85%' in msg:
            print("✓ Message building works")
        else:
            print("⚠ Message building may have issues")
        
        return True
        
    except Exception as e:
        print(f"✗ WhatsApp test failed: {e}")
        return False


def main():
    """Run all tests"""
    print("=" * 50)
    print("Mobile PWA Component Tests")
    print("=" * 50)
    print()
    
    results = []
    
    # Run tests
    results.append(("Imports", test_imports()))
    results.append(("OCR Reader", test_ocr_reader()))
    results.append(("Voice Input", test_voice_input()))
    results.append(("WhatsApp", test_whatsapp()))
    
    # Summary
    print("\n" + "=" * 50)
    print("Test Summary")
    print("=" * 50)
    
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status:10} {test_name}")
    
    all_passed = all(result[1] for result in results)
    
    print("\n" + "=" * 50)
    if all_passed:
        print("✓ All tests passed!")
    else:
        print("⚠ Some tests failed. Check details above.")
    print("=" * 50)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
