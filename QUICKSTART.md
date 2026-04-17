# Quick Start - Mobile PWA for Algerian Factories
# Guide de Démarrage Rapide / دليل البدء السريع

## 🚀 For Technicians / Pour les Techniciens / للفنيين

### Option 1: Mobile Phone App (Recommended) 📱

**Works with ZERO factory infrastructure - just your smartphone!**

#### Step 1: Installation
1. Open this link on your phone: `http://[server-ip]:8502`
2. Tap "Add to Home Screen" 
3. The app will work like a normal app

#### Step 2: Take Your First Reading
1. Open the app
2. Enter machine ID (example: M001)
3. Select reading type (Temperature, Pressure, etc.)
4. **Take a photo** of the gauge → AI reads it automatically!
5. Or use **voice input**: Say the value in Arabic or French
6. Or **type manually** if you prefer
7. Tap Submit ✅

#### Step 3: Sync Data
-Your readings save automatically (works offline!)
- When WiFi is available, data auto-syncs to the server
- No mobile data used (only WiFi)

---

## 👨‍💻 For Administrators / Pour les Administrateurs / للمديرين

### Running the Mobile PWA

#### Prerequisites
- Windows PC with Python 3.8+
- WiFi network for phones to connect

#### Installation (One-Time)

**Option A: Automatic (Recommended)**
```powershell
# Run this in PowerShell:
./install.ps1
```

**Option B: Manual**
```powershell
# Create virtual environment
python -m venv venv
./venv/Scripts/Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Download OCR models (one-time, ~500MB)
python -c "import easyocr; easyocr.Reader(['en', 'ar'], gpu=False)"
```

#### Running the Mobile App

```powershell
# Activate virtual environment
./venv/Scripts/Activate.ps1

# Start mobile PWA
python -m streamlit run src/mobile/pwa_app.py --server.port 8502
```

The app will open at: `http://localhost:8502`

**To access from phones on same WiFi:**
1. Find your PC's IP address: `ipconfig` (look for IPv4 under WiFi)
2. On phones, open: `http://[YOUR-IP]:8502`
3. Example: `http://192.168.1.100:8502`

---

## 🔔 Setting Up WhatsApp Alerts

### Get Twilio Credentials
1. Sign up at: https://www.twilio.com/try-twilio
2. Get free trial credits (includes WhatsApp)
3. Note your credentials:
   - Account SID
   - Auth Token
   - WhatsApp number (usually: `whatsapp:+14155238886`)

### Configure in Windows

```powershell
# Set environment variables
$env:TWILIO_ACCOUNT_SID='your_account_sid_here'
$env:TWILIO_AUTH_TOKEN='your_auth_token_here'
$env:TWILIO_WHATSAPP_NUMBER='whatsapp:+14155238886'

# Test it
python -c "from src.integrations.whatsapp_alerts import WhatsAppAlerter; a=WhatsAppAlerter(); a.test_connection('whatsapp:+213XXXXXXXXX')"
```

Replace `+213XXXXXXXXX` with an Algerian WhatsApp number to test.

---

## 📊 Demo Mode for Sales

### Show All Three Scenarios

**Scenario 1: Modern Factory (Has PLCs)**
```powershell
# Show PLC integration (coming in Phase 8)
python src/integrations/plc_demo.py
```

**Scenario 2: Traditional Factory (Mobile PWA)** ⭐
```powershell
# Run mobile PWA and demonstrate:
python -m streamlit run src/mobile/pwa_app.py --server.port 8502

# 1. Photo capture → automatic reading
# 2. Arabic voice input → hands-free
# 3. Works offline → reliable
```

**Scenario 3: Desktop Manual Entry**
```powershell
# Main system with manual forms
python -m streamlit run src/app.py
```

---

## 🧪 Testing the System

### Test All Components
```powershell
python test_mobile_pwa.py
```

Expected output:
```
✓ PASS       Imports
✓ PASS       OCR Reader
⚠ PASS       Voice Input (mic may need permissions)
⚠ PASS       WhatsApp (needs credentials)
```

### Test Individual Features

**Test OCR Gauge Reading:**
```python
from src.mobile.ocr_gauge_reader import GaugeReader

reader = GaugeReader(languages=['en', 'ar'])
result = reader.read_gauge('path/to/gauge_photo.jpg')
print(f"Value: {result['value']}, Confidence: {result['confidence']:.0%}")
```

**Test Voice Input:**
```python
from src.mobile.voice_input import ArabicVoiceInput

voice = ArabicVoiceInput()
result = voice.listen_for_value(language='ar-DZ')
print(f"Recognized: {result['text']}, Value: {result['value']}")
```

**Test WhatsApp:**
```python
from src.integrations.whatsapp_alerts import WhatsAppAlerter

alerter = WhatsAppAlerter()
alerter.send_alert(
    to_number='whatsapp:+213XXXXXXXXX',
    machine_id='M001',
    failure_type='Heat Dissipation',
    probability=0.85,
    time_to_failure=12,
    language='ar'
)
```

---

## 🏭 Deployment Checklist

### Before Pilot (Week 1)
- [ ] Install dependencies on server PC
- [ ] Configure WhatsApp alerts
- [ ] Test mobile app on 2-3 phones
- [ ] Train 1-2 technicians (15 min each)
- [ ] Prepare machine ID list

### During Pilot (Week 2-4)
- [ ] Monitor daily sync status
- [ ] Review data quality weekly
- [ ] Collect technician feedback
- [ ] Track prediction accuracy

### Success Metrics
- Number of readings per day
- Data sync reliability (%)
- Technician satisfaction (1-10)
- Predicted failures vs actual

---

## ❓ Troubleshooting

### "Cannot import easyocr"
```powershell
pip install easyocr opencv-python --upgrade
```

### "Microphone not found"
- Check Windows microphone permissions
- Settings → Privacy → Microphone → Allow apps

### "WhatsApp sending failed"
- Verify Twilio credentials
- Check WhatsApp number format: `whatsapp:+213...`
- Ensure recipient has WhatsApp active

### "OCR not reading gauge correctly"
- Ensure good lighting
- Hold camera steady
- Try different angle
- Use manual entry as backup

---

## 📞 Support

For factory staff: Contact your system administrator
For administrators: Check `README.md` for full documentation

**Emergency**: Use manual entry if photo/voice fails
**Data loss**: All readings saved locally first, syncs later
