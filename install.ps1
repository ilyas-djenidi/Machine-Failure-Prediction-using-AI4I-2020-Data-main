# Installation Script for Algerian Factory Deployment
# Run this script to install all dependencies

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "  Predictive Maintenance System Installer" -ForegroundColor Cyan
Write-Host "  صيانة تنبؤية - Maintenance Prédictive" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

# Check Python version
Write-Host "[1/6] Checking Python version..." -ForegroundColor Yellow
$pythonVersion = python --version 2>&1
if ($pythonVersion -match "Python 3\.([8-9]|1[0-9])") {
    Write-Host "✓ Python version OK: $pythonVersion" -ForegroundColor Green
} else {
    Write-Host "✗ Python 3.8+ required. Current: $pythonVersion" -ForegroundColor Red
    Write-Host "Please install Python 3.8 or higher from python.org" -ForegroundColor Red
    exit 1
}

# Create virtual environment
Write-Host ""
Write-Host "[2/6] Creating virtual environment..." -ForegroundColor Yellow
if (Test-Path "venv") {
    Write-Host "Virtual environment already exists, skipping..." -ForegroundColor Gray
} else {
    python -m venv venv
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ Virtual environment created" -ForegroundColor Green
    } else {
        Write-Host "✗ Failed to create virtual environment" -ForegroundColor Red
        exit 1
    }
}

# Activate virtual environment
Write-Host ""
Write-Host "[3/6] Activating virtual environment..." -ForegroundColor Yellow
& "./venv/Scripts/Activate.ps1"
Write-Host "✓ Virtual environment activated" -ForegroundColor Green

# Upgrade pip
Write-Host ""
Write-Host "[4/6] Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip --quiet
Write-Host "✓ Pip upgraded" -ForegroundColor Green

# Install dependencies
Write-Host ""
Write-Host "[5/6] Installing dependencies..." -ForegroundColor Yellow
Write-Host "This may take several minutes..." -ForegroundColor Gray
Write-Host ""

# Install in stages for better error handling
Write-Host "  → Installing core ML libraries..." -ForegroundColor Cyan
pip install pandas numpy scikit-learn xgboost lightgbm --quiet
if ($LASTEXITCODE -ne 0) {
    Write-Host "✗ Failed to install core libraries" -ForegroundColor Red
    exit 1
}

Write-Host "  → Installing Mobile PWA dependencies..." -ForegroundColor Cyan
pip install streamlit pillow opencv-python --quiet
if ($LASTEXITCODE -ne 0) {
    Write-Host "✗ Failed to install PWA dependencies" -ForegroundColor Red
    exit 1
}

# EasyOCR requires special handling
Write-Host "  → Installing OCR engines (this may take longer)..." -ForegroundColor Cyan
pip install easyocr pytesseract --quiet
if ($LASTEXITCODE -ne 0) {
    Write-Host "⚠ OCR installation had issues, but continuing..." -ForegroundColor Yellow
}

Write-Host "  → Installing remaining dependencies..." -ForegroundColor Cyan
pip install -r requirements.txt --quiet
if ($LASTEXITCODE -ne 0) {
    Write-Host "⚠ Some packages failed, but core system should work" -ForegroundColor Yellow
} else {
    Write-Host "✓ All dependencies installed" -ForegroundColor Green
}

# Download EasyOCR models
Write-Host ""
Write-Host "[6/6] Setting up OCR models..." -ForegroundColor Yellow
Write-Host "Downloading Arabic and English OCR models (one-time setup)..." -ForegroundColor Gray
python -c "import easyocr; reader = easyocr.Reader(['en', 'ar'], gpu=False)" 2>$null
if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ OCR models downloaded" -ForegroundColor Green
} else {
    Write-Host "⚠ OCR models will download on first use" -ForegroundColor Yellow
}

# Create necessary directories
Write-Host ""
Write-Host "Creating data directories..." -ForegroundColor Yellow
New-Item -ItemType Directory -Force -Path "data/mobile_readings" | Out-Null
New-Item -ItemType Directory -Force -Path "models" | Out-Null
New-Item -ItemType Directory -Force -Path "reports" | Out-Null
Write-Host "✓ Directories created" -ForegroundColor Green

Write-Host ""
Write-Host "==========================================" -ForegroundColor Green
Write-Host "✓ Installation Complete!" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "1. Run mobile app: python -m streamlit run src/mobile/pwa_app.py" -ForegroundColor White
Write-Host "2. Or run main system: python -m streamlit run src/app.py" -ForegroundColor White
Write-Host ""
Write-Host "For WhatsApp alerts, set environment variables:" -ForegroundColor Yellow
Write-Host "  `$env:TWILIO_ACCOUNT_SID='your_account_sid'" -ForegroundColor Gray
Write-Host "  `$env:TWILIO_AUTH_TOKEN='your_auth_token'" -ForegroundColor Gray
Write-Host "  `$env:TWILIO_WHATSAPP_NUMBER='whatsapp:+14155238886'" -ForegroundColor Gray
Write-Host ""
Write-Host "Documentation: README.md" -ForegroundColor Cyan
Write-Host ""
