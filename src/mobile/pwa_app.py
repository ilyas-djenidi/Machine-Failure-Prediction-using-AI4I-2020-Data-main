"""
Mobile Progressive Web App for Data Collection
Works offline with photo-based gauge reading
"""

import streamlit as st
import pandas as pd
from datetime import datetime
from pathlib import Path
import sys
import json
import logging

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from mobile.ocr_gauge_reader import GaugeReader
from mobile.voice_input import ArabicVoiceInput

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Page configuration
st.set_page_config(
    page_title="Maintenance Mobile",
    page_icon="🔧",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Bilingual labels
LABELS = {
    'ar': {
        'title': 'نظام الصيانة التنبؤية',
        'subtitle': 'جمع البيانات الميدانية',
        'machine_id': 'رقم الآلة',
        'reading_type': 'نوع القراءة',
        'capture_photo': 'التقط صورة',
        'use_voice': 'استخدم الصوت',
        'manual_entry': 'إدخال يدوي',
        'value': 'القيمة',
        'unit': 'الوحدة',
        'submit': 'إرسال',
        'history': 'السجل',
        'sync_status': 'حالة المزامنة',
        'offline': 'غير متصل',
        'synced': 'متزامن',
        'pending': 'في الانتظار'
    },
    'fr': {
        'title': 'Système de Maintenance Prédictive',
        'subtitle': 'Collecte de Données sur le Terrain',
        'machine_id': 'ID Machine',
        'reading_type': 'Type de Lecture',
        'capture_photo': 'Prendre une Photo',
        'use_voice': 'Utiliser la Voix',
        'manual_entry': 'Saisie Manuelle',
        'value': 'Valeur',
        'unit': 'Unité',
        'submit': 'Soumettre',
        'history': 'Historique',
        'sync_status': 'État de Synchro',
        'offline': 'Hors ligne',
        'synced': 'Synchronisé',
        'pending': 'En attente'
    }
}


def init_session_state():
    """Initialize session state variables"""
    if 'language' not in st.session_state:
        st.session_state.language = 'ar'
    if 'readings' not in st.session_state:
        st.session_state.readings = []
    if 'ocr_reader' not in st.session_state:
        st.session_state.ocr_reader = None
    if 'voice_input' not in st.session_state:
        st.session_state.voice_input = None


def get_label(key: str) -> str:
    """Get label in current language"""
    lang = st.session_state.language
    return LABELS[lang].get(key, key)


def save_reading_offline(reading: dict):
    """Save reading to local storage (simulated with session state)"""
    reading['timestamp'] = datetime.now().isoformat()
    reading['synced'] = False
    st.session_state.readings.append(reading)
    
    # Also save to local file for persistence
    data_dir = Path('data/mobile_readings')
    data_dir.mkdir(parents=True, exist_ok=True)
    
    filename = data_dir / f"readings_{datetime.now().strftime('%Y%m%d')}.json"
    
    try:
        if filename.exists():
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
        else:
            data = []
        
        data.append(reading)
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            
        logger.info(f"✓ Reading saved offline: {filename}")
        return True
    except Exception as e:
        logger.error(f"Error saving reading: {e}")
        return False


def main():
    """Main PWA application"""
    init_session_state()
    
    # Language selector (compact)
    col1, col2 = st.columns([3, 1])
    with col2:
        lang = st.selectbox('🌐', ['العربية', 'Français'], 
                           label_visibility='collapsed',
                           index=0 if st.session_state.language == 'ar' else 1)
        st.session_state.language = 'ar' if lang == 'العربية' else 'fr'
    
    # Header
    st.title(get_label('title'))
    st.caption(get_label('subtitle'))
    
    # Offline indicator
    st.info(f"📴 {get_label('offline')} - {len(st.session_state.readings)} {get_label('pending')}")
    
    st.divider()
    
    # Data entry form
    with st.form("reading_form", clear_on_submit=True):
        # Machine ID
        machine_id = st.text_input(get_label('machine_id'), 
                                   placeholder="M001",
                                   help="Enter machine identifier")
        
        # Reading type
        reading_types = {
            'ar': ['درجة الحرارة', 'الضغط', 'الاهتزاز', 'السرعة', 'أخرى'],
            'fr': ['Température', 'Pression', 'Vibration', 'Vitesse', 'Autre']
        }
        reading_type = st.selectbox(
            get_label('reading_type'),
            reading_types[st.session_state.language]
        )
        
        # Data entry method tabs
        tab1, tab2, tab3 = st.tabs([
            f"📸 {get_label('capture_photo')}", 
            f"🎤 {get_label('use_voice')}",
            f"⌨️ {get_label('manual_entry')}"
        ])
        
        value = None
        confidence = 1.0
        entry_method = 'manual'
        
        with tab1:
            # Photo capture
            st.caption("📸 Prenez une photo du cadran / التقط صورة للمؤشر")
            uploaded_file = st.file_uploader(
                "Upload gauge photo",
                type=['jpg', 'jpeg', 'png'],
                label_visibility='collapsed'
            )
            
            if uploaded_file is not None:
                # Display image
                st.image(uploaded_file, caption="Gauge Photo", use_container_width=True)
                
                # Process with OCR
                if st.form_submit_button("🔍 Read Value / قراءة القيمة"):
                    with st.spinner("Processing..."):
                        # Save uploaded file temporarily
                        temp_path = Path(f"temp_{uploaded_file.name}")
                        with open(temp_path, 'wb') as f:
                            f.write(uploaded_file.getbuffer())
                        
                        # Initialize OCR reader if needed
                        if st.session_state.ocr_reader is None:
                            st.session_state.ocr_reader = GaugeReader(languages=['en', 'ar', 'fr'])
                        
                        # Read gauge
                        result = st.session_state.ocr_reader.read_gauge(str(temp_path))
                        
                        # Clean up
                        temp_path.unlink(missing_ok=True)
                        
                        if result['success']:
                            value = result['value']
                            confidence = result['confidence']
                            entry_method = 'photo'
                            st.success(f"✓ Detected: {value} (confidence: {confidence:.0%})")
                            if result.get('warning'):
                                st.warning(result['warning'])
                        else:
                            st.error(f"❌ {result.get('error', 'Failed to read gauge')}")
        
        with tab2:
            # Voice input
            st.caption("🎤 Dites la valeur / قل القيمة")
            
            if st.form_submit_button("🎤 Start Recording / ابدأ التسجيل"):
                with st.spinner("Listening..."):
                    if st.session_state.voice_input is None:
                        st.session_state.voice_input = ArabicVoiceInput()
                    
                    lang_code = 'ar-DZ' if st.session_state.language == 'ar' else 'fr-FR'
                    result = st.session_state.voice_input.listen_for_value(language=lang_code)
                    
                    if result['success']:
                        value = result['value']
                        entry_method = 'voice'
                        st.success(f"✓ Recognized: {result['text']}")
                        if value:
                            st.info(f"Value: {value}")
                        else:
                            st.warning("Could not extract numeric value. Please use manual entry.")
                    else:
                        st.error(f"❌ {result.get('error', 'Voice recognition failed')}")
        
        with tab3:
            # Manual entry
            value_manual = st.number_input(
                get_label('value'),
                min_value=0.0,
                step=0.1,
                format="%.2f"
            )
            if value is None:
                value = value_manual
        
        # Unit selection
        units = ['°C', 'bar', 'PSI', 'Hz', 'RPM', 'mm/s', 'g', 'kW', 'A', 'V']
        unit = st.selectbox(get_label('unit'), units)
        
        # Notes (optional)
        notes = st.text_area("Notes (optional)", placeholder="Any observations...")
        
        # Submit button
        submitted = st.form_submit_button(
            f"✅ {get_label('submit')}",
            use_container_width=True,
            type="primary"
        )
        
        if submitted:
            if not machine_id:
                st.error("❌ Machine ID is required")
            elif value is None or value == 0:
                st.error("❌ Please enter a value")
            else:
                # Create reading record
                reading = {
                    'machine_id': machine_id,
                    'reading_type': reading_type,
                    'value': value,
                    'unit': unit,
                    'notes': notes,
                    'entry_method': entry_method,
                    'confidence': confidence,
                    'language': st.session_state.language
                }
                
                # Save offline
                if save_reading_offline(reading):
                    st.success(f"✅ Reading saved! Total: {len(st.session_state.readings)}")
                    st.balloons()
                else:
                    st.error("❌ Failed to save reading")
    
    # Recent readings history
    if st.session_state.readings:
        st.divider()
        st.subheader(get_label('history'))
        
        # Display recent readings
        df = pd.DataFrame(st.session_state.readings[-10:])  # Last 10
        
        # Format for display
        display_df = df[['machine_id', 'reading_type', 'value', 'unit', 'timestamp']].copy()
        display_df['timestamp'] = pd.to_datetime(display_df['timestamp']).dt.strftime('%H:%M')
        
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        
        # Sync button (placeholder - will implement actual sync later)
        if st.button("🔄 Sync when WiFi available / المزامنة عند توفر الواي فاي", 
                     use_container_width=True):
            st.info("📡 Will sync automatically when WiFi is detected")


if __name__ == "__main__":
    main()
