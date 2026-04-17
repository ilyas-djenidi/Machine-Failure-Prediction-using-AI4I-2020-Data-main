"""
Desktop Manual Entry Application for Tier 2 Factories
For factories with computers but no PLCs
Bilingual interface (Arabic/French)
"""

import streamlit as st
import pandas as pd
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Add parent to path
sys.path.append(str(Path(__file__).parent.parent))

# Bilingual labels
LABELS = {
    'ar': {
        'title': 'نظام إدخال البيانات اليدوي',
        'subtitle': 'للمصانع - إدخال دفعات القراءات',
        'shift': 'الوردية',
        'morning': 'صباحية',
        'afternoon': 'مسائية',
        'night': 'ليلية',
        'machine': 'الآلة',
        'temperature': 'درجة الحرارة (°C)',
        'pressure': 'الضغط (bar)',
        'vibration': 'الاهتزاز (mm/s)',
        'speed': 'السرعة (RPM)',
        'power': 'الطاقة (kW)',
        'current': 'التيار (A)',
        'submit': 'إرسال القراءات',
        'submit_all': 'إرسال جميع القراءات',
        'clear': 'مسح',
        'history': 'السجل',
        'export': 'تصدير إلى Excel',
        'notes': 'ملاحظات',
        'operator': 'اسم المشغل',
        'success': 'تم حفظ القراءات بنجاح!',
        'error': 'خطأ في حفظ البيانات',
        'warning': 'تحذير: قيمة غير عادية',
        'schedule': 'جدول القراءات',
        'next_reading': 'القراءة التالية',
        'overdue': 'متأخر',
        'due_soon': 'قريب'
    },
    'fr': {
        'title': 'Système de Saisie Manuelle',
        'subtitle': 'Pour Usines - Entrée par Lots',
        'shift': 'Poste',
        'morning': 'Matin',
        'afternoon': 'Après-midi',
        'night': 'Nuit',
        'machine': 'Machine',
        'temperature': 'Température (°C)',
        'pressure': 'Pression (bar)',
        'vibration': 'Vibration (mm/s)',
        'speed': 'Vitesse (RPM)',
        'power': 'Puissance (kW)',
        'current': 'Courant (A)',
        'submit': 'Soumettre Lectures',
        'submit_all': 'Soumettre Toutes',
        'clear': 'Effacer',
        'history': 'Historique',
        'export': 'Exporter vers Excel',
        'notes': 'Notes',
        'operator': 'Nom Opérateur',
        'success': 'Lectures sauvegardées avec succès!',
        'error': 'Erreur de sauvegarde',
        'warning': 'Attention: Valeur inhabituelle',
        'schedule': 'Planning des Lectures',
        'next_reading': 'Prochaine Lecture',
        'overdue': 'En retard',
        'due_soon': 'Bientôt'
    }
}

# Expected value ranges for validation
VALUE_RANGES = {
    'temperature': (0, 200),
    'pressure': (0, 50),
    'vibration': (0, 20),
    'speed': (0, 5000),
    'power': (0, 500),
    'current': (0, 100)
}


def init_database():
    """Initialize SQLite database for offline storage"""
    db_path = Path('data/desktop_readings.db')
    db_path.parent.mkdir(parents=True, exist_ok=True)
    
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    
    # Create readings table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS readings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            shift TEXT NOT NULL,
            machine_id TEXT NOT NULL,
            temperature REAL,
            pressure REAL,
            vibration REAL,
            speed REAL,
            power REAL,
            current REAL,
            notes TEXT,
            operator TEXT,
            synced INTEGER DEFAULT 0,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Create machines table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS machines (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            machine_id TEXT UNIQUE NOT NULL,
            name TEXT,
            type TEXT,
            location TEXT,
            reading_frequency INTEGER DEFAULT 480,
            last_reading TEXT,
            active INTEGER DEFAULT 1
        )
    ''')
    
    # Create schedules table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS reading_schedule (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            machine_id TEXT NOT NULL,
            shift TEXT NOT NULL,
            scheduled_time TEXT NOT NULL,
            completed INTEGER DEFAULT 0,
            completed_at TEXT,
            FOREIGN KEY (machine_id) REFERENCES machines(machine_id)
        )
    ''')
    
    conn.commit()
    conn.close()


def init_session_state():
    """Initialize session state"""
    if 'language' not in st.session_state:
        st.session_state.language = 'ar'
    if 'current_shift' not in st.session_state:
        st.session_state.current_shift = get_current_shift()
    if 'machines' not in st.session_state:
        st.session_state.machines = load_machines()


def get_label(key: str) -> str:
    """Get label in current language"""
    return LABELS[st.session_state.language].get(key, key)


def get_current_shift() -> str:
    """Auto-detect current shift based on time"""
    hour = datetime.now().hour
    if 6 <= hour < 14:
        return 'morning'
    elif 14 <= hour < 22:
        return 'afternoon'
    else:
        return 'night'


def load_machines() -> list:
    """Load machine list from database"""
    db_path = Path('data/desktop_readings.db')
    if not db_path.exists():
        # Return default machines if DB doesn't exist yet
        return [
            {'machine_id': 'M001', 'name': 'Compressor 1', 'type': 'Compressor'},
            {'machine_id': 'M002', 'name': 'Pump A', 'type': 'Pump'},
            {'machine_id': 'M003', 'name': 'Motor B', 'type': 'Motor'},
        ]
    
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    cursor.execute('SELECT machine_id, name, type FROM machines WHERE active = 1')
    machines = []
    for row in cursor.fetchall():
        machines.append({
            'machine_id': row[0],
            'name': row[1] or row[0],
            'type': row[2] or 'Unknown'
        })
    conn.close()
    
    return machines if machines else [
        {'machine_id': 'M001', 'name': 'Machine 1', 'type': 'Generic'}
    ]


def validate_value(sensor: str, value: float) -> tuple:
    """
    Validate sensor reading
    
    Returns:
        (is_valid, warning_message)
    """
    if sensor not in VALUE_RANGES:
        return (True, None)
    
    min_val, max_val = VALUE_RANGES[sensor]
    
    if value < min_val or value > max_val:
        return (False, f"{get_label('warning')}: {sensor} = {value}")
    
    # Check if value is near limits (warning zone)
    margin = (max_val - min_val) * 0.1
    if value < (min_val + margin) or value > (max_val - margin):
        return (True, f"⚠️ {sensor}: {value} (near limit)")
    
    return (True, None)


def save_readings(readings_data: list) -> bool:
    """Save readings to database"""
    try:
        db_path = Path('data/desktop_readings.db')
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        for reading in readings_data:
            cursor.execute('''
                INSERT INTO readings (
                    timestamp, shift, machine_id, temperature, pressure,
                    vibration, speed, power, current, notes, operator, synced
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
            ''', (
                reading['timestamp'],
                reading['shift'],
                reading['machine'],
                reading.get('temperature'),
                reading.get('pressure'),
                reading.get('vibration'),
                reading.get('speed'),
                reading.get('power'),
                reading.get('current'),
                reading.get('notes', ''),
                reading.get('operator', ''),
            ))
            
            # Update last reading time
            cursor.execute('''
                UPDATE machines 
                SET last_reading = ? 
                WHERE machine_id = ?
            ''', (reading['timestamp'], reading['machine']))
        
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        st.error(f"Database error: {e}")
        return False


def get_recent_readings(limit: int = 50) -> pd.DataFrame:
    """Get recent readings from database"""
    db_path = Path('data/desktop_readings.db')
    if not db_path.exists():
        return pd.DataFrame()
    
    conn = sqlite3.connect(str(db_path))
    query = '''
        SELECT timestamp, shift, machine_id, temperature, pressure,
               vibration, speed, power, current, operator
        FROM readings
        ORDER BY timestamp DESC
        LIMIT ?
    '''
    df = pd.read_sql_query(query, conn, params=(limit,))
    conn.close()
    
    return df


def export_to_excel(df: pd.DataFrame, filename: str = 'readings_export.xlsx'):
    """Export readings to Excel"""
    output_path = Path('exports') / filename
    output_path.parent.mkdir(exist_ok=True)
    
    df.to_excel(str(output_path), index=False, engine='openpyxl')
    return output_path


def main():
    """Main desktop entry application"""
    
    # Initialize
    init_database()
    init_session_state()
    
    # Page config
    st.set_page_config(
        page_title="Desktop Entry System",
        page_icon="🏭",
        layout="wide"
    )
    
    # Header with language selector
    col1, col2 = st.columns([4, 1])
    with col1:
        st.title(get_label('title'))
        st.caption(get_label('subtitle'))
    with col2:
        lang = st.selectbox(
            '🌐 Language',
            ['العربية', 'Français'],
            index=0 if st.session_state.language == 'ar' else 1,
            label_visibility='collapsed'
        )
        st.session_state.language = 'ar' if lang == 'العربية' else 'fr'
    
    st.divider()
    
    # Tabs
    tab1, tab2, tab3 = st.tabs([
        f"📝 {get_label('submit')}",
        f"📊 {get_label('history')}",
        f"📅 {get_label('schedule')}"
    ])
    
    with tab1:
        # Batch entry form
        st.subheader(f"📝 {get_label('shift')} - {get_label(st.session_state.current_shift).upper()}")
        
        # Operator name
        operator = st.text_input(get_label('operator'), placeholder="Ahmed, Fatima, etc.")
        
        # Shift selector
        shift_options = ['morning', 'afternoon', 'night']
        shift_labels = [get_label(s) for s in shift_options]
        selected_shift = st.selectbox(
            get_label('shift'),
            shift_options,
            format_func=lambda x: get_label(x),
            index=shift_options.index(st.session_state.current_shift)
        )
        
        st.divider()
        
        # Multi-machine entry
        readings_to_submit = []
        
        for i, machine in enumerate(st.session_state.machines):
            with st.expander(f"🔧 {machine['machine_id']} - {machine['name']}", expanded=i==0):
                cols = st.columns([2, 2, 2])
                
                readings = {}
                warnings = []
                
                with cols[0]:
                    temp = st.number_input(
                        get_label('temperature'),
                        min_value=0.0,
                        max_value=300.0,
                        step=0.1,
                        key=f"temp_{machine['machine_id']}"
                    )
                    if temp > 0:
                        readings['temperature'] = temp
                        valid, warning = validate_value('temperature', temp)
                        if warning:
                            warnings.append(warning)
                    
                    pressure = st.number_input(
                        get_label('pressure'),
                        min_value=0.0,
                        max_value=100.0,
                        step=0.1,
                        key=f"pressure_{machine['machine_id']}"
                    )
                    if pressure > 0:
                        readings['pressure'] = pressure
                        valid, warning = validate_value('pressure', pressure)
                        if warning:
                            warnings.append(warning)
                
                with cols[1]:
                    vibration = st.number_input(
                        get_label('vibration'),
                        min_value=0.0,
                        max_value=50.0,
                        step=0.1,
                        key=f"vibration_{machine['machine_id']}"
                    )
                    if vibration > 0:
                        readings['vibration'] = vibration
                        valid, warning = validate_value('vibration', vibration)
                        if warning:
                            warnings.append(warning)
                    
                    speed = st.number_input(
                        get_label('speed'),
                        min_value=0.0,
                        max_value=10000.0,
                        step=1.0,
                        key=f"speed_{machine['machine_id']}"
                    )
                    if speed > 0:
                        readings['speed'] = speed
                        valid, warning = validate_value('speed', speed)
                        if warning:
                            warnings.append(warning)
                
                with cols[2]:
                    power = st.number_input(
                        get_label('power'),
                        min_value=0.0,
                        max_value=1000.0,
                        step=0.1,
                        key=f"power_{machine['machine_id']}"
                    )
                    if power > 0:
                        readings['power'] = power
                        valid, warning = validate_value('power', power)
                        if warning:
                            warnings.append(warning)
                    
                    current = st.number_input(
                        get_label('current'),
                        min_value=0.0,
                        max_value=500.0,
                        step=0.1,
                        key=f"current_{machine['machine_id']}"
                    )
                    if current > 0:
                        readings['current'] = current
                        valid, warning = validate_value('current', current)
                        if warning:
                            warnings.append(warning)
                
                # Notes
                notes = st.text_area(
                    get_label('notes'),
                    placeholder="Any observations...",
                    key=f"notes_{machine['machine_id']}",
                    height=60
                )
                
                # Show warnings
                if warnings:
                    for warning in warnings:
                        st.warning(warning)
                
                # Add to submission list if has readings
                if readings:
                    readings_to_submit.append({
                        'machine': machine['machine_id'],
                        'shift': selected_shift,
                        'timestamp': datetime.now().isoformat(),
                        'operator': operator,
                        'notes': notes,
                        **readings
                    })
        
        # Submit button
        st.divider()
        col1, col2 = st.columns([3, 1])
        with col1:
            if st.button(f"✅ {get_label('submit_all')} ({len(readings_to_submit)} machines)", 
                        use_container_width=True, type="primary"):
                if not operator:
                    st.error("❌ " + get_label('operator') + " required")
                elif not readings_to_submit:
                    st.error("❌ No readings entered")
                else:
                    if save_readings(readings_to_submit):
                        st.success(f"✅ {get_label('success')} ({len(readings_to_submit)} machines)")
                        st.balloons()
                    else:
                        st.error(f"❌ {get_label('error')}")
        
        with col2:
            if st.button(f"🗑️ {get_label('clear')}", use_container_width=True):
                st.rerun()
    
    with tab2:
        # History view
        st.subheader(f"📊 {get_label('history')}")
        
        df = get_recent_readings(limit=100)
        
        if not df.empty:
            # Display recent readings
            st.dataframe(df, use_container_width=True, hide_index=True)
            
            # Export button
            if st.button(f"📥 {get_label('export')}", use_container_width=True):
                try:
                    filename = f"readings_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
                    export_path = export_to_excel(df, filename)
                    st.success(f"✅ Exported to: {export_path}")
                except Exception as e:
                    st.error(f"Export failed: {e}")
        else:
            st.info("📭 No readings yet")
    
    with tab3:
        # Reading schedule
        st.subheader(f"📅 {get_label('schedule')}")
        st.info("🚧 Schedule feature coming soon - will show reading reminders per machine")


if __name__ == "__main__":
    main()
