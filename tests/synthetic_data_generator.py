import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class SyntheticFactoryGenerator:
    def __init__(self, seed=42):
        np.random.seed(seed)
        self.sensor_config = {
            'temperature': {'base': 300, 'std': 2},
            'vibration': {'base': 1.0, 'std': 0.1},
            'current': {'base': 10, 'std': 1},
            'pressure': {'base': 100, 'std': 5},
            'rpm': {'base': 1500, 'std': 50}
        }

    def generate_machine_data(self, machine_id, machine_type, days, failures=[]):
        """
        Creates hourly sensor readings for a specified period.
        """
        hours = days * 24
        start_date = datetime.now() - timedelta(days=days)
        
        timestamps = [start_date + timedelta(hours=i) for i in range(hours)]
        data = {
            'timestamp': timestamps,
            'machine_id': [machine_id] * hours,
            'machine_type': [machine_type] * hours,
            'failure_label': [0] * hours
        }
        
        # Initialize sensors with baseline + noise
        for sensor, config in self.sensor_config.items():
            data[sensor] = np.random.normal(config['base'], config['std'], hours)
            
        df = pd.DataFrame(data)
        
        # Implement Daily Work Patterns (8am-6pm higher load)
        df['hour'] = df['timestamp'].dt.hour
        work_hours_mask = (df['hour'] >= 8) & (df['hour'] <= 18)
        df.loc[work_hours_mask, 'rpm'] += 200
        df.loc[work_hours_mask, 'temperature'] += 5
        
        # Inject deliberate failures
        for fail_day in failures:
            fail_idx = fail_day * 24 + np.random.randint(0, 24)
            if fail_idx < len(df):
                df.at[fail_idx, 'failure_label'] = 1
                self._inject_failure_patterns(df, fail_idx)
                
        return df

    def _inject_failure_patterns(self, df, fail_idx):
        """
        Add realistic failure signatures before the failure event.
        """
        pattern_type = np.random.choice(['HDF', 'wear', 'electrical', 'fade'])
        
        # Degradation period (e.g., 7 days = 168 hours)
        deg_period = 168
        start_deg = max(0, fail_idx - deg_period)
        
        if pattern_type == 'HDF':
            # Heat Dissipation Failure: Temperature rises 20-30%
            rise = np.linspace(0, 75, fail_idx - start_deg) # 300 * 0.25 approx 75
            df.loc[start_deg:fail_idx-1, 'temperature'] += rise
            
        elif pattern_type == 'wear':
            # Bearing Wear: Vibration increases 100-200%
            rise = np.linspace(0, 2, fail_idx - start_deg)
            df.loc[start_deg:fail_idx-1, 'vibration'] += rise
            
        elif pattern_type == 'electrical':
            # Electrical Fault: Current fluctuations with spikes
            df.loc[start_deg:fail_idx-1, 'current'] += np.random.normal(5, 5, fail_idx - start_deg)
            
        elif pattern_type == 'fade':
            # Power Fade: Gradual decline in RPM
            drop = np.linspace(0, -300, fail_idx - start_deg)
            df.loc[start_deg:fail_idx-1, 'rpm'] += drop

        # Sudden spike at failure
        df.at[fail_idx, 'temperature'] += 50
        df.at[fail_idx, 'vibration'] += 5

    def generate_factory(self, num_machines=10, days=90):
        """
        Generate data for multiple machines with random failures.
        """
        factory_data = []
        types = ['L', 'M', 'H']
        
        for i in range(num_machines):
            m_id = f"M_{1000 + i}"
            m_type = np.random.choice(types)
            
            # Randomly pick 0 to 2 failures over the period
            num_fails = np.random.randint(0, 3)
            fail_days = np.random.choice(range(10, days-1), num_fails, replace=False)
            
            m_data = self.generate_machine_data(m_id, m_type, days, failures=fail_days)
            factory_data.append(m_data)
            
        return pd.concat(factory_data, ignore_index=True)

if __name__ == "__main__":
    gen = SyntheticFactoryGenerator()
    factory_df = gen.generate_factory(num_machines=5, days=30)
    print(f"Generated {len(factory_df)} rows for {factory_df['machine_id'].nunique()} machines.")
    print(factory_df.head())
    
    # Save a sample to data directory
    output_path = r"c:\Users\HP\Desktop\Machine-Failure-Prediction-using-AI4I-2020-Data-main\data\synthetic_factory_data.csv"
    factory_df.to_csv(output_path, index=False)
    print(f"Sample saved to {output_path}")
