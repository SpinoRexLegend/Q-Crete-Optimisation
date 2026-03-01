import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_yard_data(num_rows: int = 4000) -> pd.DataFrame:
    np.random.seed(42)
    slab_ids: list[str] = [f"SLAB_{i:04d}" for i in range(1, num_rows + 1)]
    end_date: datetime = datetime.now()
    start_date: datetime = end_date - timedelta(days=365)
    random_seconds: np.ndarray = np.random.randint(0, int((end_date - start_date).total_seconds()), size=num_rows)
    pour_times: list[datetime] = [start_date + timedelta(seconds=int(s)) for s in random_seconds]
    pour_times.sort()
    
    ambient_temps: np.ndarray = np.random.uniform(25.0, 45.0, size=num_rows)
    humidities: np.ndarray = np.random.uniform(40.0, 95.0, size=num_rows)
    wc_ratios: np.ndarray = np.random.uniform(0.35, 0.55, size=num_rows)
    
    df: pd.DataFrame = pd.DataFrame({
        'Slab_ID': slab_ids,
        'Pour_Time': pour_times,
        'Ambient_Temp': ambient_temps,
        'Humidity': humidities,
        'Water_Cement_Ratio': wc_ratios
    })
    
    df['Time_Since_Pour'] = (end_date - pd.to_datetime(df['Pour_Time'])).dt.total_seconds() / 3600.0
    
    base_strength: pd.Series = 100.0 / (1.5 ** df['Water_Cement_Ratio'])
    temp_kelvin: pd.Series = df['Ambient_Temp'] + 273.15
    e_a: float = 40000.0
    r: float = 8.314
    t_ref: float = 293.15
    
    exp_factor: pd.Series = np.exp(-(e_a / r) * ((1.0 / temp_kelvin) - (1.0 / t_ref)))
    humidity_factor: pd.Series = 0.9 + (df['Humidity'] / 950.0)
    
    df['Strength'] = base_strength * exp_factor * humidity_factor
    noise: np.ndarray = np.random.uniform(0.95, 1.05, size=num_rows)
    df['Strength'] = df['Strength'] * noise
    
    df['Ambient_Temp'] = df['Ambient_Temp'].round(2)
    df['Humidity'] = df['Humidity'].round(2)
    df['Water_Cement_Ratio'] = df['Water_Cement_Ratio'].round(3)
    df['Strength'] = df['Strength'].round(2)
    df['Time_Since_Pour'] = df['Time_Since_Pour'].round(2)
    
    return df

if __name__ == "__main__":
    dataset: pd.DataFrame = generate_yard_data(4000)
    output_filename: str = "yard_data.csv"
    dataset.to_csv(output_filename, index=False)
