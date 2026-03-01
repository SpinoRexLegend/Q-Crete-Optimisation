import torch
import pandas as pd
import numpy as np
from datetime import datetime
import streamlit as st
from train_pinn import PINN
from typing import List

class Constants:
    TEMP: str = 'Ambient_Temp'
    HUMIDITY: str = 'Humidity'
    TIME_SINCE_POUR: str = 'Time_Since_Pour'
    POUR_TIME: str = 'Pour_Time'
    STRENGTH: str = 'Strength'
    CURRENT_STRENGTH: str = 'Current_Strength'
    READINESS: str = 'Readiness_Percent'
    FEATURES: List[str] = [TEMP, HUMIDITY, TIME_SINCE_POUR]

class PhysicsEngine:
    def __init__(self, model_path: str = 'pinn_strength_model.pth') -> None:
        self.model: PINN = self._load_model(model_path)

    @staticmethod
    @st.cache_resource
    def _load_model(path: str) -> PINN:
        model: PINN = PINN()
        model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
        model.eval()
        return model

    def predict_strength(self, df: pd.DataFrame, temp_mod: float, hum_mod: float) -> pd.DataFrame:
        dynamic_slabs: pd.DataFrame = df.copy(deep=True)
        
        if Constants.TIME_SINCE_POUR not in dynamic_slabs.columns:
            dynamic_slabs[Constants.TIME_SINCE_POUR] = (datetime.now() - pd.to_datetime(dynamic_slabs[Constants.POUR_TIME])).dt.total_seconds() / 3600.0
            
        dynamic_slabs[Constants.TEMP] = np.clip(dynamic_slabs[Constants.TEMP] + temp_mod, 10.0, 55.0)
        dynamic_slabs[Constants.HUMIDITY] = np.clip(dynamic_slabs[Constants.HUMIDITY] + hum_mod, 10.0, 100.0)

        x_sample: np.ndarray = dynamic_slabs[Constants.FEATURES].values
        x_max: np.ndarray = np.max(x_sample, axis=0)
        x_min: np.ndarray = np.min(x_sample, axis=0)
        
        diff: np.ndarray = x_max - x_min
        diff[diff == 0] = 1.0

        x_sample_norm: np.ndarray = (x_sample - x_min) / diff
        x_tensor: torch.Tensor = torch.tensor(x_sample_norm, dtype=torch.float32)

        with torch.no_grad():
            y_pred_norm: np.ndarray = self.model(x_tensor).numpy()

        if Constants.STRENGTH in dynamic_slabs.columns:
            y_raw: np.ndarray = dynamic_slabs[Constants.STRENGTH].values
            y_max, y_min = float(np.max(y_raw)), float(np.min(y_raw))
        else:
            y_max, y_min = 250.0, 0.0

        dynamic_slabs[Constants.CURRENT_STRENGTH] = (np.squeeze(y_pred_norm) * (y_max - y_min)) + y_min
        dynamic_slabs[Constants.READINESS] = np.clip((dynamic_slabs[Constants.CURRENT_STRENGTH] / 250.0) * 100.0, 0.0, 100.0)

        return dynamic_slabs
