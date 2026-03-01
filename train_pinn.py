import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from datetime import datetime

class PINN(nn.Module):
    def __init__(self) -> None:
        super(PINN, self).__init__()
        self.net: nn.Sequential = nn.Sequential(
            nn.Linear(3, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

def physics_loss(model: nn.Module, inputs: torch.Tensor) -> torch.Tensor:
    predictions: torch.Tensor = model(inputs)
    gradients: torch.Tensor = torch.autograd.grad(
        outputs=predictions, 
        inputs=inputs, 
        grad_outputs=torch.ones_like(predictions), 
        create_graph=True
    )[0]
    grad_time: torch.Tensor = gradients[:, 2]
    penalty_monotonicity: torch.Tensor = torch.relu(-grad_time).mean()
    penalty_bounds: torch.Tensor = torch.relu(-predictions).mean()
    return penalty_monotonicity + 0.1 * penalty_bounds

def main() -> None:
    df: pd.DataFrame = pd.read_csv('yard_data.csv')
    df['Pour_Time'] = pd.to_datetime(df['Pour_Time'])
    df['Time_Since_Pour'] = (datetime.now() - df['Pour_Time']).dt.total_seconds() / 3600.0
    
    x_vals: np.ndarray = df[['Ambient_Temp', 'Humidity', 'Time_Since_Pour']].values
    y_vals: np.ndarray = df['Strength'].values
    
    x_max: np.ndarray = np.max(x_vals, axis=0)
    x_min: np.ndarray = np.min(x_vals, axis=0)
    x_norm: np.ndarray = (x_vals - x_min) / (x_max - x_min)
    
    y_max: float = float(np.max(y_vals))
    y_min: float = float(np.min(y_vals))
    y_norm: np.ndarray = (y_vals - y_min) / (y_max - y_min)
    
    x_tensor: torch.Tensor = torch.tensor(x_norm, dtype=torch.float32)
    y_tensor: torch.Tensor = torch.tensor(y_norm, dtype=torch.float32).view(-1, 1)
    
    x_tensor.requires_grad = True
    
    model: PINN = PINN()
    criterion: nn.MSELoss = nn.MSELoss()
    optimizer: optim.Adam = optim.Adam(model.parameters(), lr=0.005)
    
    epochs: int = 1000
    lambda_physics: float = 0.5
    
    for epoch in range(epochs):
        predictions: torch.Tensor = model(x_tensor)
        data_loss: torch.Tensor = criterion(predictions, y_tensor)
        phys_loss: torch.Tensor = physics_loss(model, x_tensor)
        total_loss: torch.Tensor = data_loss + lambda_physics * phys_loss
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
    output_file: str = 'pinn_strength_model.pth'
    torch.save(model.state_dict(), output_file)

if __name__ == "__main__":
    main()
