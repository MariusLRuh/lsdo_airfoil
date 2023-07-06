from torch import nn


scaler_valued_nn_model = nn.Sequential(
            nn.Linear(35, 180), nn.ReLU(),
            nn.Linear(180, 160), nn.SELU(),
            nn.Linear(160, 140), nn.SELU(),
            nn.Linear(140, 120), nn.SELU(),
            nn.Linear(120, 100), nn.GELU(),
            nn.Linear(100, 80), nn.LeakyReLU(),
            nn.Linear(80, 1),
)

scaler_valued_nn_model_gelu = nn.Sequential(
            nn.Linear(35, 150), nn.GELU(),
            nn.Linear(150, 150), nn.GELU(),
            nn.Linear(150, 200), nn.ReLU(),
            # nn.Dropout(0.2),
            nn.Linear(200, 200), nn.GELU(),
            nn.Linear(200, 200), nn.ELU(),
            nn.Linear(200, 200), nn.GELU(),
            # nn.Dropout(0.2),
            nn.Linear(200, 200), nn.GELU(),
            nn.Linear(200, 120), nn.GELU(),
            nn.Linear(120, 100), nn.GELU(),
            nn.Linear(100, 1),
)