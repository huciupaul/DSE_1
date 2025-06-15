import numpy as np
from CoolProp.CoolProp import PropsSI

R_AIR = 287.05                                      # [J/(kg·K)]
G_0 = 9.80665                                       # [m/s²]
LAPSE = -0.0065                                     # [K/m]
k_air = 1.4                                         # [-] Specific heat ratio of air
MAXC = 2*906e3                                      # [W] Max Continuous Power
E_SCALE = 0.0783632513412531                        # [-] Scale factor for the error