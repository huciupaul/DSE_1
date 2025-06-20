import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
import pickle
# ── 1.  read the CSV ─────────────────────────────────────────────
df = pd.read_csv(r"C:\Users\zksie\OneDrive - Delft University of Technology\Documents\Academic Year 2024-2025\Q4 - DSE\Emissions_tool\DSE_1\RQL_1D\total_nox_map.csv")
water_grid = df.iloc[:, 0].to_numpy(float)          # first column
power_grid = df.columns[1:].astype(float).to_numpy()# header row (skip col-0)
nox_grid   = df.iloc[:, 1:].to_numpy(float)         # 2-D array (water × power)
# sanity check shapes:  (N_water, N_power)
print(water_grid.shape, power_grid.shape, nox_grid.shape)
# ── 2.  build the interpolator ───────────────────────────────────
interp = RegularGridInterpolator(
    (water_grid, power_grid),   # note order: y, x
    nox_grid,
    bounds_error=False,         # return nan outside domain
    fill_value=np.nan,
)
# ── 3.  example usage ────────────────────────────────────────────
water_flow  = 0.025        # kg/s
power_ratio = 0.97         # P / P_baseline
nox_val = interp((water_flow, power_ratio))
print(f"TOTAL NOx at (w={water_flow:.3f} kg/s, P/P₀={power_ratio:.2f}) ≈ "
      f"{nox_val:.9f} ")








# Pickleeeeeeeeeeeee the interpolator
with open('data/interpolants/NOx_interpolator.pkl', 'wb') as f:
    pickle.dump(interp, f, protocol=pickle.HIGHEST_PROTOCOL)