import os
import pickle
import numpy as np
from CoolProp.CoolProp import PropsSI
from scipy.interpolate import RegularGridInterpolator

PICKLE_PATH = os.path.join(os.path.dirname(__file__), "eglycol60_cp.pkl")

def _generate_and_save_all_props():
    T_coarse = np.linspace(300, 370, 100)  # Kelvin
    P_coarse = np.linspace(1e5, 5e6, 100)   # Pa
    fluid_type = 'INCOMP::MEG-60%'

    cp = np.zeros((len(T_coarse), len(P_coarse)))
    k = np.zeros_like(cp)
    dyn_visc = np.zeros_like(cp)
    gamma = np.zeros_like(cp)
    Pr = np.zeros_like(cp)
    rho = np.zeros_like(cp)

    for i, T in enumerate(T_coarse):
        for j, P in enumerate(P_coarse):
            cp[i, j] = PropsSI('C', 'P', P, 'T', T, fluid_type)
            k[i, j] = PropsSI('CONDUCTIVITY', 'P', P, 'T', T, fluid_type)
            dyn_visc[i, j] = PropsSI('VISCOSITY', 'P', P, 'T', T, fluid_type)
            cp0 = PropsSI('Cpmass', 'P', P, 'T', T, fluid_type)
            cv = PropsSI('Cvmass', 'P', P, 'T', T, fluid_type)
            gamma[i, j] = cp0 / cv if cv != 0 else np.nan
            Pr[i, j] = PropsSI('PRANDTL', 'P', P, 'T', T, fluid_type)
            rho[i, j] = PropsSI('D', 'P', P, 'T', T, fluid_type)

    T_fine = np.linspace(273-48, 273+250, 1000)   # Kelvin
    P_fine = np.linspace(1e2, 60e5, 1000)   # Pa

    def interp_on_fine_grid(data):
        interp = RegularGridInterpolator((T_coarse, P_coarse), data, bounds_error=False, fill_value=None)
        TT, PP = np.meshgrid(T_fine, P_fine, indexing='ij')
        points = np.stack([TT.ravel(), PP.ravel()], axis=-1)
        return interp(points).reshape(TT.shape)

    cp_fine = np.clip(interp_on_fine_grid(cp), 0, None)
    k_fine = np.clip(interp_on_fine_grid(k), 0, None)
    dyn_visc_fine = np.clip(interp_on_fine_grid(dyn_visc), 0.001, None)
    gamma_fine = np.clip(interp_on_fine_grid(gamma), 0, None)
    Pr_fine = np.clip(interp_on_fine_grid(Pr), 0.001, None)
    rho_fine = np.clip(interp_on_fine_grid(rho), 0, None)

    props = {
        "T": T_fine,  # Kelvin
        "P": P_fine,  # Pa
        "CP": cp_fine, # J/kg.K
        "K_k": k_fine,
        "DYN_VISC": dyn_visc_fine,
        "GAMMA": gamma_fine,
        "PR": Pr_fine,
        "RHO": rho_fine
    }

    with open(PICKLE_PATH, "wb") as f:
        pickle.dump(props, f)

def _load_all_props():
    if not os.path.exists(PICKLE_PATH):
        _generate_and_save_all_props()
    with open(PICKLE_PATH, "rb") as f:
        return pickle.load(f)


props_tms = _load_all_props()
k = props_tms['K_k']  # J/(kg·K)
T = props_tms['T']  # Kelvin
P = props_tms['P']  # Pa
P_i = 2e5
T_i = 200

i_T = np.searchsorted(T, T_i)
i_P = np.searchsorted(P, P_i)
cp_0 = k[i_T, i_P]
