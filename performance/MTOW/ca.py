# file: performance/MTOW/ca.py
# desc: Calculate the wing and power loading constraints for the original Beechcraft 1900D aircraft.

# Global imports
import numpy as np

# Local imports
from global_constants import \
    G_0, rho_sl, C_d_0_orig, AR, e, k, S, V_TO, V_L, \
        V_stall_L, V_cruise, rho_cruise, ROC, V_climb, \
            eff_prop, mu_TO, S_g, S_land, LW_max, TOGA, \
                MTOW_orig

# Dynamic pressures
q_climb = 0.5 * rho_sl     * V_climb**2
q_c     = 0.5 * rho_cruise * V_cruise**2

WS = np.linspace(500, 6000, 200)

def constraint_curves(C_d_0, MTOW):
    CL_max_TO = MTOW / (0.5 * rho_sl * V_stall_L**2 * S)
    CL_TO     = MTOW / (0.5 * rho_sl * V_TO**2 * S)
    CL_max_L  = LW_max / (0.5 * rho_sl * V_stall_L**2 * S)

    # Take-off
    C_D_TO = C_d_0 + k * CL_TO**2
    TW_TO  = (1.21 / (G_0 * rho_sl * CL_max_TO * S_g)) * WS \
           + (0.605 / CL_TO) * (C_D_TO - mu_TO * CL_TO) + mu_TO
    PW_TO  = TW_TO * V_TO / eff_prop

    # Cruise
    TW_cr  = (q_c * C_d_0) / WS + (k / q_c) * WS
    PW_cr  = TW_cr * V_cruise / eff_prop

    # Climb
    TW_cl  = ROC / V_climb + (q_climb * C_d_0) / WS + k * WS / q_climb
    PW_cl  = TW_cl * V_climb / eff_prop

    # Limits
    WS_stall = 0.5 * rho_sl * V_stall_L**2 * CL_max_TO
    WS_land  = 0.5 * rho_sl * V_L**2       * CL_max_L

    # Design pt - stall line on climb curve
    PW_stall_int = np.interp(WS_stall, WS, PW_cl)

    # Design pt - cruise ⋂ climb
    diff = PW_cl - PW_cr
    idx  = np.where(np.diff(np.sign(diff)))[0]
    if idx.size:
        i       = idx[0]
        ws1, ws2   = WS[i], WS[i+1]
        pwc1, pwc2 = PW_cl[i], PW_cl[i+1]
        pwr1, pwr2 = PW_cr[i], PW_cr[i+1]
        slope      = (pwc2 - pwc1) - (pwr2 - pwr1)
        ws_x       = ws1 + (pwr1 - pwc1) * (ws2 - ws1) / slope
        pw_x       = np.interp(ws_x, WS, PW_cl)
    else:
        ws_x, pw_x = np.nan, np.nan

    PW_req = np.maximum.reduce([PW_TO, PW_cr, PW_cl])

    return dict(PW_TO=PW_TO, PW_cr=PW_cr, PW_cl=PW_cl,
                WS_stall=WS_stall, WS_land=WS_land,
                PW_stall_int=PW_stall_int,
                WS_x=ws_x, PW_x=pw_x,
                PW_req=PW_req,
                x_axis=WS)
