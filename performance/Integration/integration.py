import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import matplotlib.pyplot as plt

from global_constants import Beechcraft_1900D, seat_pitch, M_PAX, \
        X_most_fwd, X_most_aft, X_cargo_fwd, X_first_seat, \
        X_wing_end, V_wing, X_EPS, \
        X_wing, M_cargo_fwd, M_cargo_per_PAX
from performance.Integration import aft_configuration as acfg

def X_PAX(num_PAX) -> float:
    """Calculate the center of gravity position of the passenger seats."""
    return ((num_PAX-3)*M_PAX * (X_first_seat + (num_PAX-5)/2 * seat_pitch / 2) + 3*M_PAX * (X_first_seat + (num_PAX-3)/2 * seat_pitch)) / (num_PAX*M_PAX)


def X_OEW(M_FC, M_TMS_fwd, M_TMS_aft, M_TMS_mid, M_EPS, M_tank, X_tank_TMS, X_tank_front) -> Tuple[float, float]:
    """Calculate the center of gravity position of the aircraft's empty weight."""
    X_OEW_og = Beechcraft_1900D['X_OEW']

    OEW_H2D2 = Beechcraft_1900D['OEW'] + M_FC + M_EPS + M_TMS_fwd + M_TMS_aft + M_tank + M_TMS_mid
    X_OEW_H2D2 = (Beechcraft_1900D['OEW'] * X_OEW_og + M_EPS * X_EPS + (M_FC + M_TMS_fwd) * X_wing + (M_TMS_aft + M_tank) * X_tank_TMS + M_TMS_mid * (X_wing_end + X_tank_front) / 2) / OEW_H2D2

    return OEW_H2D2, X_OEW_H2D2

def __cargo(X_cargo_aft: float,
            M_cargo_aft: float,
            X_OEW: float,
            OEW: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    
    """Calculate the center of gravity position and weight of OEW+cargo."""
    W_cargo_front = np.zeros(3)
    X_cargo_front = np.zeros(3)
    W_cargo_back = np.zeros(3)
    X_cargo_back = np.zeros(3)

    """Initial point: OEW"""
    W_cargo_front[0] = OEW
    X_cargo_front[0] = X_OEW
    W_cargo_back[0] = W_cargo_front[0]
    X_cargo_back[0] = X_cargo_front[0]

    """Cargo loaded: front or aft first"""
    W_cargo_front[1] = W_cargo_front[0] + M_cargo_fwd
    X_cargo_front[1] = ((X_cargo_fwd * M_cargo_fwd) + (X_OEW * OEW)) / W_cargo_front[1]
    W_cargo_back[1] = OEW + M_cargo_aft
    X_cargo_back[1] = ((X_cargo_aft * M_cargo_aft) + (X_OEW * OEW)) / W_cargo_back[1]

    """Cargo loaded: the rest"""
    W_cargo_front[2] = W_cargo_front[1] + M_cargo_aft
    X_cargo_front[2] = ((X_cargo_aft * M_cargo_aft) + (X_cargo_front[1] * W_cargo_front[1])) / W_cargo_front[2]
    W_cargo_back[2] = W_cargo_front[2]
    X_cargo_back[2] = X_cargo_front[2]

    return X_cargo_front, W_cargo_front, X_cargo_back, W_cargo_back

def __passengers(X_cargo_front: float,
                 W_cargo_front: float,
                 num_PAX: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Calculate the center of gravity position and weight of OEW+cargo+PAX."""
    X_start = X_cargo_front[-1]
    W_start = W_cargo_front[-1]
    num_2PAX_rows = int((num_PAX - 3) // 2)
    num_3PAX_rows = 1
    X_last_seat = X_first_seat + (num_2PAX_rows * seat_pitch)

    """Front to back"""
    X_seat_front = np.zeros(int(num_2PAX_rows + num_3PAX_rows + 1))
    W_seat_front = np.zeros(int(num_2PAX_rows + num_3PAX_rows + 1))
    X_seat_front[0] = X_start
    W_seat_front[0] = W_start

    for i in range(1, num_2PAX_rows + 1):
        W_seat_front[i] = W_seat_front[i - 1] + (2 * M_PAX)
        X_seat_front[i] = (((X_first_seat + ((i-1) * seat_pitch)) * 2 * M_PAX) + 
                    (X_seat_front[i - 1] * W_seat_front[i - 1])) / W_seat_front[i]
    
    W_seat_front[-1] = W_seat_front[-2] + (3 * M_PAX)
    X_seat_front[-1] = (X_last_seat * 3 * M_PAX +
                        (X_seat_front[-2] * W_seat_front[-2])) / W_seat_front[-1]

    """Back to front"""
    X_seat_back = np.zeros(int(num_2PAX_rows + num_3PAX_rows + 1))
    W_seat_back = np.zeros(int(num_2PAX_rows + num_3PAX_rows + 1))
    X_seat_back[0] = X_start
    W_seat_back[0] = W_start

    W_seat_back[1] = W_seat_back[0] + (3 * M_PAX)
    X_seat_back[1] = (X_last_seat * 3 * M_PAX +
                    (X_seat_back[0] * W_seat_back[0])) / W_seat_back[1]

    for i in range(2, num_2PAX_rows + 2):
        W_seat_back[i] = W_seat_back[i - 1] + (2 * M_PAX)
        X_seat_back[i] = (((X_last_seat - ((i-1) * seat_pitch)) * 2 * M_PAX) + 
                        (X_seat_back[i - 1] * W_seat_back[i - 1])) / W_seat_back[i]

    return X_seat_front, W_seat_front, X_seat_back, W_seat_back

def __fuel(X_seat_front: float,
           W_seat_front: float,
           M_fuel: float,
           X_tank_TMS: float) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate the center of gravity position and weight of OEW+cargo+PAX+fuel."""
    # Ensure inputs are array‑like even when np.vectorize feeds scalars
    X_seat_front, W_seat_front = np.atleast_1d(X_seat_front), np.atleast_1d(W_seat_front)
    X_fuel = np.zeros(2)
    W_fuel = np.zeros(2)
    X_fuel[0] = X_seat_front[-1]
    W_fuel[0] = W_seat_front[-1]

    W_fuel[1] = W_fuel[0] + M_fuel
    X_fuel[1] = ((X_tank_TMS * M_fuel) + (X_fuel[0] * W_fuel[0])) / W_fuel[1]

    return X_fuel, W_fuel

def min_max_X_cg_positions(
    X_cargo_aft, M_cargo_aft,
    num_PAX, M_fuel, M_FC,
    M_TMS_fwd, M_TMS_aft,
    M_TMS_mid, M_EPS, M_tank,
    X_tank_TMS, X_tank_front
    ):
    """Find the minimum and maximum center of gravity positions."""
    OEW_H2D2, X_OEW_H2D2 = X_OEW(M_FC, M_TMS_fwd, M_TMS_aft, M_TMS_mid, M_EPS, M_tank, X_tank_TMS, X_tank_front) 
    X_cargo_front, W_cargo_front, X_cargo_back, _ = __cargo(X_cargo_aft, M_cargo_aft, X_OEW_H2D2, OEW_H2D2)
    X_seat_front, W_seat_front, X_seat_back, _ = __passengers(X_cargo_front, W_cargo_front, num_PAX)
    X_fuel, _ = __fuel(X_seat_front, W_seat_front, M_fuel, X_tank_TMS)

    arrays = [X_cargo_front, X_cargo_back, X_seat_front, X_seat_back, X_fuel]
    array_names = ["X_cargo", "X_seat_front", "X_seat_back", "X_fuel"]

    min_cg = np.min(np.hstack(arrays))
    max_cg = np.max(np.hstack(arrays))

    # min_index = np.argmin(np.hstack(arrays))
    # max_index = np.argmax(np.hstack(arrays))
    
    # delta = ((M_tank + M_TMS_aft) * X_tank_TMS
    #      - (M_FC + M_EPS + M_TMS_fwd) * X_wing) / OEW_H2D2
    
    # print(f"Delta: {delta:.2f} m")
    back_loading = M_cargo_aft + M_TMS_aft + M_tank
    original_loading = Beechcraft_1900D['M_cargo_aft']
    
    delta = back_loading - original_loading    

    cumulative_lengths = np.cumsum([len(arr) for arr in arrays])

    # def __find_source_array_and_index(flat_index):
    #     """Finds the original array and the index within that array."""
    #     for i, length in enumerate(cumulative_lengths):
    #         if flat_index < length:
    #             original_index = flat_index - (cumulative_lengths[i - 1] if i > 0 else 0)
    #             return array_names[i], original_index
    #     return None, None

    # # Get source array names and indices
    # min_source, min_array_index = __find_source_array_and_index(min_index)
    # max_source, max_array_index = __find_source_array_and_index(max_index)

    # Apply 2% margin
    min_cg_with_margin = min_cg*0.98
    max_cg_with_margin = max_cg*1.02

    MTOW_cg_position = X_fuel[-1]

    # print(f"Min value with 2% margin: {min_cg*0.98}, found in {min_source} at index {min_array_index}")
    # print(f"Max value with 2% margin: {max_cg*1.02}, found in {max_source} at index {max_array_index}")

    # plt.figure()
    # plt.plot(X_cargo_front, W_cargo_front, label='Cargo Front')
    # plt.plot(X_cargo_back, W_cargo_back, label='Cargo Back')
    # plt.plot(X_seat_front, W_seat_front, label='Seat Front')
    # plt.plot(X_seat_back, W_seat_back, label='Seat Back')
    # plt.plot(X_fuel, W_fuel, label='Fuel')
    # plt.legend()
    # plt.show()

    return min_cg, max_cg, min_cg_with_margin, max_cg_with_margin, MTOW_cg_position   

def main(design: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Vectorised integration routine that accepts a 4D tensor (N, M, P, Q) and
    returns a boolean mask of feasible designs together with passenger and
    cargo arrays.  The routine flattens the first three axes so that the
    existing scalar‑based helper functions can be reused without change.
    After per‑design evaluation each output vector is reshaped back to the
    original (N, M, P) cube.

    Parameters
    ----------
    design : np.ndarray
        Tensor with shape (N, M, P, Q) where Q contains the design variables
        described in the original docstring.

    Returns
    -------
    valid_design : np.ndarray, dtype=bool, shape (N, M, P)
        Boolean mask that is True when both volumetric and CG‑envelope
        constraints are satisfied.
    N_PAX : np.ndarray, dtype=int, shape (N, M, P)
        Number of passengers accommodated for every design.
    m_cargo : np.ndarray, dtype=float, shape (N, M, P)
        Mass of the aft cargo compartment for every design.
    """
    # --------------------------------
    # 0.   Prepare / reshape the input
    # --------------------------------
    var_idx   = np.array([0, 1, 2, 3, 4, 8, 9, 10, 11, 12, 13])
    vars_4d   = design[..., var_idx]                      # (N, M, P, 11)
    flat_vars = vars_4d.reshape(-1, vars_4d.shape[-1])    # (N*M*P, 11)

    n_designs      = flat_vars.shape[0]
    valid_vec      = np.empty(n_designs, dtype=bool)
    N_PAX_vec      = np.empty(n_designs, dtype=int)
    m_cargo_vec    = np.empty(n_designs, dtype=float)

    margin = 1.05  # volumetric margin for the FC installation

    # --------------------------------
    # 1.   Evaluate every design point
    # --------------------------------
    for i in range(n_designs):
        (m_EPS,
         m_FC,
         m_H2,
         m_storage,
         V_FC,
         L_storage,
         D_storage,
         _m_cargo_unused,          # placeholder – value replaced by ac_config
         m_TMS_front,
         m_TMS_aft,
         m_TMS_mid) = flat_vars[i]
        
        # ---- 1.1   Volume constraint
        volume_ok = V_FC * margin / 2 <= V_wing
        # print(f"Design {i}: Volume OK: {volume_ok}, V_FC: {V_FC/2:.2f} m^3, V_wing: {V_wing:.2f} m^3")

        # ---- 1.2   Aircraft configuration
        ac_conf        = acfg.cargo_main(L_tank=L_storage, d_tank=D_storage)
        N_PAX          = ac_conf['num_PAX']
        m_cargo_aft    = ac_conf['M_aft_cargo']
        X_aft_cargo    = ac_conf['X_aft_cargo']
        X_tank_front   = ac_conf['X_tank_front']
        X_tank_TMS     = ac_conf['X_tank_TMS']

        # ---- 1.3   CG envelope check
        _, _, min_cg_margin, max_cg_margin, MTOW_cg_position = min_max_X_cg_positions(
            X_cargo_aft = X_aft_cargo,
            M_cargo_aft = m_cargo_aft,
            num_PAX     = N_PAX,
            M_fuel      = m_H2,
            M_FC        = m_FC,
            M_TMS_fwd   = m_TMS_front,
            M_TMS_aft   = m_TMS_aft,
            M_TMS_mid   = m_TMS_mid,
            M_EPS       = m_EPS,
            M_tank      = m_storage,
            X_tank_TMS  = X_tank_TMS,
            X_tank_front= X_tank_front
        )

        cg_ok = (
            (X_most_fwd <= MTOW_cg_position <= X_most_aft)
        )
        N_PAX_ok = N_PAX >= 15 # Beechcraft 1900D has 19 seats, so we require at least 15
        # print(f"Design {i}: CG OK: {cg_ok}, "
        #       f"CG: {MTOW_cg_position:.2f} m, "
        #       f"X_most_fwd: {X_most_fwd:.2f} m, "
        #       f"X_most_aft: {X_most_aft:.2f} m, "
        #       f"min_cg_margin: {min_cg_margin:.2f} m, "
        #       f"max_cg_margin: {max_cg_margin:.2f} m, "
        #       f"N_PAX: {N_PAX}, N_PAX_ok: {N_PAX_ok}, "
        #       f"m_cargo_aft: {m_cargo_aft:.2f} kg")
        # print(f"Design {i}: CG OK: {cg_ok}, "
        #       f"CG: {MTOW_cg_position:.2f} m, X_most_fwd: {X_most_fwd:.2f} m, "
        #       f"X_most_aft: {X_most_aft:.2f} m, ")
        m_cargo_ok = m_cargo_aft >= N_PAX*M_cargo_per_PAX

        # ---- 1.4   Store results
        valid_vec[i]   = volume_ok and N_PAX_ok and m_cargo_ok and cg_ok
        N_PAX_vec[i]   = N_PAX
        m_cargo_vec[i] = m_cargo_aft

    # --------------------------------
    # 2.   Reshape back to (N, M, P)
    # --------------------------------
    out_shape    = vars_4d.shape[:-1]  # (N, M, P)
    valid_design = valid_vec.reshape(out_shape)
    N_PAX_arr    = N_PAX_vec.reshape(out_shape)
    m_cargo_arr  = m_cargo_vec.reshape(out_shape)

    return valid_design, N_PAX_arr, m_cargo_arr