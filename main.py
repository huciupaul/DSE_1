# file: main.py
# desc: This script is the main entry point for the SHTARWaRS design tool.

# External imports
import os
import csv
import argparse
import warnings
import matplotlib.pyplot as plt
import numpy as np
from global_constants import * # Import global constants
import pickle

# Local imports
from eps.eps_sizing import eps_main  # Import EPS sizing function
from fpp.flight import fpp_main  # Import FPP sizing function
from storage.tank import main_storage  # Import storage sizing function
from performance.Integration.aft_configuration import cargo_main  # Import cargo sizing function
from tms import tms_main  # Import TMS sizing function


def main(minimum, maximum, no_of_splits, max_iter):
    #np.seterr(all='ignore')
    # Check if the output directory exists, if not create it
    output_dir = "data/logs"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Define the power splits and fuel cell percentages
    power_splits = np.linspace(minimum[0], maximum[0], no_of_splits[0])  # 10 splits from 0 to 1
    fc_toga_percentages = np.linspace(minimum[1], maximum[1], no_of_splits[1])  # Example percentages of TOGA power for the fuel cell
    fc_cruise_percentages = np.linspace(minimum[2], maximum[2], no_of_splits[2])  # Example percentages of CRUISE power for the fuel cell

    # Initialize the result tensor with zeros
    result_tensor = np.zeros((
    len(power_splits),
    len(fc_toga_percentages),
    len(fc_cruise_percentages),
    23  # Length of the tensor
    ))
    
    loading_tensor = np.zeros((
        len(power_splits),
        len(fc_toga_percentages),
        len(fc_cruise_percentages),
        4, 4
    ))
    
    convergence_tensor = np.full((
        len(power_splits),
        len(fc_toga_percentages),
        len(fc_cruise_percentages),
        max_iter
        ), np.nan)

    for i_split, split in enumerate(power_splits):
        for i_toga, fc_toga_percentage in enumerate(fc_toga_percentages):

            for i_cruise, fc_cruise_percentage in enumerate(fc_cruise_percentages):
                # ----------------------------------------------------------
                # INITIALIZE MTOW FIXED-POINT ITERATION
                # ----------------------------------------------------------
                OEW = Beechcraft_1900D['OEW']

                # First-guess component masses (from midterm or prior run)
                m_eps_prev     = 0.0
                m_fc_tms_prev  = 326.63  # sum of prior FC+TMS masses
                m_h2_prev      = 315.39
                m_sto_prev     = 141.38
                m_cargo_prev   = 714.05
                m_pax          = 1596 # No passengers in this configuration  
                
                m_eps, Qdot_eps, V_elmo, co2_eps, P_elmo = eps_main(split)
                
                MTOW_prev = 0.0
                MTOW = (
                    OEW
                    + m_eps_prev
                    + m_fc_tms_prev
                    + m_h2_prev
                    + m_sto_prev
                    + m_cargo_prev
                    + m_pax
                )

                alpha      = 0.5     # Under-relaxation factor
                delta_prev = 0.0
                tol        = 0.001    # 0.1% convergence tolerance
                
                CD_rad = 0.0  # Initialize CD_rad
                aux_power = 0.0  # Initialize aux_power
                
                m_tms_front = 0.0
                m_tms_aft = 0.0
                m_tms_mid = 0.0
                
                # Convergence of MTOW, stop if the change is less than 1% of the previous MTOW or reached max number of iterations
                for i in range(max_iter):
                    # print(f"Split: {split:.2f}, TOGA: {fc_toga_percentage:.2f}, Cruise: {fc_cruise_percentage:.2f}, ITER: {i}")
                    # --- FPP call ---
                    TMS_inputs, m_h2, FC_outputs, _, loading_vector, emissions, P_fc_max = fpp_main(
                        split,
                        fc_toga_percentage,
                        fc_cruise_percentage,
                        MTOW_prev,
                        CD_rad,
                        aux_power,
                        dt=10
                    )
                    
                    # Update EPS variables
                    m_eps, Qdot_eps, V_elmo, co2_eps, P_elmo = eps_main(P_fc_max/TOGA)
                    
                    m_fc = FC_outputs['m_fc']
                    cost_fc = FC_outputs['fc_cost']

                    # Also get m_nox, nox_max_ppm, co2_fc from fpp_main
                    m_nox = emissions['m_NOx']
                    mdot_nox_max_takeoff = emissions['max_mdot_NOx_TO']
                    mdot_nox_max_cruise = emissions['max_mdot_NOx_cruise']
                    m_h2_nom = emissions['m_H2_nom']

                    m_fc = FC_outputs['m_fc']
                    V_fc = FC_outputs['V_fc']
                    co2_fc = FC_outputs['co2_fc']

                    # --- STORAGE call ---
                    m_sto, V_sto, _, _, _, length_sto, diameter_sto, co2_sto = main_storage(m_h2)
                                        
                    # --- CARGO call ---
                    result = cargo_main(length_sto, diameter_sto)
                    M_aft_cargo = result["M_aft_cargo"]
                    m_PAX       = result["num_PAX"] * M_PAX
                    
                    
                    
                    _, tms_outputs = tms_main(
                        TMS_inputs['Q_dot_fc'], #ok
                        np.full(4, Qdot_eps), #ok
                        np.full(4, P_A[0]), #ok
                        TMS_inputs['p_cc'], #ok
                        TMS_inputs['h2_mf_fc'], #ok
                        TMS_inputs['h2_mf_cc'], #ok
                        np.full(4, T_FC[0]),
                        TMS_inputs['t_cc'], #ok
                        TMS_inputs['air_mf_fc'], #ok
                        TMS_inputs['t_amb'], #ok
                        TMS_inputs['rho_amb'], #ok
                        TMS_inputs['V_amb'], #ok
                        TMS_inputs['P_amb'], #ok
                        TMS_inputs['h2_mf_fc_recirculated'], 
                        TMS_inputs['air_mf_fc'],
                        np.full(4, MAWP_global),  # p_sto
                        TMS_inputs['h2o_mf_fc']
                    )
                    #print("Outputs!!!!!!", tms_outputs if 'tms_outputs' in locals() else "No TMS outputs yet")
                    # Check if tms_outputs is None or has the expected length
                    if tms_outputs is None or (isinstance(tms_outputs, (list, np.ndarray)) and len(tms_outputs) > 0 and np.isnan(tms_outputs[0])):
                        break
                    
                    D_rad = tms_outputs[0]
                    aux_power = tms_outputs[1]
                    m_tms_front = tms_outputs[2]
                    m_tms_aft = tms_outputs[4]
                    m_tms_mid = tms_outputs[3]
                    
                    print(f"MTOW:{MTOW:.2f}, AUX POWER {aux_power:.2f}, DRAG PENALTY {D_rad:.6f}")
                    
                    # ----------------------------------------------------------
                    # FIXED-POINT UPDATE WITH UNDER-RELAXATION
                    # ----------------------------------------------------------
                    
                    MTOW_candidate = (
                        OEW
                        + m_eps
                        + (m_tms_front + m_tms_mid + m_tms_aft + m_fc)
                        + m_h2
                        + m_sto
                        + M_aft_cargo
                        + m_PAX
                    )
                    delta = MTOW_candidate - MTOW_prev

                    # if sign flip, damp further
                    if delta * delta_prev < 0 and i > 0:
                        alpha *= 0.5

                    MTOW = MTOW_prev + alpha * delta if i > 0 else MTOW_candidate
                    
                    convergence_tensor[i_split, i_toga, i_cruise, i] = MTOW

                    # check convergence
                    if abs(delta) / MTOW_prev < tol:
                        MTOW_prev = MTOW
                        break

                    # prepare for next iter
                    delta_prev = delta
                    MTOW_prev  = MTOW

                    # update component previous-mass trackers
                    m_eps_prev    = m_eps
                    m_fc_tms_prev = m_tms_front + m_tms_mid + m_tms_aft + m_fc
                    m_h2_prev     = m_h2
                    m_sto_prev    = m_sto
                    m_cargo_prev  = M_aft_cargo
                
                # CONVERGED
                MTOW = MTOW_prev
                
                # Update the loading vector                     
                if tms_outputs is None or (isinstance(tms_outputs, (list, np.ndarray)) and len(tms_outputs) > 0 and np.isnan(tms_outputs[0])):
                    tensor = np.full(23, np.nan)
                    result_tensor[i_split, i_toga, i_cruise, :] = tensor
                    loading_tensor[i_split, i_toga, i_cruise, :, :] = loading_vector
                
                # After convergence, store the results in the tensor, then in the 4D tensor
                tensor = np.array([
                    m_eps,                  # 0
                    m_fc,                   # 1
                    m_h2,                   # 2
                    m_sto,                  # 3
                    V_fc,                   # 4
                    V_sto,                  # 5
                    V_elmo,                 # 6
                    MTOW,                   # 7
                    length_sto,             # 8
                    diameter_sto,           # 9
                    M_aft_cargo,            # 10
                    m_tms_front,            # 11
                    m_tms_aft,              # 12
                    m_tms_mid,              # 13
                    m_nox,                  # 14
                    mdot_nox_max_takeoff,   # 15
                    mdot_nox_max_cruise,    # 16
                    cost_fc,                # 17
                    m_h2_nom,               # 18
                    P_elmo,                 # 19
                    co2_fc,                 # 20
                    co2_sto,                # 21
                    co2_eps ,               # 22
                    ])
                
                result_tensor[i_split, i_toga, i_cruise, :]     = tensor
                loading_tensor[i_split, i_toga, i_cruise, :, :] = loading_vector

    # Save the result tensor to a pickle file
    with open("data/logs/result_tensor.pkl", "wb") as f:
        pickle.dump(result_tensor, f)
    
    with open("data/logs/loading_tensor.pkl", "wb") as f:
        pickle.dump(loading_tensor, f)
        
    with open("data/logs/convergence_tensor.pkl", "wb") as f:
        pickle.dump(convergence_tensor, f)
    
    print("Result tensor saved to data/logs/result_tensor.pkl")
    print("Loading tensor saved to data/logs/loading_tensor.pkl")
    print("Convergence tensor saved to data/logs/convergence_tensor.pkl")

    return None       


if __name__=="__main__":
    # min_ = np.array([0.25, 0.2, 0.25])  # Minimum values for the splits
    # max_ = np.array([0.35, 0.35, 0.4])
    min_ = np.array([0.33, 0.29, 0.30])  # Minimum values for the splits
    max_ = np.array([0.33, 0.29, 0.30])
    
    main(
        minimum=min_,
        maximum=max_,
        no_of_splits=np.array((max_-min_)/0.01 + 1, dtype=int),  # Number of splits for each parameter
        max_iter=20
    )