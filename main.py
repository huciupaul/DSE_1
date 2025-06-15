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
    power_splits = np.linspace(0.6, 1.0, 5)  # 10 splits from 0 to 1
    fc_toga_percentages = np.linspace(minimum, maximum, no_of_splits)  # Example percentages of TOGA power for the fuel cell
    fc_cruise_percentages = np.linspace(minimum, maximum, no_of_splits)  # Example percentages of CRUISE power for the fuel cell

    # Initialize the result tensor with zeros
    result_tensor = np.zeros((
    len(power_splits),
    len(fc_toga_percentages),
    len(fc_cruise_percentages),
    21  # Length of the tensor
    ))
    
    loading_tensor = np.zeros((
        len(power_splits),
        len(fc_toga_percentages),
        len(fc_cruise_percentages),
        4, 4
    ))

    for i_split, split in enumerate(power_splits):
        
        # EPS
        m_eps, Qdot_eps, V_elmo, co2_eps, P_elmo = eps_main(split)

        for i_toga, fc_toga_percentage in enumerate(fc_toga_percentages):

            for i_cruise, fc_cruise_percentage in enumerate(fc_cruise_percentages):

                # Initialize variables for the first iteration of convergence
                MTOW = Beechcraft_1900D['OEW'] # Original OEW
                m_eps_prev, m_fc_tms_prev, m_h2_prev, m_sto_prev, m_cargo_prev = 0.0, 326.63, 315.39, 141.38, 714.05 # from midterm
                MTOW += (m_eps + m_fc_tms_prev + m_h2_prev + m_sto_prev)  # Initial MTOW
                MTOW_prev = 0.001

                aux_power, D_rad = 0.0, 0.0

                # Convergence of MTOW, stop if the change is less than 1% of the previous MTOW or reached max number of iterations
                for i in range(max_iter):
                    print(f"Split: {split:.2f}, TOGA: {fc_toga_percentage:.2f}, Cruise: {fc_cruise_percentage:.2f}, ITER: {i}")
                    if np.abs((MTOW-MTOW_prev)/MTOW_prev) < 0.01 and i > 0 or i == max_iter - 1:
                        print(f"Converged after {i} iterations with MTOW: {MTOW:.2f} kg")
                        print(f"Split: {split:.2f}, TOGA: {fc_toga_percentage:.2f}, Cruise: {fc_cruise_percentage:.2f}")
                        break
                    
                    # FPP
                    TMS_inputs, m_h2, FC_outputs, _, loading_vector, emissions = fpp_main(split, fc_toga_percentage, fc_cruise_percentage, MTOW, D_rad, aux_power, 10)
                    # Also get m_nox, nox_max_ppm, co2_fc from fpp_main
                    m_nox, mdot_nox_max_takeoff, mdot_nox_max_cruise = 0.0, 0.0, 0.0
                    m_fc = FC_outputs['m_fc']
                    V_fc = FC_outputs['V_fc']
                    co2_fc = FC_outputs['co2_fc']

                    # Storage
                    m_sto, V_sto, _, _, _, length_sto, diameter_sto, co2_sto = main_storage(m_h2)
                    
                    # Cargo
                    result = cargo_main(length_sto, diameter_sto)
                    M_aft_cargo:  float = result["M_aft_cargo"]
                    # Uncomment the following if needed
                    # X_tank_front: float = result["X_tank_front"]
                    # X_tank_back:  float = result["X_tank_back"]
                    # X_tank_TMS:   float = result["X_tank_TMS"]
                    # V_tank_TMS:   float = result["V_tank_TMS"]
                    # X_aft_cargo:  float = result["X_aft_cargo"]
                    # V_aft_cargo:  float = result["V_aft_cargo"]
                    # num_PAX:      int   = result["num_PAX"]

                    # TMS

                    # print("START",TMS_inputs['Q_dot_fc'][0], #ok
                    #     Qdot_eps, #ok
                    #     P_A[0], #ok
                    #     TMS_inputs['p_cc'][0], #ok
                    #     TMS_inputs['h2_mf_fc'][0], #ok
                    #     TMS_inputs['h2_mf_cc'][0], #ok
                    #     T_FC[0],
                    #     TMS_inputs['t_cc'][0], #ok
                    #     TMS_inputs['air_mf_fc'][0], #ok
                    #     TMS_inputs['t_amb'][0], #ok
                    #     TMS_inputs['rho_amb'][0], #ok
                    #     TMS_inputs['V_amb'][0], #ok
                    #     mission_profile['P'][0], #ok
                    #     TMS_inputs['h2_mf_fc_recirculated'][0], 
                    #     TMS_inputs['air_mf_fc'][0],
                    #     7e5,  # p_sto
                    #     TMS_inputs['h2o_mf_fc'][0], "END")
                    # m_tms_front, m_tms_aft, m_tms_mid, D_rad, aux_power = 0.0, 0.0, 0.0, 0.0, 0.0
                    # break
                    #print("Inputs!!!!!", TMS_inputs if 'TMS_inputs' in locals() else "No TMS inputs yet")
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
                    if tms_outputs is None:
                        break

                    D_rad = tms_outputs[0]
                    aux_power = tms_outputs[1]
                    m_tms_front = tms_outputs[2]
                    m_tms_aft = tms_outputs[4]
                    m_tms_mid = tms_outputs[3]
                    print(f"MTOW:{MTOW}, AUX POWER {aux_power}, DRAG PENALTY {D_rad}")
                    

                    # --------- ADD INTEGRATION HERE ---------
                    
                    #MTOW update
                    MTOW_prev = MTOW
                    MTOW += ((m_eps - m_eps_prev) + (m_h2 - m_h2_prev) + (m_sto - m_sto_prev) + \
                        (m_tms_front + m_tms_aft + m_tms_mid + m_fc - m_fc_tms_prev) + (M_aft_cargo - m_cargo_prev))
                    # Update the values based on the previous iteration
                    m_eps_prev = m_eps
                    m_fc_tms_prev = m_tms_aft + m_tms_front + m_tms_mid + m_fc
                    m_h2_prev = m_h2
                    m_sto_prev = m_sto
                    m_cargo_prev = M_aft_cargo
                if tms_outputs is None:
                    tensor = np.full(21, None)
                    result_tensor[i_split, i_toga, i_cruise, :] = tensor
                    loading_tensor[i_split, i_toga, i_cruise, :, :] = loading_vector
                    
                # After convergence, store the results in the tensor, then in the 4D tensor
                tensor = np.array([m_eps, m_fc, m_h2, m_sto,V_fc, V_sto, V_elmo, MTOW, length_sto, diameter_sto, M_aft_cargo, m_tms_front, m_tms_aft, m_tms_mid, \
                                   m_nox, mdot_nox_max_takeoff, mdot_nox_max_cruise, P_elmo, co2_fc, co2_sto, co2_eps])
                result_tensor[i_split, i_toga, i_cruise, :] = tensor
                loading_tensor[i_split, i_toga, i_cruise, :, :] = loading_vector

    # Save the result tensor to a pickle file
    with open("data/logs/result_tensor.pkl", "wb") as f:
        pickle.dump(result_tensor, f)
    
    with open("data/logs/loading_tensor.pkl", "wb") as f:
        pickle.dump(loading_tensor, f)
    
    print("Result tensor saved to data/logs/result_tensor.pkl")
    print("Loading tensor saved to data/logs/loading_tensor.pkl")

    return None       


if __name__=="__main__":
    main(
        minimum=0.1,
        maximum=1.0,
        no_of_splits=10,
        max_iter=100
    )