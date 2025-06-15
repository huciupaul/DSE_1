import numpy as np
from CoolProp.CoolProp import PropsSI as cp
import matplotlib.pyplot as plt


def efficiency(j_0):

    """
    Calculate the efficiency of a fuel cell based on the Kulikovsky equation.
    Source:
    https://doi.org/10.1016/j.enconman.2024.118266
    Input: 
    
    Returns:
        float: Efficiency of the fuel cell.
    """

    # Constants
    b = 0.03 # [V] Tafel slope
    c_h = 7.36e-6 # [mol/cm**3] Oxygen concentration in the cathode channel (p = 1 bar)
    c_ref = 8.58e-6 # [mol/cm**3] reference Oxygen concentration in the channel
    sig_t = 0.03 # [S/cm] Proton conductivity in the catalyst
    F = 96485.3329 # [C/mol] Faraday constant
    l_t = 0.0007 # [cm] Thickness of the CCL
    D_b = 0.0259 # [cm^2/s] Effective diffusion coefficient of the GDL
    V_oc = 1.145 # [V] Open circuit voltage
    LHV_H2 = 120e6 # [J/kg] Lower heating value of hydrogen
    stoic_ratio_A = 1.0 # Stoichiometric ratio of the anode,  lambda_H2
    stoic_ratio_C = 2.0 # Stoichiometric ratio of the cathode, labda_air

    # Variable parameters
    j_star = 2e-3 # [A/cm^3] Exchange current density (range 1e-3 to 4e-2 A/cm^3)
    l_b = 0.0312 # [cm] Thickness of the GDL (range 0.015 to 0.04 cm)
    D = 1e-4 # [cm^2/s] Diffusion coefficient of the CCL (range 1e-4 to 3e-4 cm^2/s)
    R_om = 0.0801 # [ohm cm^2] Area specific resistance (range 0.045 to 0.2 ohm cm^2)

    # i_star = F * S_0 * l * k_star * c_O2

    i_star = sig_t * b / l_t
    j_sig = np.sqrt(2 * i_star * sig_t * b)
    j_lim = 4 * F * D_b * c_h / l_b

    print(f"j_star: {i_star}, j_sig: {j_sig}, j_lim: {j_lim}")

    beta = (np.sqrt(2 * j_0 / j_star)/ (1 + np.sqrt(1.12 * j_0 / j_star) * np.exp(np.sqrt(2 * j_0 / j_star))) 
            + np.pi * (j_0 / j_star) / (2 + (j_0 / j_star)))


    eta_0 = np.arcsinh((j_0 / j_sig)**2 / (2 * (c_h / c_ref) * (1 - np.exp(j_0 / (2 * j_star))))) + (sig_t * b**2) / (4 * F * D * c_h) * np.log((j_0 / j_star - 1) + (j_0**2 / (j_star**2 * beta**2)) * (1 - j_0 / (j_lim * (c_h / c_ref)))**-1) - b * np.log(1 - j_0 / (j_lim * (c_h / c_ref)))
    print(eta_0)

    V_cell = V_oc - R_om * j_0 - eta_0

    P = j_0 * V_cell  # [W/cm**2] Specific power output of the fuel cell 

    m_H2 = stoic_ratio_A * j_0 / (2 * F) * cp('M', 'Hydrogen')   # Mass flow rate of hydrogen in kg/s

    # Efficiency calculation
    eta_cell = P / (m_H2 * LHV_H2)  # Efficiency of the fuel cell
    
    return eta_cell



if __name__ == "__main__":
    # Example usage
    j_0 = np.linspace(0.01, 2.0, 1000)  # Example exchange current density in A/cm^2
    eta_cell = efficiency(j_0)
    #print(f"Cell efficiancy: {eta_cell}")
    plt.plot(j_0, eta_cell)
    plt.show()