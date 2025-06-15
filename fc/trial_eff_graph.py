import numpy as np
import matplotlib.pyplot as plt

# Constants
F = 96485  # Faraday constant, C/mol
Ru = 8.314  # Universal gas constant, J/mol-K
Vo = 1.18  # Reversible voltage, V
n = 2  # Electrons transferred
mu_fuel = 0.95
eta_pc = 0.95
a = 0.0499  # BoP empirical constant
b = 0.05
delta_Hf = 285250  # Enthalpy of formation, J/mol
A = 348.4  # cm^2, active area
T = 273.15 + 160  # K, temperature (adjustable)
alpha_c = 0.6
io_c = 0.005  # A/cm^2
La = 0.05  # cm
Le = 0.51  # cm
Lc = 0.1  # cm
sigma_a = 2.49  # S/m
sigma_e = 8.3  # S/m
sigma_c = 2.9  # S/m
R_contact = 0.03  # Ohm.cm^2
Ba = Bc = 0.045  # V
ilim_a = 15  # A/cm^2
ilim_c = 2.5  # A/cm^2
delta_S = 163.2

# Current density range (A/cm^2)
i_values = np.linspace(0.01, 2.5, 100)

def delta_V_act_cross(i):
    iloss = (i * A) / (n * F) * (2 * F / A)
    return (Ru * T / F) * (1 / alpha_c) * np.log((iloss + i) / io_c)

def delta_V_ohmic(i):
    return i * A * (La / (sigma_a * A) + Le / (sigma_e * A) + Lc / (sigma_c * A) + R_contact / A)

def delta_V_conc(i):
    return Ba * np.log(1 - i / ilim_a) + Bc * np.log(1 - i / ilim_c)

def cell_voltage(i):
    return Vo - delta_V_act_cross(i) - delta_V_ohmic(i) - delta_V_conc(i)

def E_nernst(i):
    E0 = delta_Hf / (n * F)
    return E0 + (Ru * T / n / F) * np.log(i / (io_c * A))

def eta_BoP(i):
    E = E_nernst(i)
    return 1 - a - (b / E * i)

def eta_total(i):
    E = E_nernst(i)
    return (n * F * E / delta_Hf) * mu_fuel * eta_pc * eta_BoP(i)

# Calculate efficiency over current density
eta_vals = eta_total(i_values)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(i_values, eta_vals, label='System Efficiency (η_total)', color='blue')
plt.xlabel('Current Density (A/cm²)')
plt.ylabel('Efficiency (η_total)')
plt.title('PEM Fuel Cell System Efficiency vs. Current Density')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
