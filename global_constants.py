# file: global_constants.py
# desc: Global constants used throughout the project.

# Global imports
import numpy as np
from typing import Tuple
import pickle

# General constants
R_AIR = 287.05                                          # [J/(kg·K)]
G_0 = 9.80665                                           # [m/s²]
LAPSE = -0.0065                                         # [K/m]
k_air = 1.4                                             # [-] Specific heat ratio of air

# Sea level atmosphere constants
T_sl = 288.15                                           # [K] Sea level temperature
P_sl = 101_325.0                                        # [Pa] Sea level pressure
rho_sl = 1.225                                          # [kg/m³] Sea level density

# ISA model
def isa_atmosphere(h: np.ndarray, T_sl: float = T_sl, P_sl: float = P_sl) -> Tuple[float, float, float]:
    """Thin-layer ISA (no tropopause).
    Returns T [K], P [Pa], rho [kg/m^3], and speed of sound."""
    T = T_sl + LAPSE * h
    P = P_sl * (T / T_sl) ** (-G_0 / (LAPSE * R_AIR))
    rho = P / (R_AIR * T)
    a = np.sqrt(k_air * R_AIR * T)  # speed of sound
    return T, P, rho, a

# Original Aircraft geometry constants
C_d_0_orig = 0.023
AR, e = 10.808, 0.735809
k = 1 / (np.pi * AR * e)
S = 28.79                                               # [m²]
# MTOW_orig = 7_765   * G_0                               # [N] Original Maximum Take-Off Weight
MTOW_orig = 8037.9 * G_0                               # [N] Original Maximum Take-Off Weight (from FZO report)

# Modified Aircraft MTOW (requirement)PLACEHOLDER
MTOW_mod  = 9459.6* G_0                               # [N] Modified Maximum Take-Off Weight  

# Original aircraft performance constants
V_TO, V_L  = 54.0167, 54.0167
V_stall_L  = 46.3
V_cruise   = 144.044
rho_cruise = 0.550798
ROC, V_climb = 10.16, 73.43


# 100 year GWP of gases (compared to CO2)
GWP_CO2 = 1.0                                            # [-] Global Warming Potential of CO2
GWP_NOx = 59                                             # [-] Global Warming Potential of NOx https://doi.org/10.3390/app122010429
GWP_H2O = 0.0005                                         # [-] Global Warming Potential of H2O        https://iopscience.iop.org/article/10.1088/1748-9326/aae018/pdf
GWP_beech = 335000 * 30/19                               # [-] Global Warming Potential of Beechcraft 1900D https://odr.chalmers.se/items/a33bc0b8-bf61-47cc-87b8-3ab92fe388ed
GWP_H2 = 2                                              # [kgCO2e/kgH2] Global Warming Potential of Hydrogen https://doi.org/10.1016/j.apenergy.2023.122586    NOT THIS ONE https://doi.org/10.1038/s41560-024-01563-1

# NOx KPIs
NOx_pPax_TO     = 0.000012783399 # [kg/s/PAX]
NOx_pPax_cruise = 0.000009341715 # [kg/s/PAX]

#######################################################################################
#################### Code specific constants from each part ###########################

###########
# From FC #
###########

# Fuel Cell Stack Characteristics
mass_specific_power = 5000                              # [W/kg] Specific power of the fuel cell stack (FZO Roadmap-report)
volume_specific_power = 3100000                         # [W/m^3] Volume specific power of the fuel cell stack (FZO Roadmap-report)
stoic_ratio_A = 1.05                                     # [-] Stoichiometric ratio of the anode
stoic_ratio_C = 1.6                                     # [-] Stoichiometric ratio of the cathode
pickle_path = 'fc/spline_eta_of_P.pkl'                     # Path to the efficiency function file
# Load efficiency function from file
with open(pickle_path, 'rb') as f:
    efficiency_per_throttle = pickle.load(f)

P_A=1.6 * 101325 + 0.06 * 1e5,                          # [Pa] Anode Pressure
P_C=1.6 * 101325,                                       # [Pa] Cathode Pressure
T_FC = 273.15 + 160,                                    # [K] Fuel Cell Temperature


# Aircraft lifetime:
years_of_life = 20                                      # [years] Aircraft lifetime  
flights_per_year = 920                                    # [flights/year] Average number of flights per year
flight_lifetime = flights_per_year * years_of_life                   # number of flights over the aircraft's lifetime
avg_flight_duration = 1.5                                   # [h] Average flight duration
time_lifetime = flight_lifetime * avg_flight_duration                   # number of flight hours over the aircraft's lifetime
num_aircraft = 500                                      # number of aircraft in the fleet

# Costs and emissions
FC_prod_gwp = 30.5                                      # [kg CO2/kW] GWP of the fuel cell production with bop (see excel) DOI: 10.4271/2024-24-0020
FC_cost_no_bop = 555/2                                  # [EUR/kW] Cost of the fuel cell stack without balance of plant (flyzero)
FC_cost = 555                                           # [EUR/kW] Cost of the fuel cell with bop (flyzero)
FC_maint_cost = 220/2400 * time_lifetime                # [EUR/kWh] Maintenance cost of the fuel cell per kW
FC_disposal_cost = 9.35                                 # [EUR/kgFC] Disposal cost of the fuel cell stack per kg of fuel cell stack
Sto_disposal_cost = 0.6                                 # [EUR/kgSto] Disposal cost of the storage system per kg of tank
Sto_maint_cost = 43                                     # [EUR/kgH2/yr] Maintenance cost of the storage system
AC_disposal_cost = 6.693 * 4932                         # [EUR/kgAC] Disposal cost of the aircraft per kg of aircraft
Insurance_cost = 19000 * years_of_life                            # [EUR] Insurance cost of the aircraft over its lifetime
Crew_cost = 2 * 89 * time_lifetime                      # [EUR] Crew cost of the aircraft over its lifetime
Landing_tax = 54.53 * flight_lifetime                   # [EUR] Landing tax of the aircraft over its lifetime
Beech_maint_cost = 1100 * time_lifetime * 555/640       # [EUR] Maintenance cost of the Beechcraft 1900D https://www.guardianjet.com/jet-aircraft-online-tools/aircraft-brochure.cfm?m=Beech-1900D-198
depreciation_rate = 0.05                                # [-] Depreciation rate of the aircraft

Sto_cost = 212                                          # [EUR/kgH2] Cost of the storage system https://www.horizon-europe.gouv.fr/advanced-materials-hydrogen-storage-tanks-34822
EPS_cost = 267                                          # [EUR/kWELMO] Cost of the electrical power system 
AC_dev_cost = 255900000 / num_aircraft                  # [EUR] Development cost of the aircraft per aircraft https://www.mdpi.com/2226-4310/9/7/349
AC_purchase_cost = 1_300_000                            # [EUR] Purchase cost of the aircraft (1.5 times the development cost)
AC_maint_cost = 1_311_000                               # [EUR] Maintenance cost of the aircraft over its lifetime per aircraft

### REVIEW THIS
H2_cost = 4                                             # [EUR/kg] Cost of liquid hydrogen



### FOR CODE CHECKING PURPOSES ONLY ###
#50% power split
m_H2_FC = 0.01757833343024256                           # [kg/s] Mass flow rate of hydrogen in the fuel cell
P_FC_stack = 1476580.0081403746                         # [W] Power of the Fuel Cell Stack
Q_dot_FC = 532372.38388734                              # [W] Heat generated by the Fuel Cell Stack
LHV_H2 = 120e6                                         # [J/kg] Lower Heating Value of Hydrogen


####################
# From performance #
####################
A_inlet = 2*0.038410516                                 # [m^2] Inlet area
MAXC = 2*906e3                                          # [W] Max Continuous Power
TOGA =  1908*1000                                        # [W] Take-Off/Go-Around power

# Engine performance models
with open('data/interpolants/engine2035_interpolators.pkl', 'rb') as f:
    engine_interpolators = pickle.load(f)

mdot_fuel = engine_interpolators['mf_fuel_from_power']  # [kg/s] Mass flow rate of fuel from power
mdot_air = engine_interpolators['mf_air_from_power']    # [kg/s] Mass flow rate of air from power

# Engine NOx emission model
with open('data/interpolants/T_peak_interpolant.pkl', 'rb') as f:
    T_peak_interpolator = pickle.load(f)
    
with open('data/interpolants/NOx_interpolator.pkl', 'rb') as f:
    NOx_interpolator = pickle.load(f)

mdot_NOx = NOx_interpolator                             # [kg/s] Mass flow rate of NOx from shaft power

eff_prop   = 0.80
mu_TO      = 0.04
S_g, S_land = 1163, 851
LW_max     = 7_604 * G_0
TOGA = 1908*1000                                        # [W] Take-Off/Go-Around power
base_AP = 2 * 8.4 * 1e3                                 # [W] Base Auxiliary Power               
E_SCALE = 0.0783632513412531                            # [-] Scale factor for the error

# Dynamic pressures
q_climb = 0.5 * rho_sl     * V_climb**2
q_c     = 0.5 * rho_cruise * V_cruise**2


################
# From storage #
################

dormancy = 24                                           #[hours]
stratification_factor = 2
k_str = 1.9                                             #[W/mK]
MAWP_global = 700000                                    #[Pa]
p_vent_global = 700000                                  #[Pa]
p_sl = 101325                                           #[Pa]

r_in = 0.74                                             #[m]

Q_leak_min = 10                                         #W (determined from Nicolas's thesis)
Q_leak_max = 1000                                       #W (determined from Nicolas's thesis)

T_hot = 40                                              #[Celsius] conservative for 

gf = 'S-Glass fiber'
fos = 2/3                                               #factor of safety for the tensile strength of the glass fiber
gf_density, gf_tensile_strength, gf_thermal_cond, gf_thermal_emis, gf_co2, gf_ee, gf_fvf = 1905, 1730*1e6*fos, 0.745, 0.95, 7.22, 116.5, 0.675

# Thesis values
Q_original_str = 0.4                                    #W
kevlar_thermal_cond = 1.9                               #W/mK
mass_original_str = 2.1                                 #kg
mass_originalg_lh2 = 6.2                                #kg
gravimetric_index = 0.35                                #from NASA report
kevlar_co2 = 13.1                                       #kg/kg (Kevlar 149)
kevlar_emb_energy = 257                                 #MJ/kg (Embodied Energy for Kevlar 149)

t_min = 0.001                                           #m (minimum thickness of the tank wall)

mli_density = 7900                                      #kg/m^3 https://www.sciencedirect.com/science/article/pii/S135943112200391X
mli_emis = 0.21                                         #https://www.thermalengineer.com/library/effective_emittance.htm
vacuum_thermal_cond = 0.015*1e-1                        #3 # W/mK https://www.researchgate.net/publication/321219004_Cylindrical_Cryogenic_Calorimeter_Testing_of_Six_Types_of_Multilayer_Insulation_Systems
mli_thermal_cond = 17.4                                 # W/mK  https://www.sciencedirect.com/science/article/pii/S135943112200391X
mli_ss_co2 = 3                                          #kg/kg (for SS)
mli_ss_ee = 42.74                                       #MJ/kg (Embodied Energy for SS)
mli_layers = 40
mli_thickness = 0.03 *1e-3 * mli_layers                 #https://www.sciencedirect.com/science/article/pii/S135943112200391X

####################
# From integration #
####################
seat_pitch = 0.75
rho_cargo = 161                                         # https://www.icao.int/MID/Documents/2024/Aviation%20Statistics%20Workshop/PPT-Session%203.pdf

M_PAX = 84                                              # https://www.easa.europa.eu/en/light/topics/passenger-weight-and-why-it-matters-safe-and-efficient-air-operations
M_cargo_fwd = 1.01371425*rho_cargo
M_cargo_per_PAX = 17.5                                  # [kg] Cargo mass per passenger, based on original aircraft design

X_most_aft  = 7.51
X_most_fwd  = 6.79

X_cargo_fwd = 3.87
X_first_seat = 4.70
X_front_of_first_seat = 4.38
X_aft_cone_beg = 12.61
X_aft_cone_end = 13.78
X_EPS = 6.3
X_wing = 7.29
X_wing_end = 8.18

V_cargo_fwd = 1.01371425
V_wing      = 0.7508 # [m^2] space in one wing
    
l_aft_cyl  = 1.50
w_aft_cyl  = 1.60
h_aft_cyl_ave  = 1.92

d_aft_cone_beg   = 1.60
d_aft_cone_end   = 1.18
l_aft_cone  = 1.18

Beechcraft_1900D = dict(
    MTOW        = 7766,
    # MTOW        = 8037.9,
    M_fuel      = 300,
    OEW         = 4932,
    X_OEW       = 6.76,
    M_cargo_aft = 4.8106*rho_cargo,
    num_PAX     = 19,
    X_cargo_aft = 12.64,
    X_aft_cyl_beg = 11.22
    )

######################
# Thermal Management System
######################
from propsi import _load_all_props
props_tms = _load_all_props()
T_list = props_tms['T']  # Kelvin
P_list = props_tms['P']  # Pa
cp_meg = props_tms['CP']  # J/(kg·K)
k_meg = props_tms['K_k']  # W/(m·K)
dyn_visc_meg = props_tms['DYN_VISC']  # Pa·s
gamma_meg = props_tms['GAMMA']  # Specific heat ratio
Pr_meg = props_tms['PR']  # Prandtl number
rho_meg = props_tms['RHO']  # kg/m³


# TEG
efficiency_teg = 0.05

# Fan
fan_eff = 0.7
delta_pressure = 1000  # Pa, pressure drop across the fan
vel_fan = 100

# Ram Air HX
h_air = 250         # REF: Shah 
cp_air = 1005.0

# Skin HX
area_wing = 2.3
S_w = 28.79

# -- Air
prandtl_air = 0.71
reynolds_air = 1e7  # depends on temperature?

# HX
# -- FC
deltaT_fc = 20
HEX_1_deltaT = 20 # Evaporator HX
