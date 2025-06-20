import numpy as np
from CoolProp.CoolProp import PropsSI 
from CoolProp.CoolProp import PropsSI as call_propssi
import tms_plotting 
from global_constants import *
import time
from functools import lru_cache
import csv
import math


start_time = time.time()
# ------------------------------ PARAMETERS -----------------------------------
# TODO: 
# - Delete all parameters that are not used in the code
# - List all constants clearly and cite from literature


# Ambient Conditions --------------------------------
TO_conditions= {
    'T': 293.15,  # Ambient temperature in K
    'M': 0.157,  # Flight Mach number
    'V': 54.0167,  # Flight speed in m/s
    'rho': 1.225 # Air density at cruise in kg/m^3
}



# Main Classes for Heat Sinks (TMS) -------------------------------
    
class SkinHeatExchanger():
    def __init__(self, area, U, fluid):
        super().__init__()
        self.area = area
        self.U = U             # overall HT coefficient (W/m^2.K)
        self.coolant_temp = fluid.T
        self.ambient_temp = None          # to be set based on flight condition
        self.fluid = fluid                

    def set_ambient(self, T_ambient_K, recovery_factor=0.9, mach=0.3):
        # Set effective ambient (adiabatic wall) temperature for convection calculations
        T_aw = T_ambient_K * (1 + recovery_factor*0.5*(1.4-1)*mach**2)
        self.ambient_temp = T_aw 
    def absorb_heat(self, heat_w):
        if self.ambient_temp is None:
            raise RuntimeError("Ambient conditions not set for SkinHeatExchanger")

        Q_capacity = self.U * self.area * (self.coolant_temp - self.ambient_temp)
        Q_absorbed = min(heat_w, Q_capacity)
        t_coolant_out = self.coolant_temp - Q_absorbed / (self.fluid.cp * self.fluid.mf_given)
        if Q_absorbed < heat_w:
            return Q_absorbed, t_coolant_out
        else:
            return Q_absorbed, t_coolant_out
    
class RamAirHeatExchanger():
    def __init__(self, coolant_temp_K, fluid, ambient_conditions,p_amb):
        #self.U = U_W_per_m2K
        self.coolant_temp = coolant_temp_K
        self.Pr = fluid.Pr
        self.dyn_visc = fluid.dyn_visc
        self.mf_coolant = fluid.mf_given
        self.k_cool = fluid.k
        self.fluid = fluid

        # To be determined during sizing:
        self.U_ra = None
        self.required_area = 0.0    # m² of radiator face
        self.inlet_area = 0.0       # m² of total open hole area
        self.skin_area = 0.0        # m² of skin needed (porous area)
        self.hole_count = 0         # total number of small holes
        self.TR = 0.0
        #self.drag_N = 0.0           # N, raw drag before Meredith effect
        self.net_drag_N = 0.0       # N, after Meredith recovery
        self.ambient_conditions = ambient_conditions
        self.p_amb = p_amb
    
    def U_rad(self):

        # Coolant constants
        dh = 0.002    # 4*A/P  hydraulic diameter

        # Reynolds calc
        Re = 4*self.mf_coolant/(np.pi * dh * self.dyn_visc)
        # if Re < 2300:
        #     raise ValueError("Reynolds number in laminar regime, use a turbulent/combined relation correlation.")
        
        # Nusselt calc
        f = (0.79 * np.log(Re) - 1.64) ** -2
        nu = 0.023*Re**(4/5)*self.Pr**0.3        #nu = ((f/8)*(Re-1000)*self.Pr)/(1+12.7*(f/8)**0.5 * (self.Pr**0.66-1)) # Nusselt number for turbulent flow in tubes [m²/s]
        h_cool = self.k_cool * nu / dh
        wing_coolant_temp = 320  # K

        self.U_ra = 1 / (1/h_air + 1/h_cool)

        return self.U_ra


    def absorb_heat(self, heat_w):
        return heat_w

    def size_exchanger(self, heat_w, T_air_in_K):
        self.U_ra = self.U_rad()

        coolant_temp_out = self.coolant_temp - heat_w / (self.fluid.cp * self.fluid.mf_given)  

        # Iteration for delta T
        deltaT_grid = np.linspace(1, 150, 1000 )        
        tol = 0.009    
        AR = 3.6                               # “almost 1” → ±1 %

        for T in deltaT_grid:

            dT1 = self.coolant_temp - T_air_in_K #coolant_temp_out - T_air_in_K                      
            dT2 = self.coolant_temp - (T_air_in_K + T)                

            dT_lm = (dT2 - dT1) / np.log(dT2 / dT1)

            self.required_area = heat_w / (self.U_ra * dT_lm)
            beta  = 1100
            vol_rad = self.required_area/beta
            front_area_rad = vol_rad/0.019
            front_area_duct = front_area_rad/AR
            #print(front_area_duct, "duct m2")
            length_rad = np.sqrt(front_area_rad)
            if self.ambient_conditions['V'] < vel_fan:
                mf_air = front_area_duct * vel_fan * self.ambient_conditions['rho']
                fan = Fan(fan_eff=fan_eff, ra_mf=mf_air, ra_density=self.ambient_conditions['rho'], delta_pressure=delta_pressure)
                power_fan = fan.power()
            else:
                mf_air = front_area_duct * self.ambient_conditions['V'] * self.ambient_conditions['rho']
                power_fan = 0.0

            # --- ratio we want to drive to 1 -----------------------------------------
            ratio = (mf_air * cp_air * T) / (self.U_ra * dT_lm)
            if abs(ratio - self.required_area) < tol:           # close enough
                # print(mf_air, "kg/s")
                # print(front_area_rad, "front area rad")
                # print(T, "K")
                break


        self.delta_t_air = T
        A_0 = front_area_rad
        P_air = self.ambient_conditions['rho'] * self.ambient_conditions['V']**2 * 0.5 + self.p_amb 
        mu_air = PropsSI('VISCOSITY', 'P', P_air, 'T', self.ambient_conditions['T'], 'Air')
        g_c = 1  
        L = length_rad
        P = 4 * L
        G = mf_air / A_0 
        r_h = A_0/P
        D_h = 4 * r_h
        Re = G * D_h / mu_air
        f = 0.046 * Re**-0.2
        eff_duct = 0.7
        #AR = 3.6
        t_w = f /( 2 * g_c * self.ambient_conditions['rho'] / G**2 )
        pressure_drop_HX = mu_air /(2* g_c * self.ambient_conditions['rho']) * (4 * L /D_h**2) * (mf_air/A_0) * (f * Re) 
        pressure_drop_duct = (0.5*self.ambient_conditions['rho']*self.ambient_conditions['V']**2)*(eff_duct)*(1-1/AR**2)
        pressure_drop = pressure_drop_duct + pressure_drop_HX
        eta_p = 1 - pressure_drop / P_air 

        cool_out = Fluid_Coolant(name = "Coolant Out", T = coolant_temp_out, P = self.fluid.P, 
                         mf = self.fluid.mf_given, fluid_type = self.fluid.fluid_type)
        
        new_drag = True

        if new_drag == True:
            # Estimate exit velocity using bernoulli
            V_exit = np.sqrt(self.ambient_conditions['V']**2 - 2 * pressure_drop / self.ambient_conditions['rho'])
            D_g = mf_air * (self.ambient_conditions['V'] - V_exit ) #mf_air * (self.ambient_conditions['V'] - V_exit)  # mass flow rate of air through radiator
            F_n = (self.TR + 1 )*D_g
            net_drag = D_g - F_n
            C_D_rad = net_drag/(0.5*self.ambient_conditions['rho'] * self.ambient_conditions['V']**2 * S_w/2) # C_D for both radiator 
            #C_D_rad = D_g/(0.5*self.ambient_conditions['rho'] * self.ambient_conditions['V']**2 * S_w/2)
        else:
            D_g = mf_air * self.ambient_conditions['V']
            #C_D_rad = D_g/(0.5*self.ambient_conditions['rho'] * self.ambient_conditions['V']**2 * S_w/2) # C_D for both radiator 
            F_n = (self.TR + 1 )*D_g
            net_drag = D_g - F_n
            C_D_rad = net_drag/(0.5*self.ambient_conditions['rho'] * self.ambient_conditions['V']**2 * S_w/2) # C_D for both radiator 

        
        return self.required_area, cool_out, power_fan, eta_p, C_D_rad, length_rad, A_0, mf_air

    # THRUST RECOVERY   
    def thrust_ratio(self,gamma, R, M0, T0, eta_p07, CD_d_CD_sp):

        deltaT_HX0 = self.delta_t_air
        comp_ratio = (1.0 + 0.5 * (gamma - 1.0) * M0**2) * eta_p07 ** ((gamma - 1.0) / gamma)

        # Guard against unphysical D <= 1
        term2 = 1.0 - 1.0 / comp_ratio

        numerator = (2.0 * gamma / (gamma - 1.0) * R) * term2 * ( T0 * (1.0 + 0.5 * (gamma - 1.0) * M0**2) + deltaT_HX0)

        denominator = M0 * np.sqrt(gamma * R * T0) * (1.0 + CD_d_CD_sp / 2.0)
        
        TR = np.sqrt(numerator) / denominator - 1.0
        #TR = np.where(value > 0, np.sqrt(value) - 1.0, np.nan)  # nan for impossible points
        return TR
    
    '''
    # -- sweep ΔT_HX0 from 0 to 110 K --------------------------------------
    deltaT = np.linspace(0, 100, 100)
    eta_values = [0.995, 0.92, 0.90, 0.85]

    for eta in eta_values:
        TR = thrust_ratio(
            deltaT,
            gamma = k_air,
            R     = 287.058,
            M0    = ambient_conditions["M"],
            T0    = ambient_conditions["T"],
            eta_p07       = eta,   # optional override
            CD_d_CD_sp    = 0.11    # optional override
        )
        # plt.plot(deltaT, TR, label=f"ηₚ,07 = {eta:.3f}")

    # # # # # # print(TR[-1])
    # -- plot ---------------------------------------------------------------
    #plt.plot(deltaT, TR, marker="o")
    
    plt.xlabel("ΔT_HX⁰ (K)")
    plt.ylabel("Thrust Ratio (TR)")
    plt.title("Temperature difference of Radiator vs TR (for different pressure ratios)")
    plt.grid(True)
    plt.legend()
    plt.show()
    '''


class ThermoElectricGenerator():
    def __init__(self, hot_temp_K, cold_temp_K, efficiency):
        super().__init__("TEG")
        self.efficiency = efficiency
        self.hot_temp = hot_temp_K
        self.cold_temp = cold_temp_K
    def absorb_heat(self, heat_w):
        Q_absorbed = heat_w 
        Power_output = (1-self.efficiency) * Q_absorbed
        Q_rejected = Q_absorbed - Power_output

        return Power_output, Q_rejected  


# -------------------------------- Additional Components -------------------------------
class Fan():
    def __init__(self, fan_eff, ra_mf, ra_density, delta_pressure):
        self.fan_eff = fan_eff
        self.ra_mf = ra_mf
        self.ra_density = ra_density
        self.delta_pressure = delta_pressure
    def power(self):
        return (self.ra_mf * self.delta_pressure) / (self.fan_eff * self.ra_density)    
    
class Pump(): 
    def __init__(self, fluid, pump_eff, delta_pressure):
        self.pump_eff = pump_eff
        self.coolant_mf = fluid.mf_given
        self.coolant_density = fluid.rho
        self.delta_pressure = delta_pressure
    def power(self):
        return (self.coolant_mf * self.delta_pressure) / (self.pump_eff * self.coolant_density)

class HEX():
    def __init__(self, name, fluid_cold, fluid_hot, Q_req):
        self.name = name
        self.fluid_cold = fluid_cold
        self.fluid_hot = fluid_hot
        self.Q_req = Q_req
        self.density = 7900

    def pressure_drop(self,fluid, Re, L_plate, w_plate, d_plate, dh, n_passes, pipe_diam): 
        corr_angle = 30 * np.pi / 180 
        pipe_area = np.pi * (dh/2)**2  
        f = (corr_angle/30)**0.83 * ((30.2/Re)**5 + (6.28/Re**0.5)**5)**0.2
        Dh = 4*w_plate*d_plate / (2*(w_plate + d_plate))  
        drop_channel = 4 * f * (fluid.rho * (fluid.vdot/pipe_area)**2)/2 * L_plate * Dh
        D_coll = pipe_diam 
        drop_collector = 1.4 * n_passes * (fluid.mf_given / (np.pi * (D_coll**2/4)))**2 * 1/(2* fluid.rho) * 1

        return drop_channel + drop_collector
    
    def size(self,pipe_diam, alpha, type_hx = 'sh',t_cold_out_given = None):
        # Plate properties
        plate_thickness = 3.6 * 1e-3 #m 
        plate_thermal_conductivity = 17.5 # W/(m·K), SS
        size_factor = 1.15 #(1.15-1.25)
        gap_bt_plates = 3e-3  # from excel tool
        N_plates = 10          # from Michelle 

        # Pipe Properties
        N_passes = 1
        dh_coolant = 4*gap_bt_plates / size_factor  # diameter of coolant pipes
        dh_h2 = dh_coolant                          # diameter of hydrogen pipes
        

        #initial guess for mass flow rate of coolant
        #self.fluid_hot.mf_given = self.fluid_hot.mf_calculated # self.Q_req / (self.fluid_hot.cp * (self.fluid_hot.T - self.fluid_cold.T))
        
        L_h2 = 447000 # J/kg
        area = 0.5 # assumed area of heat exchanger plate in m² (total)
        H_hx = 500 # Overall heat exchange coefficient for HX [W/m².K]
        #self.fluid_cold.cp = 9384

        # Calculation
        Q = 1000
        iteration = 0
        factor_exists = False

        while abs(Q-self.Q_req) > 1e-5:
            if factor_exists:
                area *=factor
            '''
            t_hot_out = 0
            while t_hot_out < 273 - 48:
                self.fluid_hot.mf_given *= 1.2
                t_hot_out = -self.Q_req / (self.fluid_hot.mf_given * self.fluid_hot.cp) + self.fluid_hot.T
                # # # # # print(f"Iteration {iteration}: t_hot_out = {t_hot_out}, mf_hot = {self.fluid_hot.mf_given}")
            '''
            #t_hot_out = -self.Q_req / (self.fluid_hot.mf_given * self.fluid_hot.cp) + self.fluid_hot.T
            t_hot_in = self.fluid_hot.T
            t_cold_in = self.fluid_cold.T
            #t_cold_out = self.fluid_hot.T - 5
            #t_cold_out = t_cold_in + (self.fluid_cold.mf_given/self.fluid_hot.mf_given )*(L_h2/self.fluid_hot.cp)
            if type_hx == 'sh':
                t_cold_out = t_hot_in - 20
                t_hot_out =  t_hot_in - 100

            elif type_hx == 'evap':
                t_cold_out = t_cold_out_given
                t_hot_out = t_hot_in - 100
            delta_t1 = t_hot_out - t_cold_in
            delta_t2 = t_hot_in - t_cold_out
            
            self.fluid_hot.mf_given = self.Q_req / (self.fluid_hot.cp * (t_hot_in - t_hot_out))

            lmtd = (delta_t2 - delta_t1) / np.log(delta_t2 / delta_t1)
            S_eff = self.Q_req / (lmtd * H_hx)
            #s_eff = area * size_factor #assumed area of heat exchanger plate
            #N_plates = np.ceil(S_eff / s_eff)
            s_eff = (area)  * size_factor  # TOTAL EFF AREA
            N_plates = np.ceil(S_eff / s_eff)
            n_channels = np.ceil((N_plates-1) / 2)
            volume = N_plates * area * (plate_thickness+gap_bt_plates)  # m³
            
            # Coolant
            C_h_coolant = 0.348 # Chevron angle of 30deg
            y_coolant = 0.663   # Chevron angle of 30deg
            R_fh = 0.0003526
            Pr_coolant = self.fluid_hot.mu * self.fluid_hot.cp / self.fluid_hot.k
            Re_coolant = self.fluid_hot.mf_calculated * 4/  (np.pi * dh_coolant * self.fluid_hot.mu) 
            Nu_coolant = C_h_coolant * Re_coolant**y_coolant * Pr_coolant**0.33 * 1 # Assuming (mu/mu_w)**0.17 = 1, given the tubes are very narrow
            hc_hot = Nu_coolant * self.fluid_hot.k / dh_coolant

            # H2
            C_h_h2 = 0.348  # Chevron angle of 30deg
            y_h2 = 0.663    # Chevron angle of 30deg
            R_fc = 8.815 * 1e-5 # for vapour
            Pr_h2 = self.fluid_cold.Pr  # Prandtl number for H2
            Re_h2 = self.fluid_cold.mf_calculated * 4 / (np.pi * dh_h2 * self.fluid_cold.mu) 
            Nu_h2 = C_h_h2 * Re_h2**y_h2 * Pr_h2**0.33 * 1 # Assuming (mu/mu_w)**0.17 = 1, given the tubes are very narrow
            hc_cold = Nu_h2 * self.fluid_cold.k / dh_h2


            # Recalculate overall heat transfer coefficient
            H_hx = 1/ (1/hc_cold + 1/hc_hot + R_fc + R_fh + plate_thickness/plate_thermal_conductivity)

            # Recalculate the heat transferred
            Q = H_hx * s_eff * lmtd * N_plates  * N_passes 
            factor = self.Q_req / Q 
            hole_area = np.pi * (dh_coolant/2)**2
            ratio =  hole_area / area
            iteration += 1
            factor_exists = True


        w_plate = np.sqrt(area/2)
        L_plate = 2 * w_plate
        drop_h2 = self.pressure_drop(self.fluid_cold, Re_h2, L_plate, w_plate, gap_bt_plates, dh_h2, N_passes, pipe_diam)
        drop_coolant = self.pressure_drop(self.fluid_hot, Re_coolant, L_plate, w_plate, gap_bt_plates, dh_coolant, N_passes, pipe_diam)

        #Coolant In
        self.fluid_hot.T = t_hot_in
        cool_in = self.fluid_hot

        #Coolant Out
        cool_out = Fluid_Coolant(name = "Coolant Out", T = t_hot_out, P = self.fluid_hot.P - drop_coolant, 
                         mf = self.fluid_hot.mf_given * 0.5, fluid_type = self.fluid_hot.fluid_type)
        
        #H2 Out
        if type_hx == 'sh':
            H2_out = Fluid(name = "H2_5", T = t_cold_out, P = self.fluid_cold.P - drop_h2, mf = self.fluid_cold.mf_given, fluid_type = self.fluid_cold.fluid_type)
        elif type_hx == 'evap':
            H2_out = Fluid(name = "H2_3", T = t_cold_out_given, P = self.fluid_cold.P - drop_h2, mf = self.fluid_cold.mf_given, fluid_type = self.fluid_cold.fluid_type)
        

        return cool_in, cool_out, H2_out, area, N_plates   

class Compressor():
    def __init__(self, comp_efficiency, pressure_ratio, fluid):
        self.efficiency = comp_efficiency
        self.pressure_ratio = pressure_ratio
        self.fluid = fluid  

    def power(self):
        inlet_temperature = self.fluid.T  
        gamma = self.fluid.gamma 
        T_out = inlet_temperature * (1 + 1 / self.efficiency * (self.pressure_ratio ** ((gamma - 1) / gamma) - 1))  # Isentropic relation

        cp = self.fluid.cp  
        mdot = self.fluid.mf_given
        power_req = mdot * cp * (T_out - inlet_temperature) / self.efficiency  # Power in Watts

        return power_req
    
    def mass(self, power):
        return power/1000 * 0.0400683 + 5.17242

class Turbine():
    def __init__(self, turbine_efficiency, fluid):
        self.efficiency = turbine_efficiency
        self.fluid = fluid

    def power(self, T_in, T_out):
        # Calculate the power produced by the turbine
        cp_gas = self.fluid.cp
        mdot = self.fluid.mf_given
        power_provide = mdot * cp_gas * (T_in - T_out) / self.efficiency  # Power in Watts\
        return power_provide
    
    def mass(self, power):
        # MASSIVE assumption of turbine mass being calculated in the same way as compressor
        return power/1000 * 0.0400683 + 5.17242

class Valve():
    def __init__(self, fluid, valve_efficiency = 0.9 ):
        self.efficiency = valve_efficiency
        self.fluid = fluid

    def valve_mass(self):
        mdot_coolant = self.fluid.mf_given
        return (0.568 * mdot_coolant ** 0.5541)

class Fluid():
    def __init__(self, name, T, P, mf, fluid_type, C = None):
        sat_temp = 0
        gas = True
        if fluid_type == 'ParaHydrogen' or fluid_type == 'Water':
            if P < PropsSI('PCRIT', fluid_type):
                sat_temp = PropsSI('T', 'P', P, 'Q', 0, fluid_type)  # Saturation temperature at given pressure
            if abs(T - sat_temp) < 1e-3:
                gas = False
        self.name = name
        self.T = T  # Temperature in Kelvin
        self.P = P  # Pressure in Pascals
        self.cp = PropsSI('C', 'P', P, 'T', T, fluid_type) if gas else PropsSI('C', 'P', P, 'Q', 0, fluid_type)  # Specific heat capacity at constant pressure
        self.C = C if C is not None else self.cp * mf  # Heat capacity rate
        self.mf_calculated = self.C/self.cp
        self.mf_given = mf
        self.k = PropsSI('CONDUCTIVITY', 'P', P, 'T', T, fluid_type) if gas else PropsSI('CONDUCTIVITY', 'P', P, 'Q', 0, fluid_type)  # Thermal conductivity of LH2 at storage pressure
        self.dyn_visc = PropsSI('VISCOSITY', 'P', P, 'T', T, fluid_type) if gas else PropsSI('VISCOSITY', 'P', P, 'Q', 0, fluid_type) # Viscosity of LH2 at storage pressure
        self.gamma = PropsSI('CP0MASS', 'P', P, 'T', T, fluid_type) / PropsSI('CVMASS', 'P', P, 'T', T, fluid_type) if gas else PropsSI('CP0MASS', 'P', P, 'Q', 0, fluid_type) / PropsSI('CVMASS', 'P', P, 'Q', 0, fluid_type) # Heat capacity ratio of LH2 at storage pressure
        self.Pr = PropsSI('PRANDTL', 'P', P, 'T', T, fluid_type) if gas else PropsSI('PRANDTL', 'P', P, 'Q', 0, fluid_type) # Prandtl number of LH2 at storage pressure
        self.rho = PropsSI('D', 'P', P, 'T', T, fluid_type) if gas else PropsSI('D', 'P', P, 'Q', 0, fluid_type) # Density of LH2 at storage pressure
        self.vdot = mf / self.rho  # Volumetric flow rate in m³/s
        self.kin_visc = self.dyn_visc / self.rho  # Kinematic viscosity of LH2 at storage pressure
        self.fluid_type = fluid_type
        self.mu = self.dyn_visc  # Dynamic viscosity of the fluid

class Fluid_Coolant():
    def __init__(self, name, T, P, mf, fluid_type, C = None):
        self.name = name
        self.T = T  # Temperature in Kelvin
        self.P = P  # Pressure in Pascals
        i_T = np.searchsorted(T_list, T)
        i_P = np.searchsorted(P_list, P)
        if i_P == 1000:
            i_P = 999
        if i_T == 1000:
            i_T = 999
        self.cp = cp_meg[i_T, i_P] 
        self.C = C if C is not None else self.cp * mf  # Heat capacity rate
        self.mf_calculated = self.C/self.cp
        self.mf_given = mf
        self.k = k_meg[i_T, i_P]  
        self.dyn_visc = dyn_visc_meg[i_T, i_P]
        self.gamma = gamma_meg[i_T, i_P]
        self.Pr = Pr_meg[i_T, i_P]
        self.rho = rho_meg[i_T, i_P] 
        self.vdot = mf / self.rho 
        self.kin_visc = self.dyn_visc / self.rho  
        self.fluid_type = fluid_type
        self.mu = self.dyn_visc 
    
class Pipe():
    def __init__(self, length, d_in, fluid, type = 'normal', cryo = False):
        """Initialize the Heat Pipe Analyzer"""
        self.length = length
        self.d_in = d_in
        if isinstance(fluid, Fluid_Coolant):
            self.d_out = 0.027
            self.mat_density = 0.9e3
        else:
            self.d_out = self.d_in + 1e-3 # https://www.swagelok.com/downloads/webcatalogs/EN/MS-01-107.PDF
            self.mat_density = 7900
        if cryo:
            self.d_is = self.d_out * 5e-3 
        else:
            self.d_is = self.d_out * 1e-3
        self.temp_fluid = fluid.T
        self.temp_air = 273.15 + 20
        self.therm_conduc_fluid = fluid.k
        self.prandtl_number = fluid.Pr
        self.mass_flow = fluid.mf_given
        self.spec_heat = fluid.cp
        self.heat_trans_air = 10
        self.therm_conduc_pipe = 15
        self.therm_conduc_insulation = 15
        self.density_fluid = fluid.rho
        self.kinematic_visc = fluid.kin_visc
        self.material_roughness = 0.004e-3
        self.fluid = fluid  # Store the fluid object for later use
        self.type = type

    def analyze_heat_pipe(self, out_name):
        """
        Combined analysis of heat pipe: calculate both pressure drop and temperature distribution

        Parameters:
        length: pipe length
        d_in: inner diameter
        d_out: outer diameter
        d_is: insulation outer diameter
        temp_fluid: initial fluid temperature
        temp_air: ambient air temperature
        therm_conduc_fluid: thermal conductivity of fluid
        prandtl_number: Prandtl number
        mass_flow: mass flow rate
        spec_heat: specific heat of fluid
        heat_trans_air: heat transfer coefficient air
        therm_conduc_pipe: thermal conductivity of pipe material
        therm_conduc_insulation: thermal conductivity of insulation material
        density_fluid: fluid density
        kinematic_visc: kinematic viscosity of fluid
        material_roughness: absolute roughness of pipe material

        Returns:
        dict: {'pressure_drop': float, 'temperature_profile': np.array, 'reynolds': float}
        """
        # Calculate fluid velocity
        velocity_fluid = abs(self.mass_flow / (self.density_fluid * ((self.d_in / 2) ** 2) * np.pi))

        # Calculate Reynolds number
        reynolds = velocity_fluid * self.d_in / self.kinematic_visc
        

        # ==================== PRESSURE DROP CALCULATION ====================
        # Check for laminar or turbulent flow and calculate Darcy-Weisbach coefficient
        if reynolds < 2300:
            darcy_weisbach = 64 / reynolds
        else:
            x_darcy = (-2.457 * np.log((7 / reynolds) ** 0.9 + 0.27 * self.material_roughness / self.d_in)) ** 16
            y_darcy = (37530 / reynolds) ** 16
            darcy_weisbach = 8 * ((8 / reynolds) ** 12 + (x_darcy + y_darcy) ** -1.5) ** (1 / 12)

        # Calculate pressure drop at the end of the pipe
        pressure_drop = self.length * (darcy_weisbach * self.density_fluid * velocity_fluid ** 2) / (self.d_in * 2)

        # ==================== TEMPERATURE PROFILE CALCULATION ====================
        # Initialize temperature array
        temperature = [self.temp_fluid]
        energies = 0
        steps = 100

        # Calculate heat transfer coefficient
        nu_turb = 0.0223 * (reynolds ** 0.8) * self.prandtl_number ** 0.4
        heat_trans_co = nu_turb * self.therm_conduc_fluid / self.d_in

        # Calculate surface areas
        surface_in = np.pi * self.d_in * (self.length)
        surface_out = np.pi * self.d_is * (self.length / steps)
        

        # Calculate thermal resistances
        r_fluid = (1 / (self.mass_flow * self.spec_heat * (1 - np.exp(-1 * heat_trans_co * surface_in / (self.mass_flow * self.spec_heat))))) * steps
        r_air = 1 / (self.heat_trans_air * surface_out)
        r_wall = np.log(self.d_out / self.d_in) / (2 * np.pi * self.therm_conduc_pipe * (self.length / steps))
        r_insulation = np.log(self.d_is / self.d_out) / (2 * np.pi * self.therm_conduc_insulation * (self.length / steps))
        r_tot = r_fluid + r_air + r_wall + r_insulation


        # Calculate time in pipe
        time_in_pipe = self.length / velocity_fluid


        temps_fluid = self.temp_fluid
        mass_section = (self.length / steps) * (self.d_in / 2) ** 2 * np.pi * self.density_fluid

        # Calculate temperature at each step
        for i in range(steps):
            delta_q = (temps_fluid - self.temp_air) / r_tot
            delta_t = time_in_pipe * delta_q / (mass_section * self.spec_heat) / steps
            temps_fluid = temps_fluid - delta_t
            temperature.append(temps_fluid)
            energies = energies + delta_q
        
        energy_lost = energies
        if self.fluid.fluid_type == 'ParaHydrogen':
            output_fluid = Fluid(name=out_name , T=self.fluid.T, P=self.fluid.P, mf=self.fluid.mf_given, fluid_type=self.fluid.fluid_type)
        else:
            output_fluid = Fluid_Coolant(name=out_name , T=self.fluid.T, P=self.fluid.P, mf=self.fluid.mf_given, fluid_type=self.fluid.fluid_type)
        if self.type == 'normal':
            output_fluid.P -= pressure_drop
            output_fluid.T = temperature[-1]  # Update fluid temperature to final value
        if self.type == 'inv':
            output_fluid.P += pressure_drop
            output_fluid.T += (temperature[0]- temperature[-1]) 

        return output_fluid

    def mass(self):
        """
        Calculate the mass of the pipe
        Returns:
        float: mass of the pipe in kg
        """
    
        mass = self.mat_density * self.length * np.pi * ((self.d_out / 2) ** 2 - (self.d_in / 2) ** 2)
        mass += (self.d_in/2)**2 * np.pi * self.length * self.fluid.rho
        return mass


def size_pipes_h2(h2_mf_fc, h2_mf_cc, p_sto,fluid,diam_est):
    sf_delta_p = 2e5
    pressure_drop = p_sto -4.511e5 - sf_delta_p 
    total_pressure_drop = 0

    # First half
    length1 = 6.3425
    diam1 = diam_est
    mf1 = (h2_mf_fc + h2_mf_cc)
    v_fluid1 = mf1 / (fluid.rho * ((diam1 / 2) ** 2) * np.pi)
    reynolds1 = v_fluid1 * diam1 / fluid.kin_visc

    if reynolds1 < 2300:
            darcy_weisbach = 64 / reynolds1
    else:
        x_darcy = (-2.457 * np.log((7 / reynolds1) ** 0.9 + 0.27 * 0.004e-3 / diam1)) ** 16
        y_darcy = (37530 / reynolds1) ** 16
        darcy_weisbach = 8 * ((8 / reynolds1) ** 12 + (x_darcy + y_darcy) ** -1.5) ** (1 / 12)
    pressure_drop1 = length1 * (darcy_weisbach * fluid.rho * v_fluid1 ** 2) / (diam1 * 2)

    # Second half
    diam2 = diam_est * h2_mf_cc / (h2_mf_cc + h2_mf_fc) 
    length2 = 3.225
    mf2 = h2_mf_cc
    v_fluid2 = mf2 / (fluid.rho * ((diam2 / 2) ** 2) * np.pi)
    reynolds2 = v_fluid2 * diam2 / fluid.kin_visc

    if reynolds2 < 2300:
            darcy_weisbach = 64 / reynolds2
    else:
        x_darcy = (-2.457 * np.log((7 / reynolds2) ** 0.9 + 0.27 * 0.004e-3 / diam2)) ** 16
        y_darcy = (37530 / reynolds2) ** 16
        darcy_weisbach = 8 * ((8 / reynolds2) ** 12 + (x_darcy + y_darcy) ** -1.5) ** (1 / 12)
    pressure_drop2 = length2 * (darcy_weisbach * fluid.rho * v_fluid2 ** 2) / (diam2 * 2)

    # Total pressure drop
    total_pressure_drop = pressure_drop1 + pressure_drop2
    press_err = total_pressure_drop - pressure_drop
    '''
    if press_err < 0:
        # # # # # print(f"Selected pipe diameter {diam_est:.3f} m is adequate, pressure drop is {total_pressure_drop:.2f} Pa, with an estimated maximum allowed of {pressure_drop:.2f} Pa")
    else: 
        # # # # # print(f"Increase pipe diameter, pressure drop is {total_pressure_drop:.2f} Pa, with an estimated maximum allowed of {pressure_drop:.2f} Pa")
    '''

    
# ------------------------------ MAIN PROGRAM -----------------------------------


def tms_main(Q_dot_fc_l, Q_dot_eps_l, p_fc_l, p_cc_l, h2_mf_fc_l, h2_mf_cc_l, T_fc_l, T_cc_l, air_mf_fc_l, T_amb_l, rho_amb_l, V_amb_l, p_amb_l, h2_mf_rec_l, air_out_fc_l, p_sto_l,h2o_mf_fc_l):
    m_pipe_h2_12 = [0]
    m_pipe_h2_34 = [0]
    m_sh = [0]
    m_pipe_cool_201 = [0]
    m_vap = [0]
    m_pipe_h2_56 = [0]
    m_valve_6711 = [0]
    m_pipe_h2_1112 = [0]
    m_valve_121326 = [0]
    m_pipe_h2_1314 = [0]
    m_pipe_h2_2618 = [0]
    m_comp_1415 = [0]
    m_pipe_h2_1516 = [0]
    m_pipe_h2_1719 = [0]
    m_valve_1920 = [0]
    m_pipe_2021 = [0]
    m_pipe_h2_78 = [0]
    m_pipe_valve89 = [0]
    m_pipe_h2_2223 = [0]
    m_pipe_h2_910 = [0]
    m_comp_2324 = [0]
    m_pipe_h2_2425 = [0]
    m_pipe_cool_2_3 = [0]
    m_pipe_cool_4_5 = [0]
    m_was_list = [0]
    m_pipe_cool_1922 = [0]
    m_pipe_cool_2120 = [0]
    m_pipe_cool_12_13 = [0]
    m_valve_132014 = [0]
    m_pipe_cool_1415 = [0]
    m_pump_water = [0]
    water_turb_p_list = [0]
    m_valve_151617 = [0]
    m_pipe_cool_1718 = [0]
    m_skin_hx = [0]
    m_pipe_cool_23 = [0]
    m_rads = [0]
    m_pipe_cool_26 = [0]
    m_pipe_cool_25 = [0]
    m_fans_rad = [0]
    air_turb_p_list = [0]
    m_pipe_cool_27 = [0]
    drags = [0]
    fan_powers = [0]
    m_valve_water = [0]
    m_pipe_cool_24 = [0]
    pump_26_27_power_list = [0]
    m_pipe_cool_16_24 = [0]
    m_fans_1 = [0]
    air_comp_power_list = [0]
    m_pump_2627 = [0]
    fan_powers_1 = [0]
    water_turbine_mass_list = [0]
    air_comp_mass_list = [0]
    air_turbine_mass_list = [0]
    p_comp_2324 = [0]
    pump_water_power_list = [0]
    p_comp_1415 = [0]
    Q_heater = np.zeros(len(Q_dot_fc_l)) 

    # Check for NaN in any input list
    input_lists = [
        Q_dot_fc_l, Q_dot_eps_l, p_fc_l, p_cc_l, h2_mf_fc_l, h2_mf_cc_l, T_fc_l, T_cc_l,
        air_mf_fc_l, T_amb_l, rho_amb_l, V_amb_l, p_amb_l, h2_mf_rec_l, air_out_fc_l, p_sto_l, h2o_mf_fc_l
    ]
    for lst in input_lists:
        if any((x is None) or (isinstance(x, float) and np.isnan(x)) for x in (lst if isinstance(lst, (list, np.ndarray)) else [lst])):
            return None, None
    
    for i in range(len(h2_mf_fc_l)): 
        Q_dot_fc = Q_dot_fc_l[i]  # W
        Q_dot_eps = Q_dot_eps_l[i]
        p_fc = p_fc_l[i]  # Pa
        p_cc = p_cc_l[i]  # Pa
        h2_mf_fc = h2_mf_fc_l[i]  # kg/s
        h2_mf_cc = h2_mf_cc_l[i]  # kg/s
        T_fc = T_fc_l[i]   # K
        T_cc = T_cc_l[i]   # K
        air_mf_fc = air_mf_fc_l[i]
        T_amb = T_amb_l[i] 
        rho_amb = rho_amb_l[i]  # kg/m³
        V_amb = V_amb_l[i]
        p_amb = p_amb_l[i]
        h2_mf_rec = h2_mf_rec_l[i]
        air_out_fc = air_out_fc_l[i]
        p_sto = p_sto_l[i]  # Pa, storage pressure
        h2o_mf_fc = h2o_mf_fc_l[i]  # kg/s, water mass flow rate to fuel cell
        
        
        ambient_conditions= {
            'T': T_amb,  # Ambient temperature in K
            'M': 0.4,  # Flight Mach number
            'V': V_amb,  # Flight speed in m/s
            'rho': rho_amb  # Air density at cruise in kg/m^3
        }


        # Ram Air HX -------------------------------
        # Air properties
        T_air_in = ambient_conditions['T']
        T_air_out = ambient_conditions['T'] + 20.0

        # Coolant and HX Properties 
        ra_coolant_temp = 163 + 20 + 273.15 # K

        # Fan 
        ra_density = ambient_conditions['rho']  # kg/m³, air density

        #Skin HX -----------------------------------------------	

        # Air
        h_ext_w = ambient_conditions['rho'] * ambient_conditions['V'] * cp_air * 0.185 * (np.log10(reynolds_air))**-2.584 * prandtl_air**-0.66
        recovery_factor = prandtl_air ** 0.33  

        coolant = 'INCOMP::MEG-60%'
        # coolant = 'Water'
        p_cool = 5.7e5
        fc_press_drop_cool = 100000
        cool_0 = Fluid_Coolant(name="FuelCellCoolantGeneric", T=T_fc, P=p_cool, C=None, mf=10, fluid_type=coolant)  # Coolant to FC
        cool_mf_per_fc = Q_dot_fc / (cool_0.cp * deltaT_fc * 2)
        cool_0.mf_given = cool_mf_per_fc  # Mass flow rate of coolant to FC

        alpha_sh = 1.2
        alpha_evap = 1.2

        m_front = 0
        m_mid = 0
        m_rear = 0

        

        Q_dot_rem = Q_dot_fc + Q_dot_eps  

        h2_mf_fc = h2_mf_fc   # Total H2 mass flow rate to fuel cell (both wings)
        h2_mf_cc = h2_mf_cc   # Total H2 mass flow rate to combustion chamber (both wings)

        # Calculate pipe diameter for H2
        h2_test = Fluid(
            name="H2_Test",
            T=PropsSI('T', 'P', p_sto, 'Q', 1, 'ParaHydrogen'),  # Q=1 for saturated vapor (gaseous H2)
            P=p_sto,
            mf=h2_mf_fc + h2_mf_cc,
            fluid_type='ParaHydrogen'
        )
        diam_est = 0.05
        diam_est_cool =  0.019 # Calculated from 2m/s flow velocity in longest coolant pipes
        ratio_h2_to_fc = h2_mf_fc / (h2_mf_fc + h2_mf_cc)  
        
        size_pipes_h2(h2_mf_fc, h2_mf_cc, p_sto,h2_test, diam_est)

        # Initialize -----------------------------
        T_tank = PropsSI('T', 'P', p_sto, 'Q', 0, 'ParaHydrogen')  # Initial temperature of LH2 tank
        h2_1 = Fluid(name="H2_1", T=T_tank, P=p_sto, C = None, mf = h2_mf_fc + h2_mf_cc, fluid_type='ParaHydrogen')

        # print(f"Initial H2_1: {h2_1.name}, T: {h2_1.T}, P: {h2_1.P}, mf: {h2_1.mf_given}, fluid_type: {h2_1.fluid_type}")
        # print(f"mf_h2_fc: {h2_mf_fc}, mf_h2_cc: {h2_mf_cc}, p_sto: {p_sto}, diam_est: {diam_est}")
        # Pipe h2 12
        pipe_h2_12 = Pipe(1.42, diam_est, h2_1, cryo=True) 
        m_pipe_h2_12.append(pipe_h2_12.mass())
        h2_2 = pipe_h2_12.analyze_heat_pipe('H2_2')

        # Pipe h2 34
        #print(f"Pipe h2 12: {h2_2.name}, T: {h2_2.T}, P: {h2_2.P}, mf: {h2_2.mf_given}, fluid_type: {h2_2.fluid_type}")
        T_sat_h2 = PropsSI('T', 'P', h2_2.P, 'Q', 0, h2_2.fluid_type)  # Saturation temperature of H2 at pressure P
        h2_3 = Fluid(name="H2_3", T=T_sat_h2 + HEX_1_deltaT, P=p_sto, C = None, mf = h2_mf_fc + h2_mf_cc, fluid_type='ParaHydrogen')
        pipe_h2_34 = Pipe(1.42, diam_est, h2_3) 
        m_pipe_h2_34.append(pipe_h2_34.mass())
        h2_4 = pipe_h2_34.analyze_heat_pipe('H2_4')

        # Superheater HEX
        cool_19 = Fluid_Coolant(name="Cool_19", T=T_fc, P=p_cool-fc_press_drop_cool, C=None, mf = cool_mf_per_fc * 2, fluid_type=coolant) 
        t_cold_out_given = T_sat_h2 + HEX_1_deltaT
        Q_h2_heat = (PropsSI('H', 'P', h2_2.P, 'T', T_fc-10, h2_2.fluid_type) - PropsSI('H', 'P', h2_2.P, 'T', t_cold_out_given, h2_2.fluid_type)) * h2_2.mf_given   # Heat of vaporization
        if Q_h2_heat > Q_dot_rem:
            Q_heater[i] += Q_h2_heat - Q_dot_rem

        Q_h2_heat = min(Q_h2_heat, Q_dot_rem)
        hx_heat = HEX(name="SuperHeater", fluid_cold=h2_4, fluid_hot=cool_19, Q_req=Q_h2_heat)
        _,cool_20,h2_5, hx_heat_area, hx_heat_n_plates = hx_heat.size(diam_est, type_hx='sh',alpha=alpha_sh) 
        cool_20_new = Fluid_Coolant(name="Cool_20_new", T=cool_20.T, P=cool_20.P, C=None, mf=cool_19.mf_given, fluid_type=cool_19.fluid_type)  # Coolant out of Superheater
        mass_superheater_hx = hx_heat_n_plates * hx_heat_area * hx_heat.density * 3.6 * 1e-3
        m_sh.append(mass_superheater_hx)
        Q_dot_rem -= Q_h2_heat 

        # Pipe cool 20-1
        ratio_to_hex = (cool_20_new.mf_given/(2*cool_mf_per_fc)) * 2
        pipe_cool_20_1 = Pipe(1.42, diam_est_cool *ratio_to_hex, cool_20_new)  
        m_pipe_cool_201.append(pipe_cool_20_1.mass())
        cool_1 = pipe_cool_20_1.analyze_heat_pipe("Cool_1")
        cool_1_keep = Fluid_Coolant(name="Cool_1_keep", T=cool_1.T, P=cool_1.P, C=None, mf=cool_1.mf_given, fluid_type=cool_1.fluid_type)  # Coolant out of Superheater
        
        if Q_dot_rem > 0:
            # Vaporizer HEX
            Q_vap = (PropsSI('H', 'P', h2_2.P, 'T', t_cold_out_given, h2_2.fluid_type) - PropsSI('H', 'P', h2_2.P, 'Q', 0, h2_2.fluid_type)) * h2_2.mf_given   # Heat of vaporization
            if Q_vap > Q_dot_rem:
                Q_heater[i] += Q_vap - Q_dot_rem
            Q_vap = min(Q_vap, Q_dot_rem) 
            hx_vap = HEX(name="Vaporizer", fluid_cold=h2_2, fluid_hot=cool_1_keep, Q_req=Q_vap)
            _,cool_2,h2_3, hx_vap_area, hx_vap_n_plates = hx_vap.size(diam_est, t_cold_out_given=t_cold_out_given, type_hx='evap',alpha = alpha_evap)
            cool_2 = Fluid_Coolant(name="Cool_2", T=cool_2.T, P=cool_2.P, C=None, mf=cool_1_keep.mf_given, fluid_type=cool_1.fluid_type)  
            mass_vap_hx = hx_vap_n_plates * hx_vap_area * hx_vap.density * 3.6 * 1e-3
            Q_dot_rem -= Q_vap
        
        else:
            h2_3 = Fluid(name="H2_3", T=h2_2.T, P=h2_2.P, C=0, mf=h2_2.mf_given, fluid_type='ParaHydrogen')  # No vaporization needed
            cool_2 = Fluid_Coolant(name="Cool_2", T=cool_1.T, P=cool_1.P, C=0, mf=cool_1.mf_given, fluid_type=cool_1.fluid_type)  # No vaporization needed
            mass_vap_hx = 0

        m_vap.append(mass_vap_hx)

        # Pipe h2 5-6
        pipe_h2_5_6 = Pipe(4.96, diam_est, h2_5)  
        m_pipe_h2_56.append(pipe_h2_5_6.mass())
        h2_6 = pipe_h2_5_6.analyze_heat_pipe("H2_6")

        # Intersection 6-11-7
        h2_7 = Fluid(name="H2_7", T=h2_6.T, P=h2_6.P, C=0, mf=h2_mf_fc, fluid_type='ParaHydrogen')  # H2 to fuel cell
        h2_11 = Fluid(name="H2_11", T=h2_6.T, P=h2_6.P, C=0, mf=h2_mf_cc, fluid_type='ParaHydrogen')  # H2 to CC
        valve_6711 = Valve(fluid=h2_6, valve_efficiency=0.9)  # Valve for H2 to fuel cell
        valve_6711_mass = valve_6711.valve_mass()  # Mass of the valve
        m_valve_6711.append(valve_6711_mass)

        # PATH to CC ----------------------------------------------
        # Pipe h2 11-12
        pipe_h2_11_12 = Pipe(0.43, diam_est * (1-ratio_h2_to_fc), h2_11) 
        m_pipe_h2_1112.append(pipe_h2_11_12.mass())
        h2_12 = pipe_h2_11_12.analyze_heat_pipe("H2_12")

        # Intersection 12-13-26
        comp_14_15_PI = 5
        split_13_26 = 1- ((p_cc+1 - h2_12.P * comp_14_15_PI) / (h2_12.P * (1 - comp_14_15_PI)))
        h2_13 = Fluid(name="H2_13", T=h2_12.T, P=h2_12.P, C=0, mf=h2_mf_cc * split_13_26, fluid_type='ParaHydrogen')  # H2 to Compressor
        h2_26 = Fluid(name="H2_26", T=h2_12.T, P=h2_12.P, C=0, mf=h2_mf_cc * (1-split_13_26), fluid_type='ParaHydrogen')  # H2 around Compressor
        valve_121326 = Valve(fluid=h2_12, valve_efficiency=0.9)  # Valve for H2 to CC
        valve_121326_mass = valve_121326.valve_mass()  # Mass of the valve
        m_valve_121326.append(valve_121326_mass)

        # Pipe h2 13-14
        pipe_h2_13_14 = Pipe(0.29, diam_est * (1-ratio_h2_to_fc), h2_13)  
        m_pipe_h2_1314.append(pipe_h2_13_14.mass())
        h2_14 = pipe_h2_13_14.analyze_heat_pipe("H2_14")

        # Pipe h2 26-18
        pipe_h2_26_18 = Pipe(0.64, diam_est * (1-ratio_h2_to_fc), h2_26)  
        m_pipe_h2_2618.append(pipe_h2_26_18.mass())
        h2_18 = pipe_h2_26_18.analyze_heat_pipe('H2_18')

        # Compressor 14-15
        comp_14_15 = Compressor(comp_efficiency=0.9, pressure_ratio=comp_14_15_PI, fluid=h2_14)
        comp14_15_power = comp_14_15.power()
        p_comp_1415.append(comp14_15_power)
        comp_14_15_mass = comp_14_15.mass(comp14_15_power)
        m_comp_1415.append(comp_14_15_mass)
        h2_15 = Fluid(name="H2_15", T=h2_14.T, P=h2_14.P * comp_14_15_PI, C=0, mf=h2_14.mf_given, fluid_type='ParaHydrogen')

        # Pipe h2 15-16
        pipe_h2_15_16 = Pipe(0.28, diam_est * (1-ratio_h2_to_fc), h2_15)
        m_pipe_h2_1516.append(pipe_h2_15_16.mass())
        h2_16 = pipe_h2_15_16.analyze_heat_pipe("H2_16")

        # Intersection 16-17-18
        h2_17 = Fluid(name="H2_17", T=h2_16.T, P=p_cc + 1e5, C=0, mf=h2_mf_cc, fluid_type='ParaHydrogen')  # H2 to CC

        # Pipe h2 17-19
        pipe_h2_17_19 = Pipe(0.28, diam_est * (1-ratio_h2_to_fc), h2_17)  
        m_pipe_h2_1719.append(pipe_h2_17_19.mass())
        h2_19 = pipe_h2_17_19.analyze_heat_pipe("H2_19")

        # Valve 19-20
        valve_1920 = Valve(fluid=h2_19, valve_efficiency=0.9) 
        valve_1920_mass = valve_1920.valve_mass()  # Mass of the valve
        m_valve_1920.append(valve_1920_mass)
        h2_20 = Fluid(name="H2_20", T=h2_19.T, P=p_cc, C=0, mf=h2_mf_cc, fluid_type='ParaHydrogen')  # H2 to CC

        # Pipe h2 20-21
        h2_20_copy = Fluid(name="H2_20_copy", T=h2_20.T, P=h2_20.P, C=0, mf=h2_mf_cc/2, fluid_type='ParaHydrogen')
        pipe_h2_20_21 = Pipe(0.28, (diam_est*(1-ratio_h2_to_fc))/2, h2_20_copy) 
        m_pipe_2021.append( 2 * pipe_h2_20_21.mass())
        h2_21 = pipe_h2_20_21.analyze_heat_pipe("H2_21")

        # PATH to FC -------------------------------------------------
        pipe_h2_7_8 = Pipe(1.56, diam_est * ratio_h2_to_fc, h2_7) 
        m_pipe_h2_78.append(pipe_h2_7_8.mass()) 
        h2_8 = pipe_h2_7_8.analyze_heat_pipe("H2_8")
        
        valve_8_9 = Valve(fluid=h2_8, valve_efficiency=0.9)  # Valve for H2 to fuel cell
        valve_8_9_mass = valve_8_9.valve_mass()  # Mass of the valve
        m_pipe_valve89.append(2*valve_8_9_mass)
        h2_9 = Fluid(name="H2_9", T=h2_8.T, P=p_fc, C=0, mf=h2_mf_fc, fluid_type='ParaHydrogen')  # H2 to fuel cell

        # Pipe h2 9-10
        h2_9_copy = Fluid(name="H2_9_copy", T=h2_9.T, P=h2_9.P, C=0, mf=h2_mf_fc/2, fluid_type='ParaHydrogen')
        pipe_h2_9_10 = Pipe(0.85, diam_est*ratio_h2_to_fc, h2_9_copy)  
        m_pipe_h2_910.append( 2 * pipe_h2_9_10.mass())
        h2_10 = pipe_h2_9_10.analyze_heat_pipe("H2_10")

        # FC Circulation 
        fc_press_drop_h2 = 0.06e5
        h2_22 = Fluid(name="H2_22", T=T_fc, P=p_fc - fc_press_drop_h2 , C=0, mf=h2_mf_rec, fluid_type='ParaHydrogen')  
        # Pipe h2 22-23
        pipe_h2_22_23 = Pipe(0.57, diam_est, h2_22)  
        m_pipe_h2_2223.append(pipe_h2_22_23.mass() * 2)
        h2_23 = pipe_h2_22_23.analyze_heat_pipe("H2_23")

        # Compressor 23-24
        if h2_23.P < p_fc:
            comp_23_24_PI = p_fc / h2_23.P
            comp_23_24 = Compressor(comp_efficiency=0.9, pressure_ratio=comp_23_24_PI, fluid=h2_23)
            comp23_24_power = comp_23_24.power()
            comp_23_24_mass = comp_23_24.mass(comp23_24_power)
        else:
            comp23_24_power = 0
            comp_23_24_mass = 0
        
        p_comp_2324.append(comp23_24_power)
        m_comp_2324.append( comp_23_24_mass * 2)
        h2_24 = Fluid(name="H2_24", T=h2_23.T, P=h2_23.P * comp_23_24_PI, C=0, mf=h2_mf_fc/2, fluid_type='ParaHydrogen')

        # Pipe h2 24-25
        pipe_h2_24_25 = Pipe(0.57, diam_est * ratio_h2_to_fc * 0.5, h2_24)  
        m_pipe_h2_2425.append(pipe_h2_24_25.mass() * 2)
        h2_25 = pipe_h2_24_25.analyze_heat_pipe("H2_25")

        # FC Cooling ----------------------------------------

        # D_cool_2_3 = np.sqrt(4 * cool_2.mf_given / (np.pi * cool_2.rho))
        # print(f"Diameter of pipe 2-3: {D_cool_2_3:.3f} m")
        
        # Pipe cool 2-3
        pipe_cool_2_3 = Pipe(4.96, diam_est_cool, cool_2)  
        m_pipe_cool_2_3.append(pipe_cool_2_3.mass())
        cool_3 = pipe_cool_2_3.analyze_heat_pipe("Cool_3")

        # Intersection 3-4A-4B
        cool_4 = Fluid_Coolant(name="Cool_4", T=cool_3.T, P=cool_3.P, C=0, mf=cool_3.mf_given/2, fluid_type=coolant) 

        # Pipe cool 4-5
        pipe_cool_4_5 = Pipe(1.56, diam_est_cool, cool_4)
        m_pipe_cool_4_5.append( pipe_cool_4_5.mass()*2 )
        cool_5 = pipe_cool_4_5.analyze_heat_pipe("Cool_5")

        # Pipe 19-22
        pipe_cool_19_22 = Pipe(4.96, diam_est_cool, cool_19, type ='inv')  
        m_pipe_cool_1922.append(pipe_cool_19_22.mass())
        cool_22 = pipe_cool_19_22.analyze_heat_pipe("Cool_22")

        # Intersection 22-21A-21B
        cool_21 = Fluid_Coolant(name="Cool_21", T=cool_22.T, P=cool_22.P, C=0, mf=cool_22.mf_given/2, fluid_type=coolant)

        # Pipe 21-20
        pipe_cool_21_20 = Pipe(2.83, diam_est_cool, cool_21, type='inv') 
        m_pipe_cool_2120.append(pipe_cool_21_20.mass() * 2)
        cool_20 = pipe_cool_21_20.analyze_heat_pipe("Cool_20")

        # Pipe 12-13
        cool_12 = Fluid_Coolant(name="Cool_12", T=T_fc, P=p_cool, C=0, mf=cool_mf_per_fc, fluid_type=coolant)  # Coolant to FC
        pipe_cool_12_13 = Pipe(1.42, diam_est_cool, cool_12)  
        m_pipe_cool_12_13.append(pipe_cool_12_13.mass() * 2)
        cool_13 = pipe_cool_12_13.analyze_heat_pipe("Cool_13")

        # Valve 13-20-14
        valve_132014 = Valve(fluid=cool_13, valve_efficiency=0.9)  # Valve for coolant to FC
        valve_132014_mass = valve_132014.valve_mass()  # Mass of the valve
        m_valve_132014.append(valve_132014_mass * 2)
        cool_14 = Fluid_Coolant(name="Cool_14", T=cool_13.T, P=cool_13.P, C=None, mf=cool_13.mf_given - cool_20.mf_given, fluid_type=coolant)  # Coolant to FC
        
        # Pipe 14-15
        ratio_to_wing = (cool_14.mf_given / (2 * cool_mf_per_fc))
        pipe_cool_14_15 = Pipe(0.71, diam_est_cool, cool_14)  
        m_pipe_cool_1415.append(pipe_cool_14_15.mass() * 2)
        cool_15 = pipe_cool_14_15.analyze_heat_pipe("Cool_15")

        # Valve 15-16-17
        valve_151617 = Valve(fluid=cool_15, valve_efficiency=0.9)  # Valve for coolant to FC   
        valve_151617_mass = valve_151617.valve_mass()  # Mass of the valve
        m_valve_151617.append(valve_151617_mass * 2)
        
        # Calculate bypass flow rate:
        T_not_HEX = (cool_12.mf_given * (cool_12.T - deltaT_fc) - cool_20.mf_given * cool_5.T) / (cool_12.mf_given - cool_20.mf_given)  # Temperature of coolant to FC without HEX
        cool_17_mf_minimum = Q_dot_rem / (cool_15.cp * (cool_15.T - T_amb)) 
        cool_mf_bypass = (T_not_HEX * cool_14.mf_given - cool_17_mf_minimum * T_amb) / cool_15.T
        cool_17_mf = np.maximum(cool_17_mf_minimum, cool_15.mf_given - cool_mf_bypass)  # Minimum mass flow rate of coolant to FC
        cool_16 = Fluid_Coolant(name="Cool_16", T=cool_15.T, P=cool_15.P, C=0, mf=cool_15.mf_given - cool_17_mf, fluid_type=coolant)  # Coolant to FC


        if cool_17_mf > 0.0001:
            cool_17 = Fluid_Coolant(name="Cool_17", T=cool_15.T, P=cool_15.P, C=0, mf=cool_17_mf, fluid_type=coolant)  # Coolant to FC
            # Pipe 17-18
            pipe_cool_17_18 = Pipe(0.43, diam_est_cool, cool_17)   
            m_pipe_cool_1718.append(pipe_cool_17_18.mass() * 2)
            cool_18 = pipe_cool_17_18.analyze_heat_pipe("Cool_18")


            # Skin Heat Exchanger
            # -------------------------------------------------------------------------------------------------
            Pr_coolant = cool_18.mu * cool_18.cp / cool_18.k
            Re_coolant = cool_18.mf_calculated * 4/  (np.pi * diam_est_cool * ratio_to_wing * cool_18.mu)
            f_coolant = (0.79 * np.log(Re_coolant) - 1.64) ** -2 
            Nu_coolant = ((f_coolant/8)*(Re_coolant-1000)*Pr_coolant)/(1+12.7*(f_coolant/8)**0.5 * (Pr_coolant**(2/3)-1))  # Nusselt number for coolant
            h_int = cool_18.k * Nu_coolant / (diam_est_cool * ratio_to_wing)
            U_cool = 1 / (1/h_int + 1/h_ext_w)  
            skin_hx = SkinHeatExchanger(area = area_wing*2, fluid = cool_18, U = U_cool )
            skin_hx.set_ambient(T_ambient_K = ambient_conditions['T'])
            Q_abs, t_cool_out = skin_hx.absorb_heat(Q_dot_rem)
            skin_hx_mass = area_wing * 2 * 7900 * 1e-3
            m_skin_hx.append(skin_hx_mass * 2)
            cool_23 = Fluid_Coolant(name="Cool_23", T=t_cool_out, P=cool_18.P, C=0, mf=cool_18.mf_given, fluid_type=coolant)  
            Q_dot_rem -= Q_abs  

            # Pipe cool 23
            pipe_cool_23 = Pipe(0.43, diam_est_cool, cool_23)
            m_pipe_cool_23.append(pipe_cool_23.mass() * 2)
            cool_23_out = pipe_cool_23.analyze_heat_pipe("Cool_23_prime")
            if Q_dot_rem > 0:
                Rad_1 = RamAirHeatExchanger(coolant_temp_K=ra_coolant_temp, fluid=cool_23_out, ambient_conditions=ambient_conditions,p_amb = p_amb)
                tol = 0.1
                rad_area_0 = 0
                radiator_area = 1
                rad_exists = False
                while abs(radiator_area - rad_area_0)>tol:
                    if rad_exists:
                        rad_area_0 = radiator_area
                    radiator_area, cool_24, power_fan,eta_p,net_drag,length_rad,A_0,mf_air = Rad_1.size_exchanger(Q_dot_rem/2, T_amb)
                    Rad_1.TR = Rad_1.thrust_ratio(
                        gamma = k_air,
                        R     = R_AIR,
                        M0    = ambient_conditions["M"],
                        T0    = ambient_conditions["T"],
                        eta_p07 = eta_p,  
                        CD_d_CD_sp = 0.11)  
                    rad_exists = True
                A_in = A_0 * (3.28084**2)
                st_pressure = p_amb * (0.0001450377)
                l_inlet = length_rad * (3.28084)
                radiator_mass = 0.32 * 0.4535924 * l_inlet * A_in ** 0.65 * st_pressure**0.6 + 1.735*(l_inlet * A_in**0.5 * st_pressure * 1.33)**0.7331
                m_rads.append(radiator_mass * 2)
                fan_mass = 2.959 * (power_fan/1000)**0.707
                m_fans_rad.append(fan_mass * 2)
            else:
                # No need for radiator, all waste heat is absorbed
                cool_24 = Fluid_Coolant(name="Cool_24", T=cool_23_out.T, P=cool_23_out.P, C=0, mf=cool_23_out.mf_given, fluid_type=coolant)
                power_fan = 0
                net_drag = 0

            drags.append(net_drag)
            fan_powers.append(power_fan*2)

            # Pipe 24
            pipe_cool_24 = Pipe(0.43, diam_est_cool, cool_24)
            m_pipe_cool_24.append(pipe_cool_24.mass() * 2)
            cool_24_out = pipe_cool_24.analyze_heat_pipe("Cool_24_Out")

            # Pipe 16-24
            pipe_cool_16_24 = Pipe(1.28, diam_est_cool, cool_16)
            m_pipe_cool_16_24.append(pipe_cool_16_24.mass() * 2)
            cool_24_prime = pipe_cool_16_24.analyze_heat_pipe("Cool_24_prime")

            cool_24_new = Fluid_Coolant(name="Cool_24", 
                            T=(cool_24_prime.T * cool_24_prime.mf_given + cool_24_out.T * cool_24_out.mf_given) / (cool_24_out.mf_given + cool_24_prime.mf_given), 
                            P=(cool_24_prime.P * cool_24_prime.mf_given + cool_24_out.P * cool_24_out.mf_given) / (cool_24_out.mf_given + cool_24_prime.mf_given), 
                            C=0, mf=cool_24_out.mf_given + cool_24_prime.mf_given, fluid_type=coolant)  
        else:
            # Pipe 16-24
            pipe_cool_16_24 = Pipe(1.28, diam_est_cool, cool_16)
            m_pipe_cool_16_24.append(pipe_cool_16_24.mass() * 2)
            cool_24_prime = pipe_cool_16_24.analyze_heat_pipe("Cool_24_prime")

            cool_24_new = Fluid_Coolant(name="Cool_24_new", T = cool_24_prime.T,
                            P = cool_24_prime.P, C=0, mf=cool_24_prime.mf_given, fluid_type=coolant)
        
        # Intersection 24-5-25
        cool_25 = Fluid_Coolant(name="Cool_25", 
                        T=(cool_24_new.T * cool_24_new.mf_given + cool_5.T * cool_5.mf_given) / (cool_5.mf_given + cool_24_new.mf_given), 
                        P=cool_24_new.P, C=0, mf=cool_24_new.mf_given + cool_5.mf_given, fluid_type=coolant) 
        
        #Pipe Cool 25
        pipe_cool_25 = Pipe(0.28, diam_est_cool, cool_25)
        m_pipe_cool_25.append(pipe_cool_25.mass() * 2)
        cool_25_out = pipe_cool_25.analyze_heat_pipe("Cool_25_out")

        # Electric heater:
        if cool_25_out.T <= T_fc - deltaT_fc:
            Q_dot_heater = (cool_25_out.mf_given * cool_25_out.cp * (T_fc - deltaT_fc - cool_25_out.T))  # Heat required to heat the coolant to FC temperature
            Q_heater[i] += Q_dot_heater
        else:
            Q_dot_heater = 0
        cool_26 = Fluid_Coolant(name="Cool_26", T= cool_25_out.T + Q_dot_heater / (cool_25_out.cp * cool_25_out.mf_given), P=cool_25_out.P, C=0, mf=cool_25_out.mf_given, fluid_type=coolant) 

        # Pipe 26
        pipe_cool_26 = Pipe(0.28, diam_est_cool, cool_26)
        m_pipe_cool_26.append(pipe_cool_26.mass() * 2)
        cool_26_out = pipe_cool_26.analyze_heat_pipe("Cool_26_out")

        # Coolant pump
        pump_26_27 = Pump(fluid=cool_26_out, pump_eff=0.9, delta_pressure=p_cool - cool_26_out.P)  # Pump for coolant to FC
        pump_26_27_power = pump_26_27.power()
        pump_26_27_power_list.append(pump_26_27_power)
        pump_26_27_mass = 0.0138289 * pump_26_27_power - 0.366129
        m_pump_2627.append(abs(pump_26_27_mass) * 2)
        cool_27 = Fluid_Coolant(name="Cool_27", T=cool_26_out.T, P=p_cool, C=0, mf=cool_26_out.mf_given, fluid_type=coolant)  # Coolant to FC after pump

        # Pipe 27
        pipe_cool_27 = Pipe(0.43, diam_est_cool, cool_27)
        m_pipe_cool_27.append(pipe_cool_27.mass() * 2)
        cool_27_out = pipe_cool_27.analyze_heat_pipe("Cool_27_out")

        # Air Loop ----------------------------------------
        p_stag = p_amb 
        if ambient_conditions['V'] < 100:
            fan1 = Fan(fan_eff=fan_eff, ra_mf=air_mf_fc, ra_density=ambient_conditions['rho'], delta_pressure=delta_pressure)
            power_fan1 = fan1.power()
            fan_mass = 2.959 * (power_fan1/1000)**0.707
            m_front += fan_mass
        else:
            power_fan1 = 0
            fan_mass = 0
        
        fan_powers_1.append(power_fan1)
        m_fans_1.append(fan_mass)


        air_1 = Fluid(name="Air_1", T=T_amb, P=p_stag, C=0, mf=air_mf_fc, fluid_type='Air')  
        Q_dot_heater = (air_mf_fc * cp_air * (T_fc - T_amb))
        Q_heater[i] += Q_dot_heater
        air_2 = Fluid(name="Air_2", T=T_fc, P=p_stag, C=0, mf=air_mf_fc, fluid_type='Air') 

        compressor_air = Compressor(comp_efficiency=0.9, pressure_ratio=p_fc / p_stag * 1.2, fluid=air_2)
        comp_air_power = compressor_air.power()
        air_comp_power_list.append(comp_air_power)
        comp_air_mass = compressor_air.mass(comp_air_power)
        air_comp_mass_list.append(comp_air_mass)

        # Turbine Air
        air_fc_exit = Fluid(name="Air_FC_Exit", T=T_fc, P=p_fc, C=0, mf=air_out_fc, fluid_type='Air')
        turbine_air = Turbine(turbine_efficiency=0.9, fluid=air_fc_exit)
        turbine_air_power = turbine_air.power(T_in=air_fc_exit.T,T_out=80+273)
        air_turb_p_list.append(turbine_air_power)
        turbine_air_mass = turbine_air.mass(turbine_air_power)
        air_turbine_mass_list.append(turbine_air_mass)
        air_3 = Fluid(name="Air_3", T=80+273, P=air_fc_exit.P, C=0, mf=air_fc_exit.mf_given, fluid_type='Air')  

        # Turbine Water
        water_fc_exit = Fluid(name="Water_FC_Exit", T=T_fc, P=p_fc, C=0, mf=h2o_mf_fc, fluid_type='Water')  # Water from FC
        turbine_water = Turbine(turbine_efficiency=0.9, fluid=water_fc_exit)
        turbine_water_power = turbine_water.power(T_in=water_fc_exit.T,T_out=80+273)
        water_turb_p_list.append(turbine_water_power)
        turbine_water_mass = turbine_water.mass(turbine_water_power)
        water_turbine_mass_list.append(turbine_water_mass)
        water_4 = Fluid(name="Water_4", T=80+273, P=water_fc_exit.P, C=0, mf=water_fc_exit.mf_given, fluid_type='Water')

        # Water - Air Separator
        tot_mf = air_3.mf_given + water_4.mf_given
        mass_was = tot_mf * 5.33
        m_was_list.append(mass_was)
        volume_was = tot_mf * 0.02635586797

        water_5 = Fluid(name="Water_5", T=water_4.T, P=water_4.P-6000, C=0, mf=water_4.mf_given * 0.85, fluid_type='Water')  

        # Pump Water
        p_water_in = 13.5 * 101325
        if water_5.P < p_water_in:
            pump_water = Pump(fluid=water_5, pump_eff=0.9, delta_pressure=p_water_in-water_5.P) 
            pump_water_power = pump_water.power() 
            pump_water_mass = 0.0138289 * pump_water_power - 0.366129
            m_front += pump_water_mass
        else:
            pump_water_power = 0
            pump_water_mass = 0
        
        pump_water_power_list.append(pump_water_power)
        m_pump_water.append(pump_water_mass * 2)
        water_6 = Fluid(name="Water_6", T=water_5.T, P=p_water_in, C=0, mf=water_5.mf_given, fluid_type='Water')  
        valve_water = Valve(fluid=water_6, valve_efficiency=0.9)
        valve_water_mass = valve_water.valve_mass()  
        m_valve_water.append(valve_water_mass)

        # Store all Fluid instances in a dictionary with their name as key
        fluids_dict = {}
        for var_name, var_value in locals().items():
            if isinstance(var_value, Fluid):
                fluids_dict[var_value.name] = {
                    'massflow': var_value.mf_given,
                    'temperature': var_value.T,
                    'pressure': var_value.P
                }
        for var_name, var_value in locals().items():
            if isinstance(var_value, Fluid_Coolant):
                fluids_dict[var_value.name] = {
                    'massflow': var_value.mf_given,
                    'temperature': var_value.T,
                    'pressure': var_value.P
                }

    heater_eff = 0.9
    #heater_power = Q_heater.max() / heater_eff 
    heater_power = 0 
    # # # # # # # print(m_fans_rad)
    power_required = -np.max(water_turb_p_list) + -np.max(air_turb_p_list) + np.max(fan_powers) + np.max(pump_26_27_power_list) + np.max(air_comp_power_list) + np.max(fan_powers_1) + np.max(pump_water_power_list) + np.max(p_comp_2324) + np.max(p_comp_1415) + heater_power
    m_front = np.max(m_valve_6711) + np.max(m_pipe_h2_1112) + np.max(m_valve_121326) + np.max(m_pipe_h2_1314) + np.max(m_pipe_h2_2618) + np.max(m_comp_1415) + np.max(m_pipe_h2_1516) + np.max(m_pipe_h2_1719) + np.max(m_valve_1920) + np.max(m_pipe_2021) + np.max(m_pipe_h2_78) + np.max(m_pipe_valve89) + np.max(m_pipe_h2_910) + np.max(m_pipe_h2_2223) + np.max(m_comp_2324) + np.max(m_pipe_h2_2425) + np.max(m_pipe_cool_4_5) + np.max(m_pipe_cool_2120) + np.max(m_pipe_cool_12_13) \
    + np.max(m_valve_132014) + np.max(m_pipe_cool_1415) + np.max(m_valve_151617) + np.max(m_pipe_cool_1718) + np.max(m_skin_hx) + np.max(m_pipe_cool_23) + np.max(m_fans_rad) + np.max(m_rads) + np.max(m_pipe_cool_24) + np.max(m_pipe_cool_16_24) + np.max(m_pipe_cool_25) + np.max(m_pipe_cool_26) + np.max(m_pump_2627) + np.max(m_pipe_cool_27) + np.max(m_fans_1) + np.max(air_comp_mass_list) + np.max(air_turbine_mass_list) \
    + np.max(water_turbine_mass_list) + np.max(m_was_list) + np.max(m_pump_water) + np.max(m_valve_water)
    
    # # # # # # print("m_fan_power:", fan_powers)
    # # # # # # print("m_fans_rad:", np.max(m_fans_rad))
    
    
    m_rear = np.max(m_pipe_h2_12) + np.max(m_pipe_h2_34) + np.max(m_sh) + np.max(m_pipe_cool_201) + np.max(m_vap)
    m_mid = np.max(m_pipe_h2_56) + np.max(m_pipe_cool_2_3) + np.max(m_pipe_cool_1922)

    output_list = [
    np.max(drags), 
    power_required,
    m_front, 
    m_mid, 
    m_rear]

    # print(f"Front TMS mass: {m_front:.2f} kg")
    # print(f"Mid TMS mass: {m_mid:.2f} kg")
    # print(f"Rear TMS mass: {m_rear:.2f} kg")

    # Print the diameter for every Pipe instance
    for var_name, var_value in locals().items():
        if isinstance(var_value, Pipe):
            if var_name == 'pipe_cool_2_3' or var_name == 'pipe_cool_19_22':
                pass
                # print(f"Pipe '{var_name}' diameter: {var_value.d_in:.4f} m")

    # print(f"Long Coolant pipe mass: {np.max(m_pipe_cool_1922):.2f} kg")
    # print(f"Long Coolant return pipe mass: {np.max(m_pipe_cool_2_3):.2f} kg")
    # print(f"Long H2 pipe mass: {np.max(m_pipe_h2_56):.2f} kg")
    
    


    return fluids_dict , output_list

# RUN -----------------------------------------------------------
if __name__ == "__main__":

    prop_counter = 0
    @lru_cache(maxsize=128)
    def PropsSI(*args):
        global prop_counter
        prop_counter += 1
        return call_propssi(*args)
    # Dummy values for main function inputs
    Q_dot_fc = 500000         # Waste heat from fuel cell [W]
    Q_dot_eps = 20000        # Waste heat from EPS [W]
    p_fc = 1.8e5               # Fuel cell pressure [Pa]
    p_cc = 12e5             # Combustion chamber pressure [Pa]
    h2_mf_fc = 0.01          # H2 mass flow to FC [kg/s]
    h2_mf_cc = 0.02944          # H2 mass flow to CC [kg/s]
    T_fc = 273.15+160            # Fuel cell temperature [K]
    T_cc = T_fc           # Combustion chamber temperature [K]
    air_mf_fc = 45*h2_mf_fc          # Air mass flow to FC [kg/s]
    T_amb = 273.15+15           # Ambient temperature [K]
    rho_amb = 1.225          # Ambient air density [kg/m^3]
    V_amb = 120.0            # Ambient velocity [m/s]
    p_amb = 101325           # Ambient pressure [Pa]
    h2_mf_rec = (0.1 * h2_mf_fc)/2   # H2 recirculation mass flow [kg/s]
    air_out_fc = 45*h2_mf_fc         # Air out of FC [kg/s]
    p_sto = 7e5              # Storage pressure [Pa]
    h2o_mf_fc = 0.05

    
    fluids, outputs = tms_main(Q_dot_fc, Q_dot_eps, p_fc, p_cc, h2_mf_fc, h2_mf_cc, T_fc, T_cc, air_mf_fc, T_amb, rho_amb, V_amb, p_amb, h2_mf_rec, air_out_fc, p_sto, h2o_mf_fc)
    

    tms_plotting.csv_fluids(fluids)
    
    # tms_plotting.plot(pdf_path = 'Diagrams\H2D2 P&ID.pdf', fluids = fluids, positions = positions, page_number=0, dpi=300)
