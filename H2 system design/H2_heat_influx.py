import CoolProp.CoolProp as CP # Used to get properties of Hydrogen
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
from scipy.optimize import fsolve
import csv

class Tank:
    def __init__(self, MAWP, material, material2, mat_property,mass_h2,mat2_property,fill,V_in,p_vent):
        # ---------------------------------------------Database ------------------------------------------------------------------
        self.material = material
        self.material2 = material2
        self.mat_property = mat_property
        self.mat_density = mat_property[0] 
        self.mat_yield_strength = mat_property[1]
        self.mat_thermal_conductivity = mat_property[2]
        self.mat_emissivity = mat_property[3]
        self.mat_co2 = mat_property[4]
        self.mat_ee = mat_property[5]
        self.mat_fibre_ratio = mat_property[6]

        self.mat2_property = mat2_property
        self.mat2_density = mat2_property[0]
        self.mat2_yield_strength = mat2_property[1]
        self.mat2_thermal_conductivity = mat2_property[2]
        self.mat2_emissivity = mat2_property[3]
        self.mat2_co2 = mat2_property[4]
        self.mat2_ee = mat2_property[5]
        self.mat2_fibre_ratio = mat2_property[6]

        # ---------------------------------------------Constants ----------------------------------------------------------------
        self.dormancy = 24 #hours
        self.fill_ratio = fill #percentage of LH2 at the start
        self.mass_h2 = mass_h2 #kg
        self.stratification_factor = 2
        self.k_str = 1.9 #[W/mK]


        # --------------------------------------------- Inputs ------------------------------------------------------------------

        self.MAWP = MAWP # Convert from Bar to Pa for the venting pressure
        self.Pvent = p_vent #Assumption

        # Hydrogen properties
        self.P0 = 101000 # Initial pressure in Pa
        self.T0 = CP.PropsSI('T', 'P', self.P0, 'Q', 0.1, 'ParaHydrogen') # Initial temperature in K
        self.rhol0 = CP.PropsSI('D', 'T', self.T0, 'Q', 0, 'ParaHydrogen') # density of liquid hydrogen
        self.rhog0 = CP.PropsSI('D', 'T', self.T0, 'Q', 1, 'ParaHydrogen') # density of gas hydrogen
        self.V_in = V_in
        self.mg0 = CP.PropsSI('D', 'T', self.T0, 'Q', 1, 'ParaHydrogen')*self.V_in*(1-self.fill_ratio) # kg
        self.ml0 = CP.PropsSI('D', 'T', self.T0, 'Q', 0, 'ParaHydrogen')*self.V_in*(self.fill_ratio) # kg
        self.hg0 = CP.PropsSI('H', 'T', self.T0, 'Q', 1, 'ParaHydrogen')#kJ/kg
        self.hl0 = CP.PropsSI('H', 'T', self.T0, 'Q', 0, 'ParaHydrogen')#kJ/kg
        self.x0 = self.mg0/(self.mg0+self.ml0) #unitless (vapor quality)

        # ---------------------------------------------- Constraint ---------------------------------------------------
        self.R_in = 0.8 #m
        self.Q_leak_min = 10 #W (determined from Nicolas's thesis)
        self.Q_leak_max = 1000 #W (determined from Nicolas's thesis)
        

    # ----------------------------------------------- Maximum Heat Load ---------------------------------------------------
    def maximum_Qin(self,Qleak):
        P = [self.P0]
        T = [self.T0]
        mg = [self.mg0]
        ml = [self.ml0]
        rhog = [self.rhog0]
        rhol = [self.rhol0]
        hg = [self.hg0]
        hl = [self.hl0]
        x = [self.x0]
        Ps = [self.P0]
        time = [0]
        dt = 250 # Time step in seconds
        # Run the simulation until the pressure reaches the venting pressure
        while P[-1]<self.Pvent:
            # Compute derivatives
            delrhog_delP = CP.PropsSI('d(D)/d(P)|T', 'T', T[-1], 'Q', 1, 'ParaHydrogen')
            delrhol_delP = CP.PropsSI('d(D)/d(P)|T', 'T', T[-1], 'Q', 0, 'ParaHydrogen')
            delrhog_delT = CP.PropsSI('d(D)/d(T)|P', 'P', P[-1], 'Q', 1, 'ParaHydrogen')
            delrhol_delT = CP.PropsSI('d(D)/d(T)|P', 'P', P[-1], 'Q', 0, 'ParaHydrogen')
            delhg_delP = CP.PropsSI('d(H)/d(P)|T', 'T', T[-1], 'Q', 1, 'ParaHydrogen')
            delhl_delP = CP.PropsSI('d(H)/d(P)|T', 'T', T[-1], 'Q', 0, 'ParaHydrogen')
            delhg_delT = CP.PropsSI('d(H)/d(T)|P', 'P', P[-1], 'Q', 1, 'ParaHydrogen')
            delhl_delT = CP.PropsSI('d(H)/d(T)|P', 'P', P[-1], 'Q', 0, 'ParaHydrogen')
            # Small temperature increment to calculate the rate of pressure change with temperature
            T_increment = 6.5e-06
            T_new = T[-1]+T_increment
            P_new = CP.PropsSI('P', 'T', T_new, 'Q', 0.1, 'ParaHydrogen')
            dPs_dT = (P_new-P[-1])/(T_new-T[-1])
            # Construct the coefficient matrix A
            A21 = -(((mg[-1]/(rhog[-1]**2))*delrhog_delP)+
            ((ml[-1]/(rhol[-1]**2))*delrhol_delP))
            A22 = -(((mg[-1]/(rhog[-1]**2))*delrhog_delT)+
            ((ml[-1]/(rhol[-1]**2))*delrhol_delT))
            A42 = (ml[-1]*delhl_delT)+(mg[-1]*delhg_delT)\
            +(((ml[-1]*delhl_delP)+(mg[-1]*delhg_delP)-self.V_in)*dPs_dT)
            A = np.array([[1, -dPs_dT, 0, 0],
            [A21, A22, (1/rhog[-1])-(1/rhol[-1]), 0],
            [0, 0, 1, 1],
            [0, A42, -(hl[-1]-hg[-1]), 0]])
            # Construct the B matrix
            B = np.array([[0],
            [0],
            [0],
            [Qleak]])
            # Solve the system of linear equations
            differential_matrix = np.linalg.solve(A, B)
            dP_dt = differential_matrix[0, 0]*self.stratification_factor
            dT_dt = differential_matrix[1, 0]
            dmg_dt = differential_matrix[2, 0]
            dml_dt = differential_matrix[3, 0]
            # Update the state variables
            P.append(P[-1]+(dt*dP_dt))
            T.append(CP.PropsSI('T', 'P', P[-1], 'Q', 0.1, 'ParaHydrogen'))
            if mg[-1]+(dt*dmg_dt)<=0:
                mg.append(0)
            else:
                mg.append(mg[-1]+(dt*dmg_dt))
                ml.append(ml[-1]+(dt*dml_dt))
                x.append(mg[-1]/(mg[-1]+ml[-1]))
                rhog.append(CP.PropsSI('D', 'P', P[-1], 'Q', 1, 'ParaHydrogen'))
                rhol.append(CP.PropsSI('D', 'P', P[-1], 'Q', 0, 'ParaHydrogen'))
                hg.append(CP.PropsSI('H', 'P', P[-1], 'Q', 1, 'ParaHydrogen'))
                hl.append(CP.PropsSI('H', 'P', P[-1], 'Q', 0, 'ParaHydrogen'))
                Ps.append(CP.PropsSI('P', 'T', T[-1], 'Q', 0.5, 'ParaHydrogen'))
                time.append(time[-1]+dt)
        # Convert time to hours and pressure to bars for plotting
        time_hours = np.array(time)/3600
        P_bars = np.array(P)/100000 # Convert pressure from Pa to bars
        return time_hours[-1] - self.dormancy

    # ----------------------------------------------- Inner tank Parameters ---------------------------------------------------
    def volume_equation(self,l):
        L_cyl = l - 2*self.R_in  # cylindrical length
        if L_cyl < 0:
            return 1e6  # Penalize invalid cases
        V = np.pi * self.R_in**2 * L_cyl + (4/3) * np.pi * self.R_in**3  # cylinder + 2 hemispheres = 1 sphere
        return V - self.V_in  # we want this to be zero

    
    def inner_tank_thickness(self):
        P_test = (self.MAWP + 0.2*100000) * 1.3 #Safety factor
        alpha = 0
        if self.material == 'G10' or self.material == 'Carbon Fibre Woven Prepreg (QI)':
            #Calc membrane loads
            N_theta_p = P_test * self.R_in
            N_phi_p = P_test * self.R_in / 2
            self.mat_property[1] = self.mat_property[1] #may add safety factors
            angle = [0,90,0.1] #helical winding angle in degrees
            t_min = 1000
            for ang in angle:
                #Calc netting thickness in hoop and helical direction
                t_heli = N_phi_p / (self.mat_property[1] * np.cos(ang)**2)
                t_hoop = (N_theta_p - N_phi_p * np.tan(ang)**2) / (self.mat_property[1])
                #Calc minimum thickness based on FVF
                t = (t_heli + t_hoop) / self.mat_property[6]
                if t < t_min:
                    t_min = t
                    alpha = ang
            return t_min, alpha
        else:
            return P_test * self.R_in / self.mat_property[1], alpha

    # ------------------------------------------------ Vacuum and Outer tank  ---------------------------------------------------

    def heat_influx(self, L_in_max, Q_str,t1,emis_mli,k_vac,t_mli, k_mli,Qmax):
        T_tank = self.T0
        print(f"Initial tank temperature: {T_tank} K")
        T_amb = 300 # K
        P_amb = 101325 # Pa
        r_in = self.R_in
        t2 = t1
        Q_in_max = Qmax
        k_1 = self.mat_property[2]
        k_2 = self.mat2_property[2]
        eps2 = self.mat2_property[3]

        # Conduction...
        def Q_cond(dv):
            # Conduction resistance
            R_cond = 0.0
            R_cond = np.log((r_in + t1) / r_in) / (2 * np.pi * L_in_max * k_1)
            R_cond += np.log((r_in + t1 + t_mli) / (r_in + t1)) / (2 * np.pi * L_in_max * k_mli)
            R_cond += np.log((r_in + t1 + t_mli + dv) / (r_in + t1 + t_mli)) / (2 * np.pi * L_in_max * k_vac)
            R_cond += np.log((r_in + t1 + dv + t_mli + t2) / (r_in + t1 + dv + t_mli)) / (2 * np.pi * L_in_max * k_2)
            return (T_amb - T_tank) / R_cond

        # Radiation...
        def Q_rad(dv):
            r1 = r_in + t1 + t_mli        # Outer surface of inner tank
            r2 = r1 + dv          # Inner surface of outer tank

            # Surface areas (cylinder + two hemispherical caps)
            A1 = 2 * np.pi * r1 * (L_in_max + 2 * r1)
            A2 = 2 * np.pi * r2 * (L_in_max + 2 * r2)

            # Radiation heat transfer
            
            #denom = (1 / (eps2)) + (A2 / A1) * (1 / emis_mli - 1)
            #return 5.670374419e-8 * A2 * (T_amb**4 - T_tank**4) / denom
            num = 5.670374419e-8 * (T_amb**4 - T_tank**4)
            denom = 1/emis_mli + (1/eps2) - 1
            return num * A2 / denom

        # Total heat influx...
        def total_heat_influx(dv):
            Q_cond_value = Q_cond(dv)
            Q_rad_value = Q_rad(dv)
            return Q_cond_value + Q_rad_value + Q_str

        # Optimization eq...
        def equation(dv):
            return total_heat_influx(dv) - Q_in_max

        # Initial guess for dv
        dv_initial_guess = 0.01  # Initial guess for the vacuum gap thickness (m)

        # Solve for dv...
        dv_solution = fsolve(equation, dv_initial_guess)

        dv = dv_solution[0]  # Extract the solution from the array

        t2_min = 1000
        P_test = P_amb #Safety factor
        alpha = 0
        if self.material2 == 'G10' or self.material2 == 'Carbon Fibre Woven Prepreg (QI)':
            #Calc membrane loads
            N_theta_p = P_test * (r_in+t1+dv+t_mli)
            N_phi_p = P_test * (r_in+t1+dv+t_mli) / 2
            self.mat2_property[1] = self.mat2_property[1] #may add safety factors
            angle = [0,90,0.1] #helical winding angle in degrees
            
            for ang in angle:
                #Calc netting thickness in hoop and helical direction
                t_heli = N_phi_p / (self.mat2_property[1] * np.cos(ang)**2)
                t_hoop = (N_theta_p - N_phi_p * np.tan(ang)**2) / (self.mat2_property[1])
                #Calc minimum thickness based on FVF
                t = (t_heli + t_hoop) / self.mat2_property[6]
                if t < t2_min:
                    t2_min = t
                    alpha = ang
        else:
            t2_min = P_amb * (r_in+t1+dv+t_mli) / self.mat2_property[1]

        return dv, t2_min, alpha, Q_in_max, Q_cond(dv),  Q_rad(dv)

    # ------------------------------------------------ Tank Dimensions ---------------------------------------------------
    def total_volume(self, l,dv,t1,t2,t_mli):
        R_out = self.R_in + dv +t1 +t2 +t_mli
        V = np.pi * R_out**2 * (l-2*self.R_in) + (4/3) * np.pi * R_out**3
        return V
    
    def total_mass(self, l, dv, t1, t2,t_mli,dens_mli):
        R_out = self.R_in + dv +t1 +t2 +t_mli
        L_cyl = l - 2 * self.R_in  
        surface_inner = 2 * np.pi * self.R_in * L_cyl + 4 * np.pi * self.R_in**2
        surface_outer = 2 * np.pi * R_out * L_cyl + 4 * np.pi * R_out**2
        mass_inner = surface_inner * t1 * self.mat_density
        mass_outer = surface_outer * t2 * self.mat2_density
        mass_mli = surface_inner * t_mli * dens_mli
        return mass_inner,mass_outer,mass_mli

    def kg_co2(self,mass_inner,mass_outer,mass_mli,mass_str,mli_co2,co2_kevlar):
        lca_inner = mass_inner * self.mat_property[4]
        lca_outer = mass_outer * self.mat2_property[4]
        lca_mli = mass_mli * mli_co2
        lca_str = mass_str * co2_kevlar
        lca_total = lca_inner + lca_outer + lca_mli + lca_str
        return lca_total
    
    def embodied_energy(self,mass_inner,mass_outer,mass_mli,mass_str,kevlar_ee,mli_ee):
        ee_inner = mass_inner * self.mat_ee
        ee_outer = mass_outer * self.mat2_ee
        ee_mli = mass_mli * mli_ee
        ee_str = mass_str * kevlar_ee
        ee_total = ee_inner + ee_outer + ee_mli + ee_str
        return ee_total

# -------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------- Tank Database ---------------------------------------------------
materials = ['Al-7075-T6','G10','SS-304','Carbon Fibre Woven Prepreg (QI)','SS-316'] #From Granta and Engineering Toolbox
SF = 0.9
#density in kg/m^3, yield strength in Pa, thermal conductivity in W/mK, emissivity in [-], CO2 [kg/kg], Embodied Energy in MJ/kg, Fibre Ratio
mat_properties = [[2800,495*1e6*SF,134,0.11,7.795,106,0], 
                  [1900,298*1e6*SF,0.5,0.95,15.25,369,0.4],
                  [7955,257.5*1e6*SF,15.5,0.35,3,42.75,0],
                  [1575,549.5*1e6*SF,1.64,0.77,47.8,686.5,0.625],
                  [7970,257.5*1e6*SF,15,0.35,4.265,49.75,0]]
#MAWPS = [600000,650000,800000,1000000,1200000,1280000] #bar to Pa
MAWP = 1200000
P_vents = [600000,800000,1000000]

n_mats = 2
materials = materials[:n_mats]
mat_properties = mat_properties[:n_mats]

# ------------------------------------------------- Structure ------------------------------------------------------
# Thesis values
Q_og_str = 0.4 #W
k_kevlar = 1.9 #W/mK
og_str_mass = 2.1 #kg
og_lh2 = 6.2 #kg
grav_idx = 0.35 #from NASA report
co2_kevlar = 13.1 #kg/kg (Kevlar 149)
kevlar_ee = 257 #MJ/kg (Embodied Energy for Kevlar 149)

#Our values
mass_h2 = 532.65 #kg
estimated_mass = mass_h2/grav_idx - mass_h2
t_limit = 0.001 #m (minimum thickness of the tank wall)

#Insulation
t_mli = 10 *1e-3 #https://www.sciencedirect.com/science/article/pii/S135943112200391X
dens_mli = 180 #kg/m^3 https://www.sciencedirect.com/science/article/pii/S135943112200391X
emis_mli = 0.23  #https://www.thermalengineer.com/library/effective_emittance.htm
k_vac = 0.015*1e-1#3 # W/mK https://www.researchgate.net/publication/321219004_Cylindrical_Cryogenic_Calorimeter_Testing_of_Six_Types_of_Multilayer_Insulation_Systems
k_mli = 0.035 # W/mK  https://www.sciencedirect.com/science/article/pii/S135943112200391X
mli_co2 = 8.03 #kg/kg (for Aluminium 6463 T6)
mli_ee = 108.4 #MJ/kg (Embodied Energy for Aluminium 6463 T6)

def fA(mh2, P_vent, fl_final = 0.98):

    # From the heat influx get the maximum initial fill fraction
    rho_l_f = CP.PropsSI('D', 'P', P_vent, 'Q', 0, 'ParaHydrogen')  # Final liquid density
    print(f"Final liquid density: {rho_l_f} kg/m^3")
    V_l_fin = mh2 / rho_l_f  # Final liquid volume
    V_tot   = V_l_fin / fl_final

    # Calculate the initial fill fraction
    rho_l_0 = CP.PropsSI('D', 'P', 101325, 'Q', 0, 'ParaHydrogen')  # Initial liquid density
    print(f"Initial liquid density: {rho_l_0} kg/m^3")
    V_l_0   = mh2 / rho_l_0  # Initial liquid volume
    fl_init = V_l_0 / V_tot  # Initial fill fraction

    return V_tot, fl_init

def compute_Qleak(material, material2, mat_property, MAWP,mass_h2, Q_str,mat2_property,str_mass,fill_ratio,V_in,P_vent):
    tankh2 = Tank(MAWP, material, material2, mat_property,mass_h2,mat2_property,fill_ratio,V_in,P_vent)
    # -----------------------------Maximum allowable heat load -----------------------------------
    Q_solution = opt.root_scalar(tankh2.maximum_Qin, bracket=[tankh2.Q_leak_min, tankh2.Q_leak_max], method='brentq')

    if Q_solution.converged:
        Qmax = Q_solution.root
        print(f"Calculated Qin: {Qmax:.4f} W")
    else:
        print("Failed to find an Qin_max. Adjust the bounds.")
    return Qmax

def compute_tank_volume(material, material2, mat_property, MAWP,mass_h2, Q_str,mat2_property,str_mass,fill_ratio,V_in,P_vent,Qmax):
    tankh2 = Tank(MAWP, material, material2, mat_property,mass_h2,mat2_property,fill_ratio,V_in,P_vent)

    # -----------------------------Inner tank sizing -----------------------------------
    L_solution = opt.root_scalar(tankh2.volume_equation, bracket=[2*tankh2.R_in, 10], method='brentq')

    if L_solution.converged:
        L_in = L_solution.root
        print(f"Calculated length: {L_in:.4f} meters")
    else:
        print("Failed to find an inner length. Adjust the bounds.")

    t1, ang1_w = tankh2.inner_tank_thickness()
    if t1 < t_limit:
        t1 = t_limit

    dv, t2, ang2_w, Qleak, Qcond, Qrad = tankh2.heat_influx(L_in, Q_str,t1,emis_mli,k_vac,t_mli,k_mli,Qmax)
    t2 = max(t2,t_limit)
    
    Vt = tankh2.total_volume(L_in, dv,t1,t2,t_mli)
    mass_inner,mass_outer,mass_mli = tankh2.total_mass(L_in, dv, t1, t2,t_mli,dens_mli) 
    Mt = mass_inner + mass_outer + mass_mli + mass_h2
    Mt += str_mass
    mass_error = Mt - estimated_mass - mass_h2
    Vh2 = tankh2.V_in

    co2_kg = tankh2.kg_co2(mass_inner,mass_outer,mass_mli,str_mass,mli_co2,co2_kevlar)
    embodied_energy = tankh2.embodied_energy(mass_inner,mass_outer,mass_mli,str_mass,kevlar_ee,mli_ee)

    return Vt, Mt, mass_error,Vh2,t1,t2,dv,L_in, Vh2,tankh2.R_in, co2_kg, embodied_energy, ang1_w, ang2_w, P_vent,Qleak, Qcond, Qrad

# ------------------------------------------------- Main ------------------------------------------------------

RUN = False #Run new design
OPEN = True #Open previous design

plot_mats = []
plot_mats2 = []
plot_MAWPS = []
plot_volumes = []
plot_masses = []
plot_mass_errors = []
plot_co2 = []
plot_embodied_energy = []
plot_P_vents = []
plot_vh2 = []

if RUN:
    for P_vent in P_vents:
        V_in, fill_ratio = fA(mass_h2, P_vent)
        Qmax = compute_Qleak(materials[0], materials[0], mat_properties[0], MAWP,mass_h2,0,mat_properties[0],0,fill_ratio,V_in, P_vent)
        for material, mat_property in zip(materials, mat_properties):
            for material2, mat2_property in zip(materials, mat_properties):
                if material == 'G10' or  material == 'Carbon Fibre Woven Prepreg (QI)': #Composites
                    og_tank_mass = 4.8+3.15
                else: #Metallic materials (compared to aluminium)
                    og_tank_mass = 8.4+3.6
                ratio = og_str_mass / (og_tank_mass+og_lh2)
                str_mass = estimated_mass * ratio
                Q_str = Q_og_str * np.sqrt(ratio/ratio) #Assuming Volume increase with the same ratio as the mass
                Vt, Mt, mass_error,Vh2,t1,t2,dv,L_in,Vh2,R_in,co2_kg,emb_energy, ang1_w, ang2_w, p_vent, Qleak, Qcond, Qrad = compute_tank_volume(material, material2, mat_property, MAWP,mass_h2,Q_str,mat2_property,str_mass,fill_ratio,V_in, P_vent,Qmax)
                print(f"Material In: {material}, Material Out: {material2}, MAWP: {MAWP} Pa, P_vent:{P_vent}")
                print(f"Tank Volume: {Vt:.4f} m^3")
                print(f"Tank Mass: {Mt:.4f} kg")
                print(f"CO2 emissions: {co2_kg:.4f} kg")
                print(f"Embodied Energy: {emb_energy:.4f} MJ")

                plot_mats.append(material)
                plot_mats2.append(material2)
                plot_MAWPS.append(MAWP)
                plot_volumes.append(Vt)
                plot_masses.append(Mt)
                plot_mass_errors.append(mass_error)
                plot_co2.append(co2_kg)
                plot_embodied_energy.append(emb_energy)
                plot_P_vents.append(P_vent)
                plot_vh2.append(Vh2)

                # CSV file
                file_exists = False
                try:
                    with open('tank_results.csv', mode='r') as file:
                        file_exists = True
                except FileNotFoundError:
                    pass

                with open('tank_results.csv', mode='a', newline='') as file:
                    writer = csv.writer(file)
                    if not file_exists:
                        writer.writerow(['Material Inner', 'Material Outer', 'MAWP (Pa)', 'Tank Volume (m^3)', 
                                        'Tank Mass (kg)', 'Mass Error (kg)',
                                        'Inner Tank Thickness (m)', 'Outer Tank Thickness (m)', 
                                        'Vacuum Gap (m)', 'Inner Tank Length (m)', 'Vh2 (m^3)','R_in (m)','CO2 (kg)','Embodied Energy (MJ)','Angle of winding inner','Angle of winding outer', 'P_vent','Qleak', 'Q_cond', 'Q_rad'])
                    # Write data
                    writer.writerow([material, material2, MAWP, Vt, Mt, mass_error, t1, t2, dv, L_in, Vh2,R_in,co2_kg,emb_energy, ang1_w, ang2_w,p_vent,Qleak, Qcond, Qrad])


if OPEN:
    with open('tank_results.csv', mode='r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header row
        for row in reader:
            plot_mats.append(row[0])
            plot_mats2.append(row[1])
            plot_MAWPS.append(float(row[2]))
            plot_volumes.append(float(row[3]))
            plot_masses.append(float(row[4]))
            plot_mass_errors.append(float(row[5]))
            plot_vh2.append(float(row[10]))
            plot_co2.append(float(row[12]))
            plot_embodied_energy.append(float(row[13]))
            plot_P_vents.append(float(row[16]))
# ------------------------------------------------- Plotting ------------------------------------------------------

plot1 = True
plot2=True
plot3 = True

if plot1:
    plt.figure(figsize=(10, 6))
    unique_combinations = len(materials)**2
    colors = plt.cm.tab20(np.linspace(0, 1, unique_combinations))
    markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'h', 'H', 'X', 'd']

    for idx, (material, material2) in enumerate([(m1, m2) for m1 in materials for m2 in materials]):
        color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]
        label = f"{material}-{material2}"
        x_values = [plot_P_vents[k] for k in range(len(plot_P_vents)) if plot_mats[k] == material and plot_mats2[k] == material2]
        y_values = [plot_volumes[k] for k in range(len(plot_volumes)) if plot_mats[k] == material and plot_mats2[k] == material2]
        if x_values and y_values:
            plt.plot(x_values, y_values, label=label, color=color, marker=marker, linestyle='-')

    plt.xlabel("P_vent (Pa)")
    plt.ylabel("Tank Volume (m^3)")
    plt.title("Tank Volume vs P_vent")
    plt.subplots_adjust(left=0.1, right=0.6, wspace=0.3)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)
    plt.show()
                
        
if plot2:
    plt.figure(figsize=(10, 6))
    unique_combinations = len(materials)**2 * len(P_vents)
    colors = plt.cm.tab20(np.linspace(0, 1, unique_combinations))
    markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'h', 'H', 'X', 'd']

    for idx, (i, j) in enumerate([(i, j) for i in range(len(materials)**2) for j in range(len(P_vents))]):
        color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]
        plt.scatter(
            mass_h2 / plot_masses[i + j * len(materials)**2],
            plot_vh2[i + j * len(materials)**2] / plot_volumes[i + j * len(materials)**2],
            label=f"{plot_mats[i]}-{plot_mats2[i]}-PVENT={plot_P_vents[i + j * len(materials)**2]/100000} Bar",
            color=color,
            marker=marker
        )

    plt.xlabel(r"$\eta_g$")
    plt.ylabel(r"$\eta_v$")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.title(r"$\eta_g$ vs $\eta_v$")
    plt.subplots_adjust(left=0.1, right=0.6, wspace=0.3)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)
    plt.show()

if plot3:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    unique_combinations = len(materials)**2 * len(P_vents)
    colors = plt.cm.tab20(np.linspace(0, 1, unique_combinations))
    markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'h', 'H', 'X', 'd']

    for idx, (i, j) in enumerate([(i, j) for i in range(len(materials)**2) for j in range(len(P_vents))]):
        color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]
        eta_v = plot_vh2[i + j * len(materials)**2] / plot_volumes[i + j * len(materials)**2]
        eta_g = mass_h2 / plot_masses[i + j * len(materials)**2]
        metric = eta_g * eta_v

        # Plot kg of CO2 vs metric
        ax1.scatter(
            metric,
            plot_co2[i + j * len(materials)**2],
            color=color,
            marker=marker
        )

        # Plot embodied energy vs metric
        ax2.scatter(
            metric,
            plot_embodied_energy[i + j * len(materials)**2],
            label=f"{plot_mats[i]}-{plot_mats2[i]}-PVENT={plot_P_vents[i + j * len(materials)**2]/100000} Bar",
            color=color,
            marker=marker
        )

    ax1.set_xlabel("Metric (ηg * ηv)")
    ax1.set_ylabel("CO2 Emissions (kg)")
    ax1.set_title("CO2 Emissions vs Metric")

    ax2.set_xlabel("Metric (ηg * ηv)")
    ax2.set_ylabel("Embodied Energy (MJ)")
    ax2.set_title("Embodied Energy vs Metric")

    ax2.legend(
        loc='center left',
        bbox_to_anchor=(1, 0.5),
        title="Material Combinations",
        frameon=False
    )

    plt.tight_layout()
    plt.show()
    
#-------------------------------------------------- Draw Tank ---------------------------------------------------
draw1 = True

if draw1:
    for i, (material, material2, MAWP, Vt, Mt, mass_error, t1, t2, dv, L_in, Vh2, R_in,co2_kg) in enumerate(
        zip([(row[0]) for row in csv.reader(open('tank_results.csv')) if row[0] != 'Material Inner'],
            [(row[1]) for row in csv.reader(open('tank_results.csv')) if row[0] != 'Material Inner'],
            [float(row[2]) for row in csv.reader(open('tank_results.csv')) if row[0] != 'Material Inner'],
            [float(row[3]) for row in csv.reader(open('tank_results.csv')) if row[0] != 'Material Inner'],
            [float(row[4]) for row in csv.reader(open('tank_results.csv')) if row[0] != 'Material Inner'],
            [float(row[5]) for row in csv.reader(open('tank_results.csv')) if row[0] != 'Material Inner'],
            [float(row[6]) for row in csv.reader(open('tank_results.csv')) if row[0] != 'Material Inner'],
            [float(row[7]) for row in csv.reader(open('tank_results.csv')) if row[0] != 'Material Inner'], 
            [float(row[8]) for row in csv.reader(open('tank_results.csv')) if row[0] != 'Material Inner'], 
            [float(row[9]) for row in csv.reader(open('tank_results.csv')) if row[0] != 'Material Inner'], 
            [float(row[10]) for row in csv.reader(open('tank_results.csv')) if row[0] != 'Material Inner'], 
            [float(row[11]) for row in csv.reader(open('tank_results.csv')) if row[0] != 'Material Inner'],
            [float(row[12]) for row in csv.reader(open('tank_results.csv')) if row[0] != 'Material Inner'])):
        
        # Tank dimensions
        r_inner = R_in
        r_inner_wall = r_inner + t1
        r_mli_outer = r_inner_wall + t_mli
        r_vacuum_outer = r_mli_outer + dv
        r_outer = r_vacuum_outer + t2

        # Create the figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        ax1.set_aspect('equal', adjustable='box')
        ax2.set_aspect('auto', adjustable='box')

        # ---------------- Top-down cross-section ----------------
        
        # Draw the vacuum gap
        vacuum_circle = plt.Circle((0, 0), r_vacuum_outer, color='lightgray', label='Vacuum Gap', fill=True, linestyle='--', linewidth=1)
        ax1.add_artist(vacuum_circle)

        # Draw the inner volume
        inner_volume_circle = plt.Circle((0, 0), r_inner, color='white', fill=True, edgecolor='black', linewidth=1)
        ax1.add_artist(inner_volume_circle)

        # Draw the inner tank wall
        inner_wall_circle = plt.Circle((0, 0), r_inner_wall, color='green', label='Inner Tank Wall', fill=False, linewidth=2)
        ax1.add_artist(inner_wall_circle)

        # Draw the MLI insulation
        mli_circle = plt.Circle((0, 0), r_mli_outer, color='orange', label='MLI Insulation', fill=False, linewidth=2)
        ax1.add_artist(mli_circle)

        # Draw the outer tank wall
        outer_wall_circle = plt.Circle((0, 0), r_outer, color='red', label='Outer Tank Wall', fill=False, linewidth=2)
        ax1.add_artist(outer_wall_circle)        

        # Set limits for top-down view
        padding = r_outer * 0.2
        ax1.set_xlim(-r_outer - padding, r_outer + padding)
        ax1.set_ylim(-r_outer - padding, r_outer + padding)

        # ---------------- Side cross-section ----------------
        # Draw the inner tank wall (side view)
        L_cyl = L_in - 2 * r_inner
        ax2.plot([-L_cyl / 2, L_cyl / 2], [r_inner_wall, r_inner_wall], color='green', label='Inner Tank Wall', linewidth=2)
        ax2.plot([-L_cyl / 2, L_cyl / 2], [-r_inner_wall, -r_inner_wall], color='green', linewidth=2)

        # Draw the hemispherical caps for the inner tank wall
        theta = np.linspace(np.pi/2, 3*np.pi/2, 100)
        ax2.plot(-L_cyl / 2 + r_inner * np.cos(theta), r_inner * np.sin(theta), color='green', linewidth=2)
        ax2.plot(L_cyl / 2 - r_inner * np.cos(theta), r_inner * np.sin(theta), color='green', linewidth=2)

        # Draw the MLI insulation (side view)
        ax2.plot([-L_cyl / 2, L_cyl / 2], [r_mli_outer, r_mli_outer], color='orange', label='MLI Insulation', linewidth=2)
        ax2.plot([-L_cyl / 2, L_cyl / 2], [-r_mli_outer, -r_mli_outer], color='orange', linewidth=2)

        # Draw the hemispherical caps for the MLI insulation
        ax2.plot(-L_cyl / 2 + r_mli_outer * np.cos(theta), r_mli_outer * np.sin(theta), color='orange', linewidth=2)
        ax2.plot(L_cyl / 2 - r_mli_outer * np.cos(theta), r_mli_outer * np.sin(theta), color='orange', linewidth=2)

        # Draw the vacuum gap (side view)
        ax2.plot([-L_cyl / 2, L_cyl / 2], [r_vacuum_outer, r_vacuum_outer], color='gray', label='Vacuum Gap', linestyle='--', linewidth=1)
        ax2.plot([-L_cyl / 2, L_cyl / 2], [-r_vacuum_outer, -r_vacuum_outer], color='gray', linestyle='--', linewidth=1)

        # Draw the hemispherical caps for the vacuum gap
        ax2.plot(-L_cyl / 2 + r_vacuum_outer * np.cos(theta), r_vacuum_outer * np.sin(theta), color='gray', linestyle='--', linewidth=1)
        ax2.plot(L_cyl / 2 - r_vacuum_outer * np.cos(theta), r_vacuum_outer * np.sin(theta), color='gray', linestyle='--', linewidth=1)

        # Draw the outer tank wall (side view)
        ax2.plot([-L_cyl / 2, L_cyl / 2], [r_outer, r_outer], color='red', label='Outer Tank Wall', linewidth=2)
        ax2.plot([-L_cyl / 2, L_cyl / 2], [-r_outer, -r_outer], color='red', linewidth=2)
        ax2.plot([-L_cyl / 2, -L_cyl / 2], [-r_outer, r_outer], color='red', linewidth=1,linestyle='--')
        ax2.plot([L_cyl / 2, L_cyl / 2], [-r_outer, r_outer], color='red', linewidth=1,linestyle='--')

        # Draw the hemispherical caps for the outer tank wall
        ax2.plot(-L_cyl / 2 + r_outer * np.cos(theta), r_outer * np.sin(theta), color='red', linewidth=2)
        ax2.plot(L_cyl / 2 - r_outer * np.cos(theta), r_outer * np.sin(theta), color='red', linewidth=2)

        # Set limits for side view
        max_dim = max(L_in / 2 + padding, r_outer + padding)
        ax2.set_xlim(-max_dim*1.5, max_dim*1.5)
        ax2.set_ylim(-max_dim*1.5, max_dim*1.5)


        # Add overall title
        fig.suptitle(f"Tank Design Visualization", fontsize=16)

        # Add subtitles for the cross-section plots
        ax1.set_title("Top-down Cross-section", fontsize=14)
        ax2.set_title("Side Cross-section", fontsize=14)

        # Adjust the layout to move the graphs slightly to the right
        fig.subplots_adjust(left=0.25, right=0.9, wspace=0.3)

        # Add legend to the left of the plots for ax1
        ax1.legend(
            loc='center left', 
            bbox_to_anchor=(-0.7, 0.5), 
            title=(
            f"Tank Details\n"
            f"Total Volume: {Vt:.4f} m³\n"
            f"Total Mass: {Mt:.4f} kg\n"
            f"Inner Material: {material}\n"
            f"Outer Material: {material2}"
            )
        )

        # Show the figure
        plt.show()
