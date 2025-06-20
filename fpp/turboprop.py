import numpy as np
from typing import Tuple

# Local imports
# from common.constants import R_AIR, MAXC

# Global imports
import sys
import os

from global_constants import R_AIR, MAXC
from global_constants import mdot_air as mfa
from global_constants import mdot_fuel as mff
from global_constants import mdot_NOx as mfn
from global_constants import T_peak_interpolator as tcc
from global_constants import TOGA


class Turboprop:
    """Full model of a turboprop engine. The model is based on the following
    assumptions:
        - Brayton cycle
        - Constant inlet, compressor, combustion chamber, and turbine efficiencies
        - Constant specific heat ratio of air and combustion products
        - Constant specific heat of combustion products
        - Constant pressure ratio across the compressor and combustion chamber
        - Constant turbine inlet temperature
        - ISA atmosphere
        - No bleed air extraction
        - No variable geometry
    """
    def __init__(
        self,
        name: str,
        delta_mdot: float, # [kg/s] Range of air mass flow rate through the engine
        mdot_min: float, # [kg/s] Minimum air mass flow rate through the engine
        eta_in: float,  # [-] Inlet efficiency
        PI_comp: float, # [-] Compressor pressure ratio
        eta_comp: float, # [-] Compressor efficiency
        PI_cc: float,  # [-] Combustion chamber pressure ratio
        eta_cc: float, # [-] Combustion chamber efficiency
        LHV_fuel: float, # [J/kg] Lower heating value of fuel
        T04: float,  # [K] Turbine inlet temperature
        eta_turb: float, # [-] Turbine efficiency
        eta_mech: float, # [-] Mechanical efficiency
        c_pa: float, # [J/(kg·K)] Specific heat of air
        k_air: float,   # [-] Specific heat ratio of air
        c_pg: float, # [J/(kg·K)] Specific heat of combustion products
        k_gas: float  # [-] Specific heat ratio of combustion products
    ) -> None:
        
        self.name = name
        self.delta_mdot = delta_mdot
        self.mdot_min = mdot_min
        self.eta_in = eta_in
        self.PI_comp = PI_comp
        self.eta_comp = eta_comp
        self.PI_cc = PI_cc
        self.eta_cc = eta_cc
        self.LHV_fuel = LHV_fuel
        self.T04 = T04
        self.eta_turb = eta_turb
        self.eta_mech = eta_mech
        self.c_pa = c_pa
        self.k_air = k_air
        self.c_pg = c_pg
        self.k_gas = k_gas
    
    #  INTERNAL HELPERS
    @staticmethod
    def __02(
        T0: np.ndarray,
        P0: np.ndarray,
        M0: np.ndarray,
        eta_in: float,
        k_air: float
        ) -> np.ndarray:
        """Station 2 (compressor inlet) conditions."""
        T02 = T0*(1 + (k_air-1)*M0**2/2)
        P02 = P0*(1 + eta_in*(k_air-1)*M0**2/2)**(k_air/(k_air-1))
        return T02, P02
    
    @staticmethod
    def __03(
        T02: np.ndarray,
        P02: np.ndarray,
        PI_comp: float,
        eta_comp: float,
        k_air: float
        ) -> np.ndarray:
        """Station 3 (compressor exit) conditions."""
        T03 = T02*(1 + (1/eta_comp)*(PI_comp**((k_air-1)/k_air)-1))
        P03 = P02*PI_comp
        return T03, P03
    
    @staticmethod
    def __04(
        T04: np.ndarray,
        P03: np.ndarray,
        PI_cc: float
    ):
        """Station 4 (combustion chamber exit) conditions."""
        T04 = T04
        P04 = P03*PI_cc
        return T04, P04
    
    @staticmethod
    def __mdot_fuel(
        mdot_air: np.ndarray,
        c_pg: float,
        T03: np.ndarray,
        T04: np.ndarray,
        LHV_fuel: float,
        eta_cc: float
    ):
        """Mass flow rate of fuel."""
        mdot_fuel = mdot_air*(c_pg*(T04-T03)/(LHV_fuel*eta_cc))
        return mdot_fuel
    
    @staticmethod
    def __05(
        mdot_air: np.ndarray,
        mdot_fuel: np.ndarray,
        c_pa: float,
        c_pg: float,
        T02: np.ndarray,
        T03: np.ndarray,
        T04: np.ndarray,
        P04: np.ndarray,
        eta_mech: float,
        eta_turb: float,
        k_gas: float
    ):
        """Station 5 (turbine exit) conditions."""
        mdot = mdot_air + mdot_fuel
        Wdot_comp = mdot_air*c_pa*(T03-T02)
        Wdot_turb = Wdot_comp/eta_mech
        T05 = T04 - Wdot_turb/(mdot*c_pg)
        P05 = P04*(1-(1/eta_turb)*(1-T05/T04))**(k_gas/(k_gas-1))
        return T05, P05
    
    @staticmethod
    def __mu(
        M0: np.ndarray,
        k_air: float
    ):
        """Mach number correction factor."""
        mu = 1 + (k_air-1)/2*M0**2
        return mu
    
    @staticmethod
    def __kappa(
        mu: np.ndarray,
        PI_comp: float,
        k_air: float
    ):
        """Kappa factor taken from Torenbeek.
        Note: Assumes epsilon_c = PI_comp"""
        kappa = mu*(PI_comp**((k_air-1)/k_air)-1)
        return kappa
    
    @staticmethod
    def __phi(
        T0: np.ndarray,
        T04: np.ndarray
    ):
        """Non-dimensional TET taken from Torenbeek."""
        phi = T04/T0
        return phi
    
    @staticmethod
    def __G(
        mu: np.ndarray,
        phi: np.ndarray,
        kappa: np.ndarray,
        eta_comp: float,
        eta_in: float,
        eta_turb: float,
        k_air: float
    ):
        """Gas generator function taken from Torenbeek."""
        G = (phi-kappa/eta_comp)*(1-1.01/(eta_in**((k_air-1)/k_air)*(kappa+mu)*(1-kappa/(phi*eta_comp*eta_turb))))
        return G
    
    def __PSFC(
        self,
        M0: np.ndarray,
        T0: np.ndarray
        
    ):
        """Power-specific fuel consumption estimation taken from Torenbeek.
        Torenbeek, S. (1986). Synthesis of Subsonic Airplane Design."""
        mu = self.__mu(M0, self.k_air)
        kappa = self.__kappa(mu, self.PI_comp, self.k_air)
        phi = self.__phi(self.T04, T0)
        
        G = self.__G(mu, phi, kappa, self.eta_comp, self.eta_in, self.eta_turb, self.k_air)
        
        # Eqn. H-40, Torenbeek, pp. 569, constant adjusted for SI units
        C_p = 2.41011428486284326e-8*((phi-mu-kappa/self.eta_comp)/(self.eta_turb*G-0.28*M0**2/self.eta_turb))
        C_p = np.clip(C_p, 0, None)*2 # Avoid negative values and two engines
        
        return C_p
    
    @staticmethod
    def _eta_prop(
        M0: np.ndarray,
        eta_prop_max: float = 0.85
    ):
        """Empirical propeller/propulsion efficiency.
        Taken from: Mattingly, Aircraft Engine Design, 2nd ed.
        """
        eta_prop = np.zeros_like(M0)
        
        # M0 < 0.1
        mask = M0 <= 0.1
        eta_prop[mask] = 10*M0[mask]*eta_prop_max
        
        # 0.1 < M0 <= 0.7
        mask = (M0 > 0.1) & (M0 <= 0.7)
        eta_prop[mask] = eta_prop_max
        
        # 0.7 < M0 <= 1.0
        mask = (M0 > 0.7) & (M0 <= 0.85)
        eta_prop[mask] = (1-((M0[mask]-0.7)/3))*eta_prop_max
        
        return eta_prop
    
    @staticmethod
    def __eta_th(
        mdot_air: np.ndarray,
        mdot_fuel: np.ndarray,
        T03: np.ndarray,
        T04: np.ndarray,
        T05: np.ndarray,
        P0: np.ndarray,
        P05: np.ndarray,
        V0: np.ndarray,
        c_pg: float,
        k_gas: float
    ):
        """Thermal efficiency."""
        mdot = mdot_air + mdot_fuel
        Wdot_gg = mdot*c_pg*T05*(1-(P0/P05)**((k_gas-1)/k_gas))-0.5*mdot*V0**2
        eta_th = Wdot_gg/(mdot*c_pg*(T04-T03))
        return eta_th
    
    @staticmethod
    def __mdot_air(
        Pa_TOGA: float,
        Pa: np.ndarray,
        delta_mdot: float,
        mdot_min: float
    ):
        """Linear model of the air mass flow rate."""
        # mdot_air = (Pa/Pa_TOGA)*delta_mdot + mdot_min
        mdot_air = mfa(Pa/2e3)
        return mdot_air*2
    
    @staticmethod
    def __mdot_NOx(
        Pa: np.ndarray,
        mdot_H2O: np.ndarray
    ):
        """Mass flow rate of NOx."""
        mfh = np.clip(mdot_H2O, 0, 0.1*2)/2
        p_ratio = np.clip(Pa/MAXC, 0.057, 1.04)
        
        mdot_NOx = mfn((mfh, p_ratio))  # Use the mass flow rate of water and the pressure ratio
        mdot_NOx = np.where(Pa == 0.0, 0.0, mdot_NOx)
        
        # T = np.zeros_like(Pa)
        # T = np.clip(tcc(Pa/2e3), 1570, 1740)  # Use the combustion chamber inlet temperature
        # mdot_NOx = np.where(Pa>=TOGA*0.056, mfn((T, mfh/2)), 0)  # Use the combustion chamber inlet temperature and half the mass flow rate of water
        return mdot_NOx*2  # Multiply by 2 for two engines
    
    #  COMPUTE
    def compute(
        self, 
        T0: np.ndarray,
        P0: np.ndarray,
        rho0:np.ndarray,
        V0: np.ndarray,
        R_LHV: float=1.0,
        Pr: np.ndarray=None,
        Pa: np.ndarray=None,
        mdot_H2O: np.ndarray=None
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute the turboprop engine performance."""
        
        # Atmospheric conditions
        M0 = V0/np.sqrt(self.k_air*R_AIR*T0)
        
        # Propeller efficiency
        eta_prop = self._eta_prop(M0)
        
        if Pa is None:
            Pa = Pr/eta_prop
        
        # Inlet mass flow rate
        mdot_air = self.__mdot_air(MAXC*1.04, Pa, self.delta_mdot, self.mdot_min)*2
        
        # Station 2 (compressor inlet)
        T02, P02 = self.__02(T0, P0, M0, self.eta_in, self.k_air)

        # Station 3 (compressor exit)
        T03, P03 = self.__03(T02, P02, self.PI_comp, self.eta_comp, self.k_air)

        # Station 4 (combustion chamber exit)
        T04, P04 = self.__04(self.T04, P03, self.PI_cc)

        # Power-specific fuel consumption
        C_p = self.__PSFC(M0, T0)
        
        # Fuel flow rate
        # mdot_fuel = C_p*R_LHV*Pa/(1-0.16) # Max engine installation losses
        mdot_fuel = mff(Pa/2e3)*2
        
        # NOx emissions
        if mdot_H2O is not None:
            mdot_NOx = self.__mdot_NOx(Pa, mdot_H2O)
        else:
            mdot_NOx = np.zeros_like(Pa)

        # Station 5 (turbine exit)
        T05, P05 = self.__05(mdot_air, mdot_fuel, self.c_pa, self.c_pg, T02, T03, T04, P04,
                             self.eta_mech, self.eta_turb, self.k_gas)

        # Thermal efficiency
        eta_th = self.__eta_th(mdot_air, mdot_fuel, T03, T04, T05,
                               P0, P05, V0,
                               self.c_pg,
                               self.k_gas)
        
        return mdot_fuel, eta_th, eta_prop, mdot_air, T03, P03, mdot_NOx