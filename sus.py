from global_constants import *  # Import global constants

def get_sus(GWP_fc, GWP_sto, GWP_eps, m_h2, m_nox):
    """
    Calculate the GWP of the aircraft per flight.
    :param GWP_fc: GWP of the fuel cell production/disposal [kg CO2]
    :param GWP_sto: GWP of the storage system production/disposal [kg CO2]
    :param GWP_eps: GWP of the electric propulsion system production/disposal [kg CO2]
    :param m_h2: mass of hydrogen used [kg]
    :param m_nox: mass of NOx produced [kg]
    :return: GWP of the aircraft per flight [kg CO2]
    """
    GWP_ac = (GWP_fc + GWP_sto + GWP_eps + GWP_beech) / flight_lifetime  # Total GWP of the aircraft per flight
    m_h2o = m_h2 * 18.01528 / 2.01568  # [kg] Mass of water produced over the flight
    GWP_h2 = m_h2 * GWP_H2  # [kg CO2] GWP of the hydrogen used
    GWP_h2o = m_h2o * GWP_H2O  # [kg CO2] GWP of the water produced
    GWP_nox = m_nox * GWP_NOx  # [kg CO2] GWP of the NOx produced
    GWP_total = GWP_ac + GWP_h2o + GWP_nox + GWP_h2  # Total GWP of the aircraft per flight

    return GWP_total


def get_sus_breakdown(GWP_fc, GWP_sto, GWP_eps, m_h2, m_nox):
    """
    Calculate the GWP of the aircraft per flight.
    :param GWP_fc: GWP of the fuel cell production/disposal [kg CO2]
    :param GWP_sto: GWP of the storage system production/disposal [kg CO2]
    :param GWP_eps: GWP of the electric propulsion system production/disposal [kg CO2]
    :param m_h2: mass of hydrogen used [kg]
    :param m_nox: mass of NOx produced [kg]
    :return: GWP of the aircraft per flight [kg CO2]
    """

    GWP_ac = (GWP_fc + GWP_sto + GWP_eps + GWP_beech) / flight_lifetime,  # Total GWP of the aircraft per flight
    m_h2o = m_h2 * 18.01528 / 2.01568 , # [kg] Mass of water produced over the flight
    GWP_h2 = m_h2 * GWP_H2,  # [kg CO2] GWP of the hydrogen used
    GWP_h2o = m_h2o * GWP_H2O,  # [kg CO2] GWP of the water produced
    GWP_nox = m_nox * GWP_NOx,

    GWP_dict = dict(
        GWP_FC=GWP_fc/flight_lifetime,  # GWP of the fuel cell production/disposal [kg CO2]
        GWP_STO=GWP_sto/flight_lifetime,  # GWP of the storage system
        GWP_EPS=GWP_eps/flight_lifetime,  # GWP of the electric propulsion system production/disposal [kg CO2]
        GWP_BEECH=GWP_beech/flight_lifetime,  # GWP of the beech production/disposal [kg CO2]
        GWP_ac = (GWP_fc + GWP_sto + GWP_eps + GWP_beech) / flight_lifetime,  # Total GWP of the aircraft per flight
        m_h2o = m_h2 * 18.01528 / 2.01568 , # [kg] Mass of water produced over the flight
        GWP_h2 = m_h2 * GWP_H2,  # [kg CO2] GWP of the hydrogen used
        GWP_h2o = m_h2o * GWP_H2O,  # [kg CO2] GWP of the water produced
        GWP_nox = m_nox * GWP_NOx,  # [kg CO2] GWP of the NOx produced
        GWP_total = GWP_ac + GWP_h2o + GWP_nox + GWP_h2  # Total GWP of the aircraft per flight
    )

    return GWP_dict 