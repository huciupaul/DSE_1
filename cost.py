from global_constants import *


def calc_cost(fc_cost, P_eps, m_sto, m_h2, m_h2_used):

    # -----Initial costs-----
    #fc_init = FC_cost * P_FC
    sto_init = Sto_cost * m_h2
    eps_init = P_eps * EPS_cost / 1000  # Convert to kW
    
    purchase_cost = sto_init + eps_init + AC_purchase_cost + AC_dev_cost
    # print(f"Initial costs: FC={0}, STO={sto_init}, EPS={eps_init}, AC={AC_purchase_cost}, Dev={AC_dev_cost}, Total={purchase_cost}")


    # -----Operational costs  per year-----
    #fc_maint = FC_maint_cost * P_FC / years_of_life
    sto_maint = Sto_maint_cost * m_h2 
    landing = Landing_tax / years_of_life # 7.5 tones weight range
    crew = Crew_cost / years_of_life
    fuel_cost = H2_cost * m_h2_used * flights_per_year #  920 flights/year
    insurance = Insurance_cost / years_of_life
    
    operational_costs = sto_maint + landing + crew + fuel_cost + insurance + AC_maint_cost
    # print(f"Operational costs per year: FC={0}, STO={sto_maint}, Landing={landing}, Crew={crew}, Fuel={fuel_cost}, Insurance={insurance}, Total={operational_costs}")


    # -----Disposal costs-----
    #fc_disposal = FC_disposal_cost * m_fc 
    sto_disposal = Sto_disposal_cost * m_sto

    disposal_cost =  sto_disposal + AC_disposal_cost
    # print(f"Disposal costs: FC={0}, STO={sto_disposal}, Total={disposal_cost}")


    # -----Depreciation-----
    depreciation = purchase_cost - (purchase_cost / ((1 + depreciation_rate) ** years_of_life))
    

    # -----Total cost-----
    total_cost = purchase_cost + operational_costs * years_of_life + disposal_cost + depreciation + fc_cost
    #print(f"Total cost: {total_cost}")


    # -----Ticket calculator-----
    avg_ticket_price = total_cost / (15 * flight_lifetime) # 15 pax
    return total_cost #, avg_ticket_price

# print(calc_cost(940835, 500*.99*.97, 275, 250))

def calc_cost_breakdown(fc_cost, P_eps, m_sto, m_h2, m_h2_used):

    # -----Initial costs-----
    #fc_init = FC_cost * P_FC
    sto_init = Sto_cost * m_h2
    eps_init = P_eps * EPS_cost / 1000  # Convert to kW
    
    purchase_cost = sto_init + eps_init + AC_purchase_cost + AC_dev_cost
    # print(f"Initial costs: FC={0}, STO={sto_init}, EPS={eps_init}, AC={AC_purchase_cost}, Dev={AC_dev_cost}, Total={purchase_cost}")


    # -----Operational costs  per year-----
    #fc_maint = FC_maint_cost * P_FC / years_of_life
    sto_maint = Sto_maint_cost * m_h2 
    landing = Landing_tax / years_of_life # 7.5 tones weight range
    crew = Crew_cost / years_of_life
    fuel_cost = H2_cost * m_h2_used * flights_per_year #  920 flights/year
    insurance = Insurance_cost / years_of_life
    
    operational_costs = sto_maint + landing + crew + fuel_cost + insurance + AC_maint_cost
    # print(f"Operational costs per year: FC={0}, STO={sto_maint}, Landing={landing}, Crew={crew}, Fuel={fuel_cost}, Insurance={insurance}, Total={operational_costs}")


    # -----Disposal costs-----
    #fc_disposal = FC_disposal_cost * m_fc 
    sto_disposal = Sto_disposal_cost * m_sto

    disposal_cost =  sto_disposal + AC_disposal_cost
    # print(f"Disposal costs: FC={0}, STO={sto_disposal}, Total={disposal_cost}")


    # -----Depreciation-----
    depreciation = purchase_cost - (purchase_cost / ((1 + depreciation_rate) ** years_of_life))
    

    # -----Total cost-----
    total_cost = purchase_cost + operational_costs * years_of_life + disposal_cost + depreciation + fc_cost
    #print(f"Total cost: {total_cost}")


    # -----Ticket calculator-----
    avg_ticket_price = total_cost / (15 * flight_lifetime) # 15 pax

    cost_breakdown = dict(
        purchase_cost=purchase_cost,
        # disposal_cost=disposal_cost,
        # depreciation=depreciation,
        fc_cost=fc_cost,
        sto_prod=sto_init,
        eps_prod=eps_init,
        ac_purchase=AC_purchase_cost,
        ac_dev=AC_dev_cost,
        operational_costs=operational_costs * years_of_life,
        sto_maint=sto_maint * years_of_life,
        landing=landing * years_of_life,
        crew=crew * years_of_life,
        fuel_cost=fuel_cost * years_of_life,
        insurance=insurance * years_of_life,
        disposal_cost=disposal_cost,
        sto_disposal=sto_disposal,
        ac_disposal=AC_disposal_cost,
        depreciation=depreciation,
        total_cost=total_cost,
        avg_ticket_price=avg_ticket_price
    )

    return cost_breakdown #, avg_ticket_price
