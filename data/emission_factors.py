"""
Module containing reference emission factors for different fuel types,
technologies, and processes based on the Scope 1 monitoring system.
"""

# Emission factors for different fuel types (kg CO2e per unit)
FUEL_EMISSION_FACTORS = {
    # Natural gas (kg CO2e per cubic meter)
    "natural_gas": {
        "co2": 1.89,
        "ch4": 0.00038,
        "n2o": 0.000035,
        "co2e": 1.91  # CO2 equivalent
    },
    
    # Diesel (kg CO2e per liter)
    "diesel": {
        "co2": 2.68,
        "ch4": 0.00034,
        "n2o": 0.000022,
        "co2e": 2.69  # CO2 equivalent
    },
    
    # Gasoline (kg CO2e per liter)
    "gasoline": {
        "co2": 2.31,
        "ch4": 0.00034,
        "n2o": 0.000034,
        "co2e": 2.33  # CO2 equivalent
    },
    
    # Liquefied Petroleum Gas (LPG) (kg CO2e per liter)
    "lpg": {
        "co2": 1.51,
        "ch4": 0.00005,
        "n2o": 0.000001,
        "co2e": 1.52  # CO2 equivalent
    },
    
    # Heavy Fuel Oil (kg CO2e per liter)
    "heavy_fuel_oil": {
        "co2": 3.15,
        "ch4": 0.00012,
        "n2o": 0.000016,
        "co2e": 3.17  # CO2 equivalent
    }
}

# Refrigerant Global Warming Potentials (100-year)
REFRIGERANT_GWP = {
    "R-134a": 1430,
    "R-410A": 2088,
    "R-22": 1810,
    "R-32": 675,
    "R-1234yf": 4,
    "R-404A": 3922,
    "R-407C": 1774,
    "R-507A": 3985,
    "SF6": 22800,  # Sulfur hexafluoride
    "NF3": 17200,  # Nitrogen trifluoride
}

# Vehicle emission factors (kg CO2e per km)
VEHICLE_EMISSION_FACTORS = {
    "passenger_car_gasoline": 0.180,
    "passenger_car_diesel": 0.160,
    "passenger_car_hybrid": 0.100,
    "passenger_car_electric": 0.0,  # At point of use (Scope 1)
    "light_duty_truck_gasoline": 0.250,
    "light_duty_truck_diesel": 0.230,
    "heavy_duty_truck_diesel": 0.900,
    "bus_diesel": 1.200,
    "bus_electric": 0.0  # At point of use (Scope 1)
}

# Emission reduction factors for different technologies
# (percentage reduction compared to baseline)
TECHNOLOGY_REDUCTION_FACTORS = {
    "electric_boilers": 1.0,  # 100% reduction in direct emissions
    "high_efficiency_boilers": 0.20,  # 20% reduction
    "electric_vehicles": 1.0,  # 100% reduction in direct emissions
    "hybrid_vehicles": 0.40,  # 40% reduction
    "renewable_natural_gas": 0.80,  # 80% reduction
    "led_lighting": 0.75,  # 75% reduction
    "energy_management_systems": 0.15,  # 15% reduction
    "refrigerant_leak_detection": 0.80,  # 80% reduction
    "industrial_process_optimization": 0.25  # 25% reduction
}

# Carbon price trajectories ($/tCO2e)
CARBON_PRICE_TRAJECTORIES = {
    "low": {
        2025: 30,
        2026: 32,
        2027: 34,
        2028: 36,
        2029: 38,
        2030: 40,
        2035: 50,
        2040: 60,
        2045: 70,
        2050: 80
    },
    "medium": {
        2025: 30,
        2026: 33,
        2027: 36,
        2028: 40,
        2029: 44,
        2030: 48,
        2035: 70,
        2040: 100,
        2045: 130,
        2050: 160
    },
    "high": {
        2025: 50,
        2026: 59,
        2027: 69,
        2028: 81,
        2029: 95,
        2030: 112,
        2035: 200,
        2040: 300,
        2045: 400,
        2050: 500
    }
}

# Get carbon price for a specific year and trajectory
def get_carbon_price(year, trajectory="medium"):
    """
    Get the carbon price for a specific year and trajectory.
    
    Args:
        year (int): The year to get the carbon price for
        trajectory (str): The carbon price trajectory ("low", "medium", or "high")
        
    Returns:
        float: Carbon price in $/tCO2e
    """
    prices = CARBON_PRICE_TRAJECTORIES[trajectory]
    
    # Exact match
    if year in prices:
        return prices[year]
    
    # Interpolate
    years = sorted(prices.keys())
    
    if year < years[0]:
        return prices[years[0]]
    
    if year > years[-1]:
        return prices[years[-1]]
    
    # Find surrounding years
    for i in range(len(years) - 1):
        if years[i] <= year < years[i + 1]:
            # Linear interpolation
            y1, y2 = years[i], years[i + 1]
            p1, p2 = prices[y1], prices[y2]
            return p1 + (p2 - p1) * (year - y1) / (y2 - y1)
    
    return prices[years[-1]]  # Default to last year if not found

# Testing
if __name__ == "__main__":
    print("Fuel Emission Factors:")
    for fuel, factors in FUEL_EMISSION_FACTORS.items():
        print(f"{fuel}: {factors['co2e']} kg CO2e per unit")
    
    print("\nCarbon Price in 2028 (medium trajectory):", get_carbon_price(2028, "medium"))
    print("Carbon Price in 2032 (high trajectory):", get_carbon_price(2032, "high"))