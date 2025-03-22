"""
Module containing data on emission reduction technologies, including
implementation costs, efficiency improvements, and adoption timelines.
"""

# Technology implementation costs and parameters
TECHNOLOGY_DATA = {
    # Electric boilers to replace natural gas boilers
    "electric_boilers": {
        "capex_per_unit": 500000,  # $ per unit
        "opex_change_per_year": 20000,  # $ per year (can be negative for savings)
        "lifespan_years": 15,
        "emission_reduction_percentage": 1.0,  # 100% reduction in direct emissions
        "implementation_time_months": 6,
        "maintenance_cost_percentage": 0.05,  # 5% of CAPEX per year
        "fuel_cost_change_percentage": 0.25,  # 25% increase in fuel costs
        "typical_implementation_units": 1,  # Typical units per implementation
        "applicability": ["natural_gas_boilers"]
    },
    
    # High-efficiency gas boilers
    "high_efficiency_boilers": {
        "capex_per_unit": 200000,  # $ per unit
        "opex_change_per_year": -5000,  # $ per year (negative means savings)
        "lifespan_years": 15,
        "emission_reduction_percentage": 0.20,  # 20% reduction
        "implementation_time_months": 3,
        "maintenance_cost_percentage": 0.04,  # 4% of CAPEX per year
        "fuel_cost_change_percentage": -0.15,  # 15% decrease in fuel costs
        "typical_implementation_units": 1,
        "applicability": ["natural_gas_boilers"]
    },
    
    # Electric vehicles to replace company fleet
    "electric_vehicles": {
        "capex_per_unit": 55000,  # $ per vehicle
        "opex_change_per_year": -2000,  # $ per year (savings in fuel and maintenance)
        "lifespan_years": 10,
        "emission_reduction_percentage": 1.0,  # 100% reduction in direct emissions
        "implementation_time_months": 1,
        "maintenance_cost_percentage": 0.03,  # 3% of CAPEX per year
        "fuel_cost_change_percentage": -0.70,  # 70% decrease in fuel costs
        "typical_implementation_units": 10,  # Typical fleet conversion size
        "applicability": ["vehicle_fleet"]
    },
    
    # Hybrid vehicles
    "hybrid_vehicles": {
        "capex_per_unit": 45000,  # $ per vehicle
        "opex_change_per_year": -1000,  # $ per year
        "lifespan_years": 10,
        "emission_reduction_percentage": 0.40,  # 40% reduction
        "implementation_time_months": 1,
        "maintenance_cost_percentage": 0.04,  # 4% of CAPEX per year
        "fuel_cost_change_percentage": -0.30,  # 30% decrease in fuel costs
        "typical_implementation_units": 10,
        "applicability": ["vehicle_fleet"]
    },
    
    # Renewable natural gas
    "renewable_natural_gas": {
        "capex_per_unit": 100000,  # $ for infrastructure modifications
        "opex_change_per_year": 50000,  # $ per year (higher fuel cost)
        "lifespan_years": 20,
        "emission_reduction_percentage": 0.80,  # 80% reduction
        "implementation_time_months": 4,
        "maintenance_cost_percentage": 0.02,  # 2% of CAPEX per year
        "fuel_cost_change_percentage": 0.50,  # 50% increase in fuel costs
        "typical_implementation_units": 1,
        "applicability": ["natural_gas_boilers", "diesel_generators"]
    },
    
    # Energy management systems
    "energy_management_systems": {
        "capex_per_unit": 150000,  # $ per system
        "opex_change_per_year": -25000,  # $ per year (energy savings)
        "lifespan_years": 10,
        "emission_reduction_percentage": 0.15,  # 15% reduction
        "implementation_time_months": 8,
        "maintenance_cost_percentage": 0.08,  # 8% of CAPEX per year
        "fuel_cost_change_percentage": -0.15,  # 15% decrease in energy costs
        "typical_implementation_units": 1,
        "applicability": ["natural_gas_boilers", "diesel_generators", "vehicle_fleet", "off_road_equipment"]
    },
    
    # Advanced refrigerant leak detection
    "refrigerant_leak_detection": {
        "capex_per_unit": 80000,  # $ per system
        "opex_change_per_year": -5000,  # $ per year (savings in refrigerant)
        "lifespan_years": 8,
        "emission_reduction_percentage": 0.80,  # 80% reduction
        "implementation_time_months": 2,
        "maintenance_cost_percentage": 0.05,  # 5% of CAPEX per year
        "fuel_cost_change_percentage": 0.0,  # No change in fuel costs
        "typical_implementation_units": 1,
        "applicability": ["refrigerant_leaks"]
    },
    
    # Industrial process optimization
    "industrial_process_optimization": {
        "capex_per_unit": 300000,  # $ per implementation
        "opex_change_per_year": -40000,  # $ per year (efficiency savings)
        "lifespan_years": 15,
        "emission_reduction_percentage": 0.25,  # 25% reduction
        "implementation_time_months": 12,
        "maintenance_cost_percentage": 0.03,  # 3% of CAPEX per year
        "fuel_cost_change_percentage": -0.20,  # 20% decrease in energy costs
        "typical_implementation_units": 1,
        "applicability": ["industrial_gas_leaks"]
    },
    
    # Electric off-road equipment
    "electric_off_road_equipment": {
        "capex_per_unit": 120000,  # $ per unit
        "opex_change_per_year": -8000,  # $ per year
        "lifespan_years": 8,
        "emission_reduction_percentage": 1.0,  # 100% reduction in direct emissions
        "implementation_time_months": 2,
        "maintenance_cost_percentage": 0.04,  # 4% of CAPEX per year
        "fuel_cost_change_percentage": -0.60,  # 60% decrease in fuel costs
        "typical_implementation_units": 5,
        "applicability": ["off_road_equipment"]
    }
}

# Implementation timeline for Scenario 2 (Moderate Reduction)
# Percentage of each source category converted by year
MODERATE_REDUCTION_TIMELINE = {
    "natural_gas_boilers": {
        "electric_boilers": {
            2026: 0.10,  # 10% implementation in 2026
            2027: 0.20,  # 20% implementation in 2027 (cumulative)
            2028: 0.30,  # 30% implementation in 2028 (cumulative)
            2029: 0.40,  # 40% implementation in 2029 (cumulative)
            2030: 0.50   # 50% implementation by 2030 (cumulative)
        },
        "high_efficiency_boilers": {
            2026: 0.10,
            2027: 0.20,
            2028: 0.30,
            2029: 0.30,
            2030: 0.30
        }
    },
    "vehicle_fleet": {
        "electric_vehicles": {
            2026: 0.05,
            2027: 0.10,
            2028: 0.15,
            2029: 0.20,
            2030: 0.30
        },
        "hybrid_vehicles": {
            2026: 0.10,
            2027: 0.15,
            2028: 0.20,
            2029: 0.25,
            2030: 0.30
        }
    },
    "diesel_generators": {
        "renewable_natural_gas": {
            2026: 0.05,
            2027: 0.10,
            2028: 0.15,
            2029: 0.20,
            2030: 0.25
        }
    },
    "refrigerant_leaks": {
        "refrigerant_leak_detection": {
            2026: 0.20,
            2027: 0.40,
            2028: 0.60,
            2029: 0.70,
            2030: 0.80
        }
    },
    "industrial_gas_leaks": {
        "industrial_process_optimization": {
            2026: 0.10,
            2027: 0.20,
            2028: 0.30,
            2029: 0.40,
            2030: 0.50
        }
    },
    "off_road_equipment": {
        "electric_off_road_equipment": {
            2026: 0.05,
            2027: 0.10,
            2028: 0.15,
            2029: 0.20,
            2030: 0.25
        }
    }
}

# Get implementation percentage for a specific source, technology, and year
def get_implementation_percentage(source, technology, year):
    """
    Get the implementation percentage for a specific source, technology, and year.
    
    Args:
        source (str): The emission source category
        technology (str): The technology to implement
        year (int): The year to get the implementation percentage for
        
    Returns:
        float: Implementation percentage (0.0 to 1.0)
    """
    if source not in MODERATE_REDUCTION_TIMELINE:
        return 0.0
    
    if technology not in MODERATE_REDUCTION_TIMELINE[source]:
        return 0.0
    
    timeline = MODERATE_REDUCTION_TIMELINE[source][technology]
    years = sorted(timeline.keys())
    
    # No implementation before first year
    if year < years[0]:
        return 0.0
    
    # Implementation complete after last year
    if year >= years[-1]:
        return timeline[years[-1]]
    
    # Find exact year match
    if year in timeline:
        return timeline[year]
    
    # Linear interpolation between years
    for i in range(len(years) - 1):
        if years[i] <= year < years[i + 1]:
            y1, y2 = years[i], years[i + 1]
            p1, p2 = timeline[y1], timeline[y2]
            return p1 + (p2 - p1) * (year - y1) / (y2 - y1)
    
    return 0.0  # Default if not found

# Get applicable technologies for a source
def get_applicable_technologies(source):
    """
    Get a list of technologies applicable to a specific emission source.
    
    Args:
        source (str): The emission source category
        
    Returns:
        list: List of applicable technology names
    """
    applicable = []
    for tech, data in TECHNOLOGY_DATA.items():
        if source in data["applicability"]:
            applicable.append(tech)
    return applicable

# Testing
if __name__ == "__main__":
    # Print applicable technologies for each source
    sources = ["natural_gas_boilers", "vehicle_fleet", "refrigerant_leaks"]
    for source in sources:
        techs = get_applicable_technologies(source)
        print(f"Technologies for {source}: {techs}")
    
    # Print implementation timeline for natural gas boilers
    print("\nElectric boilers implementation timeline:")
    for year in range(2025, 2031):
        pct = get_implementation_percentage("natural_gas_boilers", "electric_boilers", year)
        print(f"Year {year}: {pct*100:.1f}%")