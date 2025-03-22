"""
Module for calculating technology transition impacts on emissions and costs.
This supports the Scope 1 reduction model by providing detailed calculations
for technology implementation effects.
"""
import numpy as np
import pandas as pd

# Import from other modules
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.technology_data import TECHNOLOGY_DATA, get_implementation_percentage

class TechnologyTransitionCalculator:
    """
    Calculator for technology transition impacts on emissions and costs.
    """
    
    def __init__(self, start_year=2025, end_year=2030):
        """
        Initialize the technology transition calculator.
        
        Args:
            start_year (int): First year of implementation
            end_year (int): Last year of implementation
        """
        self.start_year = start_year
        self.end_year = end_year
    
    def calculate_emission_reductions(self, source, baseline_emissions, technologies=None):
        """
        Calculate emission reductions from implementing technologies for a specific source.
        
        Args:
            source (str): Emission source category
            baseline_emissions (dict): Dictionary mapping years to baseline emissions
            technologies (list): List of technologies to apply (if None, use all applicable)
            
        Returns:
            dict: Dictionary with emission reductions by year and technology
        """
        # If no technologies specified, use all applicable ones
        if technologies is None:
            technologies = []
            for tech, data in TECHNOLOGY_DATA.items():
                if source in data["applicability"]:
                    technologies.append(tech)
        
        # Initialize results dictionary
        results = {
            'years': list(range(self.start_year, self.end_year + 1)),
            'baseline_emissions': [],
            'reduced_emissions': [],
            'reductions_by_technology': {tech: [] for tech in technologies}
        }
        
        # Calculate reductions for each year
        for year in results['years']:
            # Get baseline emissions for this year
            if year in baseline_emissions:
                baseline = baseline_emissions[year]
            else:
                # Use last available year if this year not provided
                available_years = sorted(baseline_emissions.keys())
                baseline = baseline_emissions[max(y for y in available_years if y <= year)]
            
            results['baseline_emissions'].append(baseline)
            
            # Initialize reduced emissions to baseline
            reduced = baseline
            
            # Apply each technology
            for tech in technologies:
                # Get implementation percentage for this year
                impl_pct = get_implementation_percentage(source, tech, year)
                
                # Get emission reduction percentage for this technology
                tech_data = TECHNOLOGY_DATA[tech]
                red_pct = tech_data['emission_reduction_percentage']
                
                # Calculate reduction for this technology
                # Note: we apply to the baseline, not the already-reduced amount
                # This handles overlapping technologies appropriately
                reduction = baseline * impl_pct * red_pct
                
                # Store reduction for this technology
                results['reductions_by_technology'][tech].append(reduction)
                
                # Reduce the emissions
                reduced -= reduction
            
            # Ensure we don't go below zero
            reduced = max(0, reduced)
            results['reduced_emissions'].append(reduced)
        
        return results
    
    def calculate_implementation_costs(self, source, source_emissions, technologies=None):
        """
        Calculate costs of implementing technologies for a specific source.
        
        Args:
            source (str): Emission source category
            source_emissions (dict): Dictionary mapping years to source emissions
            technologies (list): List of technologies to apply (if None, use all applicable)
            
        Returns:
            dict: Dictionary with implementation costs by year and technology
        """
        # If no technologies specified, use all applicable ones
        if technologies is None:
            technologies = []
            for tech, data in TECHNOLOGY_DATA.items():
                if source in data["applicability"]:
                    technologies.append(tech)
        
        # Initialize results dictionary
        results = {
            'years': list(range(self.start_year, self.end_year + 1)),
            'total_capex': [],
            'total_opex_change': [],
            'total_cost': [],
            'costs_by_technology': {tech: {'capex': [], 'opex_change': []} for tech in technologies}
        }
        
        # Track cumulative implementation for operational costs
        cumulative_implementation = {tech: 0.0 for tech in technologies}
        
        # Calculate costs for each year
        for year in results['years']:
            # Get source emissions for this year
            if year in source_emissions:
                emissions = source_emissions[year]
            else:
                # Use last available year if this year not provided
                available_years = sorted(source_emissions.keys())
                emissions = source_emissions[max(y for y in available_years if y <= year)]
            
            # Initialize totals for this year
            total_capex = 0.0
            total_opex_change = 0.0
            
            # Apply each technology
            for tech in technologies:
                # Get implementation percentage for this year and previous year
                impl_pct_this_year = get_implementation_percentage(source, tech, year)
                impl_pct_prev_year = get_implementation_percentage(source, tech, year-1)
                impl_pct_delta = max(0, impl_pct_this_year - impl_pct_prev_year)
                
                # Get technology data
                tech_data = TECHNOLOGY_DATA[tech]
                
                # Calculate CAPEX (only for new implementation)
                if impl_pct_delta > 0:
                    # Estimate the scale of implementation based on emissions
                    # We assume a linear relationship between emissions and implementation units
                    emission_scale = emissions / 1000  # 1000 tCO2e is our reference scale
                    
                    # Calculate units to implement
                    typical_units = tech_data['typical_implementation_units']
                    units_to_implement = typical_units * emission_scale * impl_pct_delta
                    
                    # Calculate CAPEX
                    capex = units_to_implement * tech_data['capex_per_unit']
                else:
                    capex = 0.0
                
                # Update cumulative implementation for operational costs
                cumulative_implementation[tech] = impl_pct_this_year
                
                # Calculate OPEX change for all implemented capacity
                if impl_pct_this_year > 0:
                    # Estimate the scale of operation based on emissions
                    emission_scale = emissions / 1000  # 1000 tCO2e is our reference scale
                    
                    # Calculate units in operation
                    typical_units = tech_data['typical_implementation_units']
                    units_in_operation = typical_units * emission_scale * impl_pct_this_year
                    
                    # Calculate OPEX change
                    opex_change = units_in_operation * tech_data['opex_change_per_year']
                    
                    # Add maintenance costs
                    maintenance_pct = tech_data['maintenance_cost_percentage']
                    years_implemented = year - self.start_year + 1
                    avg_age = years_implemented / 2  # Simple approximation
                    
                    # Maintenance increases with age
                    maintenance_factor = 1 + (avg_age / 10)  # 10% increase every 10 years
                    maintenance_cost = capex * maintenance_pct * maintenance_factor
                    
                    opex_change += maintenance_cost
                else:
                    opex_change = 0.0
                
                # Store costs for this technology
                results['costs_by_technology'][tech]['capex'].append(capex)
                results['costs_by_technology'][tech]['opex_change'].append(opex_change)
                
                # Add to totals
                total_capex += capex
                total_opex_change += opex_change
            
            # Store totals for this year
            results['total_capex'].append(total_capex)
            results['total_opex_change'].append(total_opex_change)
            results['total_cost'].append(total_capex + total_opex_change)
        
        return results
    
    def generate_adoption_curve(self, source, technology, start_pct=0.0, end_pct=1.0, curve_type="linear"):
        """
        Generate an adoption curve for a technology.
        
        Args:
            source (str): Emission source category
            technology (str): Technology to generate curve for
            start_pct (float): Starting implementation percentage
            end_pct (float): Ending implementation percentage
            curve_type (str): Type of curve ("linear", "s_curve", "exponential", "delayed")
            
        Returns:
            dict: Dictionary mapping years to implementation percentages
        """
        years = list(range(self.start_year, self.end_year + 1))
        num_years = len(years)
        
        # Initialize with zeros
        adoption_curve = {year: 0.0 for year in years}
        
        if curve_type == "linear":
            # Linear adoption
            for i, year in enumerate(years):
                if num_years > 1:
                    adoption_curve[year] = start_pct + (end_pct - start_pct) * (i / (num_years - 1))
                else:
                    adoption_curve[year] = end_pct
                    
        elif curve_type == "s_curve":
            # S-curve (logistic function)
            for i, year in enumerate(years):
                # Convert to range [-6, 6] for logistic function
                x = -6 + 12 * (i / (num_years - 1)) if num_years > 1 else 6
                # Logistic function: 1 / (1 + e^-x)
                s_value = 1 / (1 + np.exp(-x))
                # Scale to range [start_pct, end_pct]
                adoption_curve[year] = start_pct + (end_pct - start_pct) * s_value
                
        elif curve_type == "exponential":
            # Exponential adoption (faster at the beginning)
            for i, year in enumerate(years):
                # Square root function for faster early adoption
                if num_years > 1:
                    expo_value = np.sqrt(i / (num_years - 1))
                else:
                    expo_value = 1.0
                # Scale to range [start_pct, end_pct]
                adoption_curve[year] = start_pct + (end_pct - start_pct) * expo_value
                
        elif curve_type == "delayed":
            # Delayed adoption (slower at the beginning)
            for i, year in enumerate(years):
                # Squared function for delayed adoption
                if num_years > 1:
                    delay_value = (i / (num_years - 1)) ** 2
                else:
                    delay_value = 1.0
                # Scale to range [start_pct, end_pct]
                adoption_curve[year] = start_pct + (end_pct - start_pct) * delay_value
                
        else:
            raise ValueError(f"Unknown curve type: {curve_type}")
        
        return adoption_curve

if __name__ == "__main__":
    # Example usage
    calculator = TechnologyTransitionCalculator(2025, 2030)
    
    # Create sample baseline emissions
    baseline_emissions = {
        2025: 5000,  # tCO2e
        2026: 5100,
        2027: 5200,
        2028: 5300,
        2029: 5400,
        2030: 5500
    }
    
    # Calculate reductions for natural gas boilers
    reductions = calculator.calculate_emission_reductions(
        "natural_gas_boilers", 
        baseline_emissions,
        ["electric_boilers", "high_efficiency_boilers"]
    )
    
    print("Emission Reductions:")
    for i, year in enumerate(reductions['years']):
        print(f"Year {year}: {reductions['baseline_emissions'][i] - reductions['reduced_emissions'][i]:.1f} tCO2e")
    
    # Calculate implementation costs
    costs = calculator.calculate_implementation_costs(
        "natural_gas_boilers", 
        baseline_emissions,
        ["electric_boilers", "high_efficiency_boilers"]
    )
    
    print("\nImplementation Costs:")
    for i, year in enumerate(costs['years']):
        print(f"Year {year}: CAPEX=${costs['total_capex'][i]:,.0f}, OPEX change=${costs['total_opex_change'][i]:,.0f}")
    
    # Generate adoption curves
    s_curve = calculator.generate_adoption_curve(
        "natural_gas_boilers", 
        "electric_boilers",
        0.0, 0.5, 
        "s_curve"
    )
    
    print("\nS-Curve Adoption:")
    for year, pct in s_curve.items():
        print(f"Year {year}: {pct*100:.1f}%")