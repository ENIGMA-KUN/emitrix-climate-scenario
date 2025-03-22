"""
Module for calculating financial impacts of emission reduction scenarios.
This module implements detailed financial analysis for the Scenario 2 
(Moderate Reduction) pathway.
"""
import numpy as np
import pandas as pd
from datetime import datetime

# Import from other modules
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.emission_factors import get_carbon_price

class FinancialImpactModel:
    """
    Model for analyzing financial impacts of emission reduction scenarios.
    """
    
    def __init__(self, 
                 start_year=2025,
                 end_year=2030,
                 carbon_price_trajectory="medium",
                 discount_rate=0.07,
                 tax_rate=0.21,
                 include_tax_benefits=True):
        """
        Initialize the financial impact model.
        
        Args:
            start_year (int): First year for financial analysis
            end_year (int): Last year for financial analysis
            carbon_price_trajectory (str): Carbon price scenario ("low", "medium", "high")
            discount_rate (float): Discount rate for NPV calculations
            tax_rate (float): Corporate tax rate for tax benefit calculations
            include_tax_benefits (bool): Whether to include tax benefits in calculations
        """
        self.start_year = start_year
        self.end_year = end_year
        self.carbon_price_trajectory = carbon_price_trajectory
        self.discount_rate = discount_rate
        self.tax_rate = tax_rate
        self.include_tax_benefits = include_tax_benefits
        
    def calculate_carbon_costs(self, emissions_data):
        """
        Calculate carbon costs for BAU and reduced emissions scenarios.
        
        Args:
            emissions_data (dict): Dictionary with BAU and reduced emissions by year
            
        Returns:
            pandas.DataFrame: DataFrame with carbon cost calculations
        """
        carbon_costs = []
        
        for year in range(self.start_year, self.end_year + 1):
            if year not in emissions_data:
                continue
                
            carbon_price = get_carbon_price(year, self.carbon_price_trajectory)
            bau_emissions = emissions_data[year]['bau']
            reduced_emissions = emissions_data[year]['reduced']
            
            row = {
                'year': year,
                'carbon_price': carbon_price,
                'bau_emissions': bau_emissions,
                'reduced_emissions': reduced_emissions,
                'bau_carbon_cost': bau_emissions * carbon_price,
                'reduced_carbon_cost': reduced_emissions * carbon_price,
                'carbon_cost_savings': (bau_emissions - reduced_emissions) * carbon_price
            }
            
            carbon_costs.append(row)
        
        return pd.DataFrame(carbon_costs)
    
    def calculate_implementation_costs(self, cost_data):
        """
        Calculate implementation costs and operational changes.
        
        Args:
            cost_data (dict): Dictionary with implementation costs by year
            
        Returns:
            pandas.DataFrame: DataFrame with implementation cost calculations
        """
        implementation_costs = []
        
        for year in range(self.start_year, self.end_year + 1):
            if year not in cost_data:
                continue
                
            capex = cost_data[year]['capex']
            opex_change = cost_data[year]['opex_change']
            
            # Calculate depreciation (straight-line over 10 years)
            depreciation = capex / 10
            
            # Calculate tax benefits if applicable
            if self.include_tax_benefits:
                # Tax deduction on depreciation and opex
                tax_benefit = (depreciation + max(0, opex_change)) * self.tax_rate
            else:
                tax_benefit = 0
            
            row = {
                'year': year,
                'capex': capex,
                'opex_change': opex_change,
                'depreciation': depreciation,
                'tax_benefit': tax_benefit,
                'net_implementation_cost': capex + opex_change - tax_benefit
            }
            
            implementation_costs.append(row)
        
        return pd.DataFrame(implementation_costs)
    
    def calculate_npv_and_roi(self, carbon_costs_df, implementation_costs_df):
        """
        Calculate NPV, ROI, and payback period.
        
        Args:
            carbon_costs_df (pandas.DataFrame): DataFrame with carbon costs
            implementation_costs_df (pandas.DataFrame): DataFrame with implementation costs
            
        Returns:
            dict: Dictionary with financial metrics
        """
        # Merge data on year
        financial_df = pd.merge(carbon_costs_df, implementation_costs_df, on='year')
        
        # Calculate annual net benefit
        financial_df['annual_net_benefit'] = (
            financial_df['carbon_cost_savings'] - 
            financial_df['net_implementation_cost']
        )
        
        # Calculate NPV factor
        financial_df['discount_factor'] = [
            1 / ((1 + self.discount_rate) ** (year - self.start_year)) 
            for year in financial_df['year']
        ]
        
        # Calculate discounted benefits and costs
        financial_df['discounted_benefit'] = (
            financial_df['carbon_cost_savings'] * financial_df['discount_factor']
        )
        
        financial_df['discounted_cost'] = (
            financial_df['net_implementation_cost'] * financial_df['discount_factor']
        )
        
        financial_df['discounted_net_benefit'] = (
            financial_df['annual_net_benefit'] * financial_df['discount_factor']
        )
        
        # Calculate cumulative values for payback period
        financial_df['cumulative_cost'] = financial_df['net_implementation_cost'].cumsum()
        financial_df['cumulative_benefit'] = financial_df['carbon_cost_savings'].cumsum()
        financial_df['cumulative_net'] = financial_df['cumulative_benefit'] - financial_df['cumulative_cost']
        
        # Calculate financial metrics
        total_discounted_benefit = financial_df['discounted_benefit'].sum()
        total_discounted_cost = financial_df['discounted_cost'].sum()
        npv = financial_df['discounted_net_benefit'].sum()
        
        # Calculate ROI
        if total_discounted_cost > 0:
            roi = total_discounted_benefit / total_discounted_cost - 1
        else:
            roi = float('inf')
        
        # Calculate payback period
        payback_year = None
        for i, row in financial_df.iterrows():
            if row['cumulative_net'] >= 0:
                payback_year = row['year']
                break
        
        if payback_year is None:
            payback_period = float('inf')
        else:
            # Interpolate for more precise payback period
            if payback_year > self.start_year:
                prev_year_data = financial_df[financial_df['year'] == payback_year - 1]
                if len(prev_year_data) > 0:
                    prev_net = prev_year_data['cumulative_net'].values[0]
                    current_net = financial_df[financial_df['year'] == payback_year]['cumulative_net'].values[0]
                    
                    # Linear interpolation
                    fraction = -prev_net / (current_net - prev_net)
                    payback_period = payback_year - 1 + fraction
                else:
                    payback_period = payback_year - self.start_year + 1
            else:
                payback_period = payback_year - self.start_year + 1
        
        # Calculate internal rate of return (IRR)
        cash_flows = [-financial_df['net_implementation_cost'].iloc[0]]  # Initial investment
        cash_flows.extend(financial_df['annual_net_benefit'].iloc[1:].values)
        
        try:
            irr = np.irr(cash_flows)
        except Exception:
            # IRR calculation may fail if no solution exists
            irr = None
        
        metrics = {
            'npv': npv,
            'roi': roi,
            'payback_period': payback_period,
            'irr': irr,
            'total_discounted_benefit': total_discounted_benefit,
            'total_discounted_cost': total_discounted_cost,
            'benefit_cost_ratio': total_discounted_benefit / total_discounted_cost if total_discounted_cost > 0 else float('inf'),
            'financial_data': financial_df.to_dict(orient='records')
        }
        
        return metrics
    
    def calculate_marginal_abatement_cost(self, emissions_data, cost_data):
        """
        Calculate marginal abatement cost curve.
        
        Args:
            emissions_data (dict): Dictionary with BAU and reduced emissions by year
            cost_data (dict): Dictionary with implementation costs by year
            
        Returns:
            dict: Dictionary with marginal abatement cost data
        """
        # Calculate total emissions reduction
        total_emissions_reduction = sum(
            emissions_data[year]['bau'] - emissions_data[year]['reduced']
            for year in range(self.start_year, self.end_year + 1)
            if year in emissions_data
        )
        
        # Calculate total net implementation cost
        total_cost = sum(
            cost_data[year]['capex'] + cost_data[year]['opex_change']
            for year in range(self.start_year, self.end_year + 1)
            if year in cost_data
        )
        
        # Calculate marginal abatement cost
        if total_emissions_reduction > 0:
            mac = total_cost / total_emissions_reduction
        else:
            mac = float('inf')
        
        # Create data for MAC curve
        mac_data = {
            'total_emissions_reduction': total_emissions_reduction,
            'total_cost': total_cost,
            'marginal_abatement_cost': mac
        }
        
        return mac_data
    
    def run_financial_analysis(self, emissions_data, cost_data):
        """
        Run the complete financial analysis.
        
        Args:
            emissions_data (dict): Dictionary with BAU and reduced emissions by year
            cost_data (dict): Dictionary with implementation costs by year
            
        Returns:
            dict: Dictionary with financial analysis results
        """
        # Calculate carbon costs
        carbon_costs_df = self.calculate_carbon_costs(emissions_data)
        
        # Calculate implementation costs
        implementation_costs_df = self.calculate_implementation_costs(cost_data)
        
        # Calculate NPV and ROI
        financial_metrics = self.calculate_npv_and_roi(carbon_costs_df, implementation_costs_df)
        
        # Calculate marginal abatement cost
        mac_data = self.calculate_marginal_abatement_cost(emissions_data, cost_data)
        
        # Compile results
        results = {
            'carbon_costs': carbon_costs_df.to_dict(orient='records'),
            'implementation_costs': implementation_costs_df.to_dict(orient='records'),
            'financial_metrics': financial_metrics,
            'marginal_abatement_cost': mac_data
        }
        
        return results
    
    def perform_sensitivity_analysis(self, emissions_data, cost_data, 
                                    carbon_price_variations=None,
                                    discount_rate_variations=None,
                                    implementation_cost_variations=None):
        """
        Perform sensitivity analysis on key parameters.
        
        Args:
            emissions_data (dict): Dictionary with BAU and reduced emissions by year
            cost_data (dict): Dictionary with implementation costs by year
            carbon_price_variations (list): List of carbon price variation factors
            discount_rate_variations (list): List of discount rate alternatives
            implementation_cost_variations (list): List of implementation cost variation factors
            
        Returns:
            dict: Dictionary with sensitivity analysis results
        """
        # Default variations if none provided
        if carbon_price_variations is None:
            carbon_price_variations = [0.5, 0.75, 1.0, 1.25, 1.5]
            
        if discount_rate_variations is None:
            discount_rate_variations = [0.03, 0.05, 0.07, 0.1, 0.12]
            
        if implementation_cost_variations is None:
            implementation_cost_variations = [0.7, 0.85, 1.0, 1.15, 1.3]
        
        results = {
            'base_case': None,
            'carbon_price_sensitivity': [],
            'discount_rate_sensitivity': [],
            'implementation_cost_sensitivity': []
        }
        
        # Run base case analysis
        base_model = FinancialImpactModel(
            self.start_year, 
            self.end_year, 
            self.carbon_price_trajectory,
            self.discount_rate,
            self.tax_rate,
            self.include_tax_benefits
        )
        
        base_results = base_model.run_financial_analysis(emissions_data, cost_data)
        results['base_case'] = base_results
        
        # Run carbon price sensitivity
        for factor in carbon_price_variations:
            # Create modified carbon price trajectory
            if factor != 1.0:
                # Adjust emissions data with new carbon prices
                modified_emissions = {
                    year: {
                        'bau': data['bau'],
                        'reduced': data['reduced']
                    }
                    for year, data in emissions_data.items()
                }
                
                # Create model with adjusted trajectory
                model = FinancialImpactModel(
                    self.start_year, 
                    self.end_year, 
                    self.carbon_price_trajectory,  # Same trajectory name, we'll adjust in the calculation
                    self.discount_rate,
                    self.tax_rate,
                    self.include_tax_benefits
                )
                
                # Override carbon price calculation in calculate_carbon_costs
                def calculate_carbon_costs_override(emissions_data):
                    carbon_costs = []
                    
                    for year in range(model.start_year, model.end_year + 1):
                        if year not in emissions_data:
                            continue
                            
                        carbon_price = get_carbon_price(year, model.carbon_price_trajectory) * factor
                        bau_emissions = emissions_data[year]['bau']
                        reduced_emissions = emissions_data[year]['reduced']
                        
                        row = {
                            'year': year,
                            'carbon_price': carbon_price,
                            'bau_emissions': bau_emissions,
                            'reduced_emissions': reduced_emissions,
                            'bau_carbon_cost': bau_emissions * carbon_price,
                            'reduced_carbon_cost': reduced_emissions * carbon_price,
                            'carbon_cost_savings': (bau_emissions - reduced_emissions) * carbon_price
                        }
                        
                        carbon_costs.append(row)
                    
                    return pd.DataFrame(carbon_costs)
                
                # Replace the method with our override
                model.calculate_carbon_costs = calculate_carbon_costs_override.__get__(model, FinancialImpactModel)
                
                # Run analysis with modified data
                modified_results = model.run_financial_analysis(modified_emissions, cost_data)
                
                # Store results
                results['carbon_price_sensitivity'].append({
                    'factor': factor,
                    'description': f"Carbon price {factor*100:.0f}% of base case",
                    'results': modified_results
                })
        
        # Run discount rate sensitivity
        for rate in discount_rate_variations:
            if rate != self.discount_rate:
                # Create model with adjusted discount rate
                model = FinancialImpactModel(
                    self.start_year, 
                    self.end_year, 
                    self.carbon_price_trajectory,
                    rate,  # Modified discount rate
                    self.tax_rate,
                    self.include_tax_benefits
                )
                
                # Run analysis
                modified_results = model.run_financial_analysis(emissions_data, cost_data)
                
                # Store results
                results['discount_rate_sensitivity'].append({
                    'rate': rate,
                    'description': f"Discount rate {rate*100:.1f}%",
                    'results': modified_results
                })
        
        # Run implementation cost sensitivity
        for factor in implementation_cost_variations:
            if factor != 1.0:
                # Adjust cost data
                modified_costs = {
                    year: {
                        'capex': data['capex'] * factor,
                        'opex_change': data['opex_change'] * factor
                    }
                    for year, data in cost_data.items()
                }
                
                # Run analysis with modified costs
                modified_results = self.run_financial_analysis(emissions_data, modified_costs)
                
                # Store results
                results['implementation_cost_sensitivity'].append({
                    'factor': factor,
                    'description': f"Implementation costs {factor*100:.0f}% of base case",
                    'results': modified_results
                })
        
        return results

if __name__ == "__main__":
    # Example usage
    model = FinancialImpactModel()
    
    # Create sample data
    emissions_data = {
        2025: {'bau': 5000, 'reduced': 5000},  # No reduction in first year
        2026: {'bau': 5100, 'reduced': 4950},
        2027: {'bau': 5200, 'reduced': 4800},
        2028: {'bau': 5300, 'reduced': 4600},
        2029: {'bau': 5400, 'reduced': 4350},
        2030: {'bau': 5500, 'reduced': 4000}   # ~27% reduction by 2030
    }
    
    cost_data = {
        2025: {'capex': 100000, 'opex_change': 0},      # Initial investment
        2026: {'capex': 200000, 'opex_change': -10000}, # Savings in opex
        2027: {'capex': 250000, 'opex_change': -20000},
        2028: {'capex': 300000, 'opex_change': -30000},
        2029: {'capex': 350000, 'opex_change': -40000},
        2030: {'capex': 400000, 'opex_change': -50000}
    }
    
    # Run analysis
    results = model.run_financial_analysis(emissions_data, cost_data)
    
    # Print key metrics
    metrics = results['financial_metrics']
    print(f"NPV: ${metrics['npv']:,.0f}")
    print(f"ROI: {metrics['roi']*100:.1f}%")
    print(f"Payback period: {metrics['payback_period']:.1f} years")
    print(f"Benefit-cost ratio: {metrics['benefit_cost_ratio']:.2f}")
    
    if metrics['irr'] is not None:
        print(f"IRR: {metrics['irr']*100:.1f}%")
    else:
        print("IRR: Not calculable")
    
    mac = results['marginal_abatement_cost']
    print(f"Marginal abatement cost: ${mac['marginal_abatement_cost']:.1f}/tCO2e")