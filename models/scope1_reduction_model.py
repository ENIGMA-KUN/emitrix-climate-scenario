"""
Core model for Scope 1 emissions reduction scenario analysis.
This module implements the Moderate Reduction scenario (Scenario 2)
which targets 30% reduction in Scope 1 emissions by 2030.
"""
import numpy as np
import pandas as pd
from datetime import datetime

# Import from other modules
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.historical_emissions import get_sample_historical_data
from data.emission_factors import get_carbon_price
from data.technology_data import (
    TECHNOLOGY_DATA, 
    get_implementation_percentage,
    get_applicable_technologies
)

class Scope1ReductionModel:
    """
    Model for analyzing Scope 1 emissions reduction scenarios.
    """
    
    def __init__(self, 
                 reduction_target=0.3,  # 30% reduction
                 target_year=2030,  
                 start_year=2025,
                 carbon_price_start=30,  # $/tCO2e
                 carbon_price_increase=0.05,  # 5% annual increase
                 discount_rate=0.07):  # 7% discount rate
        """
        Initialize the reduction model with scenario parameters.
        
        Args:
            reduction_target (float): Target reduction percentage (0.0 to 1.0)
            target_year (int): Year to achieve the reduction target
            start_year (int): Year to start implementing reduction measures
            carbon_price_start (float): Starting carbon price in $/tCO2e
            carbon_price_increase (float): Annual carbon price increase rate
            discount_rate (float): Discount rate for financial calculations
        """
        self.reduction_target = reduction_target
        self.target_year = target_year
        self.start_year = start_year
        self.carbon_price_start = carbon_price_start
        self.carbon_price_increase = carbon_price_increase
        self.discount_rate = discount_rate
        
        # Initialize data containers
        self.historical_data = None
        self.projection_data = None
        self.financial_data = None
        
    def load_sample_data(self):
        """
        Load sample historical emissions data.
        """
        yearly_data, _ = get_sample_historical_data()
        self.historical_data = yearly_data
        return yearly_data
    
    def load_historical_data(self, data):
        """
        Load external historical emissions data.
        
        Args:
            data (pandas.DataFrame): Historical emissions data
        """
        self.historical_data = data
        return data
    
    def project_bau_emissions(self, end_year=2030):
        """
        Project business-as-usual (BAU) emissions based on historical data.
        
        Args:
            end_year (int): Last year to project emissions for
            
        Returns:
            pandas.DataFrame: DataFrame with projected BAU emissions
        """
        if self.historical_data is None:
            raise ValueError("Historical data not loaded. Call load_historical_data() first.")
        
        # Calculate average annual growth rate from historical data
        hist_years = self.historical_data['year'].values
        hist_emissions = self.historical_data['emissions'].values
        
        start_emissions = hist_emissions[0]
        end_emissions = hist_emissions[-1]
        num_years = hist_years[-1] - hist_years[0]
        
        # Calculate compound annual growth rate
        growth_rate = (end_emissions / start_emissions) ** (1 / num_years) - 1
        
        # Project BAU emissions
        projection_years = list(range(hist_years[-1] + 1, end_year + 1))
        last_year_data = self.historical_data.iloc[-1].copy()
        
        projection_data = []
        
        # Add last historical year
        projection_data.append(last_year_data.to_dict())
        
        # Project future years
        for year in projection_years:
            new_row = last_year_data.copy()
            new_row['year'] = year
            
            # Calculate growth factor for this year
            years_from_last = year - hist_years[-1]
            growth_factor = (1 + growth_rate) ** years_from_last
            
            # Apply growth to total emissions
            new_row['emissions'] = last_year_data['emissions'] * growth_factor
            
            # Apply same growth to source categories
            source_columns = [col for col in self.historical_data.columns 
                             if col not in ['year', 'emissions']]
            
            for source in source_columns:
                new_row[source] = last_year_data[source] * growth_factor
            
            projection_data.append(new_row)
        
        # Convert to DataFrame
        self.projection_data = pd.DataFrame(projection_data)
        return self.projection_data
    
    def apply_reduction_measures(self):
        """
        Apply emission reduction measures based on the technology implementation timeline.
        
        Returns:
            pandas.DataFrame: DataFrame with reduced emissions
        """
        if self.projection_data is None:
            raise ValueError("BAU projection not done. Call project_bau_emissions() first.")
        
        # Create a copy of the projection data
        reduced_data = self.projection_data.copy()
        
        # Add columns for reduced emissions
        reduced_data['reduced_emissions'] = reduced_data['emissions'].copy()
        
        source_columns = [col for col in reduced_data.columns 
                         if col not in ['year', 'emissions', 'reduced_emissions']]
        
        for source in source_columns:
            reduced_data[f'reduced_{source}'] = reduced_data[source].copy()
        
        # Apply technology reductions for each year and source
        implementation_years = range(self.start_year, self.target_year + 1)
        
        for _, row in reduced_data.iterrows():
            year = row['year']
            
            if year < self.start_year:
                continue
                
            for source in source_columns:
                # Get applicable technologies
                technologies = get_applicable_technologies(source)
                
                # Apply each technology's reduction
                for tech in technologies:
                    # Get implementation percentage for this year
                    impl_pct = get_implementation_percentage(source, tech, year)
                    
                    if impl_pct > 0:
                        # Get emission reduction percentage for this technology
                        red_pct = TECHNOLOGY_DATA[tech]['emission_reduction_percentage']
                        
                        # Calculate emission reduction
                        reduction = row[source] * impl_pct * red_pct
                        
                        # Apply reduction
                        reduced_data.loc[reduced_data['year'] == year, f'reduced_{source}'] -= reduction
        
        # Recalculate total reduced emissions
        for _, row in reduced_data.iterrows():
            year = row['year']
            
            # Sum up all reduced source emissions
            total_reduced = sum(row[f'reduced_{source}'] for source in source_columns)
            
            # Update total reduced emissions
            reduced_data.loc[reduced_data['year'] == year, 'reduced_emissions'] = total_reduced
        
        return reduced_data
    
    def calculate_financial_impact(self, reduced_data):
        """
        Calculate financial impacts of emission reductions.
        
        Args:
            reduced_data (pandas.DataFrame): DataFrame with reduced emissions
            
        Returns:
            pandas.DataFrame: DataFrame with financial calculations
        """
        # Create financial data structure
        financial_data = []
        
        source_columns = [col for col in self.projection_data.columns 
                         if col not in ['year', 'emissions']]
        
        for _, row in reduced_data.iterrows():
            year = row['year']
            
            if year < self.start_year:
                continue
                
            financial_row = {
                'year': year,
                'bau_emissions': row['emissions'],
                'reduced_emissions': row['reduced_emissions'],
                'emissions_avoided': row['emissions'] - row['reduced_emissions'],
                'implementation_cost': 0.0,
                'operational_change': 0.0,
                'carbon_price': get_carbon_price(year),
                'carbon_cost_bau': 0.0,
                'carbon_cost_reduced': 0.0,
                'carbon_savings': 0.0,
                'net_benefit': 0.0,
                'npv_factor': 1 / ((1 + self.discount_rate) ** (year - self.start_year)),
                'discounted_net_benefit': 0.0
            }
            
            # Calculate carbon costs
            financial_row['carbon_cost_bau'] = row['emissions'] * financial_row['carbon_price']
            financial_row['carbon_cost_reduced'] = row['reduced_emissions'] * financial_row['carbon_price']
            financial_row['carbon_savings'] = financial_row['carbon_cost_bau'] - financial_row['carbon_cost_reduced']
            
            # Calculate technology implementation costs
            for source in source_columns:
                # Get applicable technologies
                technologies = get_applicable_technologies(source)
                
                # Apply each technology's costs
                for tech in technologies:
                    # Get implementation percentage for this year only (not cumulative)
                    impl_pct_this_year = get_implementation_percentage(source, tech, year)
                    impl_pct_prev_year = get_implementation_percentage(source, tech, year-1)
                    impl_pct_delta = max(0, impl_pct_this_year - impl_pct_prev_year)
                    
                    if impl_pct_delta > 0:
                        # Calculate source emissions scale for implementation
                        base_emissions = self.projection_data[self.projection_data['year'] == year]['emissions'].values[0]
                        source_fraction = row[source] / base_emissions
                        
                        # Calculate implementation cost
                        tech_data = TECHNOLOGY_DATA[tech]
                        typical_units = tech_data['typical_implementation_units']
                        capex_per_unit = tech_data['capex_per_unit']
                        
                        # Scale implementation by source fraction and implementation percentage
                        implementation_cost = (
                            typical_units * capex_per_unit * source_fraction * impl_pct_delta
                        )
                        
                        financial_row['implementation_cost'] += implementation_cost
                    
                    # Calculate operational changes for all implemented technologies
                    impl_pct_cumulative = impl_pct_this_year
                    
                    if impl_pct_cumulative > 0:
                        # Calculate operational change
                        tech_data = TECHNOLOGY_DATA[tech]
                        opex_change = tech_data['opex_change_per_year']
                        
                        # Scale by source fraction and implementation percentage
                        base_emissions = self.projection_data[self.projection_data['year'] == year]['emissions'].values[0]
                        source_fraction = row[source] / base_emissions
                        operational_change = opex_change * source_fraction * impl_pct_cumulative
                        
                        financial_row['operational_change'] += operational_change
            
            # Calculate net benefit and NPV
            financial_row['net_benefit'] = (
                financial_row['carbon_savings'] - 
                financial_row['implementation_cost'] - 
                financial_row['operational_change']
            )
            
            financial_row['discounted_net_benefit'] = (
                financial_row['net_benefit'] * financial_row['npv_factor']
            )
            
            financial_data.append(financial_row)
        
        self.financial_data = pd.DataFrame(financial_data)
        return self.financial_data
    
    def run_scenario(self):
        """
        Run the complete scenario analysis.
        
        Returns:
            dict: Dictionary with scenario results
        """
        # Load sample data if not already loaded
        if self.historical_data is None:
            self.load_sample_data()
        
        # Project BAU emissions
        projection_data = self.project_bau_emissions(self.target_year)
        
        # Apply reduction measures
        reduced_data = self.apply_reduction_measures()
        
        # Calculate financial impact
        financial_data = self.calculate_financial_impact(reduced_data)
        
        # Calculate scenario metrics
        metrics = self.calculate_scenario_metrics(reduced_data, financial_data)
        
        # Prepare results dictionary
        results = {
            'scenario_name': 'Moderate Reduction',
            'reduction_target': self.reduction_target,
            'target_year': self.target_year,
            'historical_data': self.historical_data.to_dict(orient='records'),
            'projection_data': projection_data.to_dict(orient='records'),
            'reduced_data': reduced_data.to_dict(orient='records'),
            'financial_data': financial_data.to_dict(orient='records'),
            'metrics': metrics
        }
        
        return results
    
    def calculate_scenario_metrics(self, reduced_data, financial_data):
        """
        Calculate key metrics for the scenario.
        
        Args:
            reduced_data (pandas.DataFrame): DataFrame with reduced emissions
            financial_data (pandas.DataFrame): DataFrame with financial calculations
            
        Returns:
            dict: Dictionary with scenario metrics
        """
        # Extract target year data
        target_data = reduced_data[reduced_data['year'] == self.target_year]
        first_year_data = reduced_data[reduced_data['year'] == self.start_year]
        
        if len(target_data) == 0 or len(first_year_data) == 0:
            raise ValueError(f"Target year {self.target_year} or start year {self.start_year} not found in projection data.")
        
        # Calculate reduction achieved
        bau_emissions_target = target_data['emissions'].values[0]
        reduced_emissions_target = target_data['reduced_emissions'].values[0]
        baseline_emissions = first_year_data['emissions'].values[0]
        
        reduction_amount = bau_emissions_target - reduced_emissions_target
        reduction_percentage = (bau_emissions_target - reduced_emissions_target) / bau_emissions_target
        reduction_from_baseline = (baseline_emissions - reduced_emissions_target) / baseline_emissions
        
        # Calculate financial metrics
        total_implementation_cost = financial_data['implementation_cost'].sum()
        total_operational_change = financial_data['operational_change'].sum()
        total_carbon_savings = financial_data['carbon_savings'].sum()
        net_present_value = financial_data['discounted_net_benefit'].sum()
        
        # Calculate ROI and payback
        if total_implementation_cost > 0:
            roi = (total_carbon_savings - total_operational_change) / total_implementation_cost
        else:
            roi = float('inf')
            
        # Simple payback period calculation
        annual_benefit = (total_carbon_savings - total_operational_change) / len(financial_data)
        if annual_benefit > 0:
            payback_years = total_implementation_cost / annual_benefit
        else:
            payback_years = float('inf')
        
        # Calculate marginal abatement cost
        total_emissions_avoided = financial_data['emissions_avoided'].sum()
        if total_emissions_avoided > 0:
            marginal_abatement_cost = (total_implementation_cost + total_operational_change) / total_emissions_avoided
        else:
            marginal_abatement_cost = float('inf')
        
        # Calculate average annual implementation cost
        annual_implementation_cost = total_implementation_cost / len(financial_data)
        
        # Compile metrics
        metrics = {
            'reduction_achieved_tco2e': reduction_amount,
            'reduction_percentage': reduction_percentage,
            'reduction_from_baseline': reduction_from_baseline,
            'total_implementation_cost': total_implementation_cost,
            'total_operational_change': total_operational_change,
            'total_carbon_savings': total_carbon_savings,
            'net_present_value': net_present_value,
            'return_on_investment': roi,
            'payback_years': payback_years,
            'marginal_abatement_cost': marginal_abatement_cost,
            'annual_implementation_cost': annual_implementation_cost,
            'target_year_bau_emissions': bau_emissions_target,
            'target_year_reduced_emissions': reduced_emissions_target
        }
        
        return metrics

if __name__ == "__main__":
    # Example usage
    model = Scope1ReductionModel()
    results = model.run_scenario()
    
    # Print key metrics
    print(f"Scenario: {results['scenario_name']}")
    print(f"Reduction target: {results['reduction_target']*100}% by {results['target_year']}")
    
    metrics = results['metrics']
    print(f"\nReduction achieved: {metrics['reduction_percentage']*100:.1f}% ({metrics['reduction_achieved_tco2e']:.1f} tCO2e)")
    print(f"Implementation cost: ${metrics['total_implementation_cost']:,.0f}")
    print(f"NPV: ${metrics['net_present_value']:,.0f}")
    print(f"ROI: {metrics['return_on_investment']*100:.1f}%")
    print(f"Payback period: {metrics['payback_years']:.1f} years")