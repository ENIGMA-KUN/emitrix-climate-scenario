"""
Module for generating historical emissions data for scenario analysis.
Provides sample data structures that match the emission source categories 
from the Scope 1 monitoring system.
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

class HistoricalEmissionsGenerator:
    """
    Generates historical emissions data for use in scenario analysis.
    """
    
    def __init__(self, start_year=2020, end_year=2024, 
                 baseline_emissions=10000, random_seed=42):
        """
        Initialize the emissions generator.
        
        Args:
            start_year (int): First year of historical data
            end_year (int): Last year of historical data
            baseline_emissions (float): Base emissions in tCO2e for start_year
            random_seed (int): Seed for reproducible random variations
        """
        self.start_year = start_year
        self.end_year = end_year
        self.baseline_emissions = baseline_emissions
        self.random_seed = random_seed
        np.random.seed(random_seed)
        
    def generate_yearly_emissions(self, growth_rate=0.02, volatility=0.05):
        """
        Generate yearly total emissions with a growth trend and random variations.
        
        Args:
            growth_rate (float): Annual growth rate for emissions
            volatility (float): Random variation scale
            
        Returns:
            pandas.DataFrame: DataFrame with yearly emissions data
        """
        years = range(self.start_year, self.end_year + 1)
        emissions = []
        
        current = self.baseline_emissions
        for year in years:
            # Add random variation to the growth rate
            variation = 1 + growth_rate + np.random.normal(0, volatility)
            current *= variation
            emissions.append(current)
        
        df = pd.DataFrame({
            'year': years,
            'emissions': emissions
        })
        
        return df
    
    def generate_emissions_by_source(self, yearly_emissions):
        """
        Break down total emissions by source category based on the 
        Scope 1 monitoring system categories.
        
        Args:
            yearly_emissions (pandas.DataFrame): DataFrame with yearly total emissions
            
        Returns:
            pandas.DataFrame: DataFrame with emissions broken down by source category
        """
        # Source category distribution based on the Scope 1 monitoring documents
        source_distribution = {
            # Category 1: Stationary Combustion
            'natural_gas_boilers': 0.40,     # 40% of emissions
            'diesel_generators': 0.10,       # 10% of emissions
            
            # Category 2: Fugitive Emissions
            'refrigerant_leaks': 0.05,       # 5% of emissions
            'industrial_gas_leaks': 0.05,    # 5% of emissions
            
            # Category 3: Mobile Combustion
            'vehicle_fleet': 0.30,           # 30% of emissions
            'off_road_equipment': 0.10       # 10% of emissions
        }
        
        # Create a new DataFrame to store the breakdown
        df = yearly_emissions.copy()
        
        # Add columns for each source category
        for source, fraction in source_distribution.items():
            # Add slight variations to the fractions for each year
            variations = np.random.normal(0, 0.01, len(df))
            adjusted_fractions = [max(0, fraction + var) for var in variations]
            
            # Calculate emissions for this source
            df[source] = df['emissions'] * adjusted_fractions
        
        # Normalize to ensure the sum equals total emissions
        source_columns = list(source_distribution.keys())
        df[source_columns] = df[source_columns].div(df[source_columns].sum(axis=1), axis=0) * df['emissions'].values[:, np.newaxis]
        
        return df
    
    def generate_monthly_data(self, yearly_df, seasonality=0.2):
        """
        Generate monthly emissions data from yearly data with seasonal patterns.
        
        Args:
            yearly_df (pandas.DataFrame): DataFrame with yearly emissions
            seasonality (float): Strength of seasonal variations
            
        Returns:
            pandas.DataFrame: DataFrame with monthly emissions data
        """
        # Create an empty list to store monthly data
        monthly_data = []
        
        # Source columns
        source_columns = [col for col in yearly_df.columns if col not in ['year', 'emissions']]
        
        # Iterate through each year
        for _, row in yearly_df.iterrows():
            year = int(row['year'])
            
            # Create monthly breakdown with seasonal patterns
            for month in range(1, 13):
                # Create seasonal pattern (higher in winter months)
                if month in [12, 1, 2]:  # Winter
                    seasonal_factor = 1 + seasonality
                elif month in [6, 7, 8]:  # Summer
                    seasonal_factor = 1 - seasonality
                else:  # Spring/Fall
                    seasonal_factor = 1
                
                # Apply monthly variation
                monthly_variation = np.random.normal(0, 0.03)
                
                # Create entry for this month
                monthly_entry = {
                    'year': year,
                    'month': month,
                    'date': datetime(year, month, 15)  # Middle of month
                }
                
                # Add total emissions
                monthly_emissions = row['emissions'] / 12 * seasonal_factor * (1 + monthly_variation)
                monthly_entry['emissions'] = monthly_emissions
                
                # Add source-specific emissions
                for source in source_columns:
                    source_monthly = row[source] / 12 * seasonal_factor * (1 + np.random.normal(0, 0.05))
                    monthly_entry[source] = source_monthly
                
                monthly_data.append(monthly_entry)
        
        # Convert to DataFrame
        monthly_df = pd.DataFrame(monthly_data)
        
        return monthly_df

    def generate_full_dataset(self, growth_rate=0.02, volatility=0.05, seasonality=0.2):
        """
        Generate a complete historical emissions dataset.
        
        Returns:
            tuple: (yearly_df, monthly_df) with emissions data
        """
        # Generate yearly data
        yearly_df = self.generate_yearly_emissions(growth_rate, volatility)
        
        # Break down by source
        yearly_df = self.generate_emissions_by_source(yearly_df)
        
        # Generate monthly data
        monthly_df = self.generate_monthly_data(yearly_df, seasonality)
        
        return yearly_df, monthly_df

# Helper function to get sample data
def get_sample_historical_data():
    """
    Get sample historical emissions data.
    
    Returns:
        tuple: (yearly_df, monthly_df) with sample emissions data
    """
    generator = HistoricalEmissionsGenerator()
    return generator.generate_full_dataset()

if __name__ == "__main__":
    # Example usage
    yearly, monthly = get_sample_historical_data()
    print("Yearly data:")
    print(yearly.head())
    print("\nMonthly data:")
    print(monthly.head())