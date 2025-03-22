"""
Unit tests for the data modules in the Emitrix Climate Scenario Analysis Platform.
Tests data generation, emission factors, and technology data.
"""
import unittest
import pandas as pd
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import modules to test
from data.historical_emissions import HistoricalEmissionsGenerator, get_sample_historical_data
from data.emission_factors import get_carbon_price, FUEL_EMISSION_FACTORS, REFRIGERANT_GWP
from data.technology_data import (
    TECHNOLOGY_DATA, 
    MODERATE_REDUCTION_TIMELINE,
    get_implementation_percentage,
    get_applicable_technologies
)


class TestHistoricalEmissions(unittest.TestCase):
    """Test cases for the historical emissions data module."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.generator = HistoricalEmissionsGenerator(
            start_year=2020,
            end_year=2024,
            baseline_emissions=10000,
            random_seed=42
        )
    
    def test_generate_yearly_emissions(self):
        """Test generation of yearly emissions data."""
        yearly_df = self.generator.generate_yearly_emissions(
            growth_rate=0.02,
            volatility=0.05
        )
        
        # Check DataFrame structure
        self.assertIsInstance(yearly_df, pd.DataFrame)
        self.assertIn('year', yearly_df.columns)
        self.assertIn('emissions', yearly_df.columns)
        
        # Check data length
        self.assertEqual(len(yearly_df), 5)  # 5 years (2020-2024)
        
        # Check values are reasonable
        self.assertGreater(yearly_df['emissions'].iloc[-1], yearly_df['emissions'].iloc[0])
        
        # Check years are as expected
        expected_years = list(range(2020, 2025))
        actual_years = yearly_df['year'].tolist()
        self.assertEqual(actual_years, expected_years)
    
    def test_generate_emissions_by_source(self):
        """Test breakdown of emissions by source category."""
        yearly_df = self.generator.generate_yearly_emissions()
        source_df = self.generator.generate_emissions_by_source(yearly_df)
        
        # Check that source categories are present
        expected_sources = [
            'natural_gas_boilers', 'diesel_generators', 
            'refrigerant_leaks', 'industrial_gas_leaks',
            'vehicle_fleet', 'off_road_equipment'
        ]
        
        for source in expected_sources:
            self.assertIn(source, source_df.columns)
        
        # Check that source emissions sum up to total emissions
        for _, row in source_df.iterrows():
            source_sum = sum(row[source] for source in expected_sources)
            self.assertAlmostEqual(source_sum, row['emissions'], delta=row['emissions'] * 0.01)
    
    def test_generate_monthly_data(self):
        """Test generation of monthly data with seasonality."""
        yearly_df = self.generator.generate_yearly_emissions()
        source_df = self.generator.generate_emissions_by_source(yearly_df)
        monthly_df = self.generator.generate_monthly_data(source_df, seasonality=0.2)
        
        # Check DataFrame structure
        self.assertIsInstance(monthly_df, pd.DataFrame)
        self.assertIn('year', monthly_df.columns)
        self.assertIn('month', monthly_df.columns)
        self.assertIn('date', monthly_df.columns)
        self.assertIn('emissions', monthly_df.columns)
        
        # Check data length
        self.assertEqual(len(monthly_df), 5 * 12)  # 5 years, 12 months each
        
        # Check seasonality effect (winter vs. summer)
        winter_months = monthly_df[monthly_df['month'].isin([1, 12])]
        summer_months = monthly_df[monthly_df['month'].isin([6, 7])]
        
        avg_winter = winter_months['emissions'].mean()
        avg_summer = summer_months['emissions'].mean()
        
        # Winter should have higher emissions than summer due to seasonality
        self.assertGreater(avg_winter, avg_summer)
    
    def test_get_sample_historical_data(self):
        """Test the helper function for getting sample data."""
        yearly, monthly = get_sample_historical_data()
        
        self.assertIsInstance(yearly, pd.DataFrame)
        self.assertIsInstance(monthly, pd.DataFrame)
        
        self.assertGreater(len(yearly), 0)
        self.assertGreater(len(monthly), 0)


class TestEmissionFactors(unittest.TestCase):
    """Test cases for emission factors data."""
    
    def test_fuel_emission_factors(self):
        """Test fuel emission factor data structure."""
        # Check that we have data for key fuels
        expected_fuels = ['natural_gas', 'diesel', 'gasoline', 'lpg', 'heavy_fuel_oil']
        for fuel in expected_fuels:
            self.assertIn(fuel, FUEL_EMISSION_FACTORS)
        
        # Check that each fuel has CO2, CH4, N2O, and CO2e factors
        for fuel, factors in FUEL_EMISSION_FACTORS.items():
            self.assertIn('co2', factors)
            self.assertIn('ch4', factors)
            self.assertIn('n2o', factors)
            self.assertIn('co2e', factors)
            
            # CO2e should be greater than CO2 (due to higher GWP of other gases)
            self.assertGreaterEqual(factors['co2e'], factors['co2'])
    
    def test_refrigerant_gwp(self):
        """Test refrigerant GWP data structure."""
        # Check that we have data for key refrigerants
        expected_refrigerants = ['R-134a', 'R-410A', 'R-22', 'SF6']
        for refrigerant in expected_refrigerants:
            self.assertIn(refrigerant, REFRIGERANT_GWP)
        
        # Check that values are in expected ranges
        for refrigerant, gwp in REFRIGERANT_GWP.items():
            # GWP values should be positive
            self.assertGreater(gwp, 0)
            
            # High-GWP refrigerants like SF6 should be very high
            if refrigerant == 'SF6':
                self.assertGreater(gwp, 20000)
    
    def test_get_carbon_price(self):
        """Test getting carbon price for different years and trajectories."""
        # Test medium trajectory
        price_2025 = get_carbon_price(2025, "medium")
        price_2030 = get_carbon_price(2030, "medium")
        
        # Price should increase over time
        self.assertLess(price_2025, price_2030)
        
        # Test trajectories relative to each other
        low_2030 = get_carbon_price(2030, "low")
        medium_2030 = get_carbon_price(2030, "medium")
        high_2030 = get_carbon_price(2030, "high")
        
        self.assertLess(low_2030, medium_2030)
        self.assertLess(medium_2030, high_2030)
        
        # Test interpolation for a year not explicitly defined
        interp_year = 2027
        interp_price = get_carbon_price(interp_year, "medium")
        
        # 2027 price should be between 2026 and 2028 prices
        price_2026 = get_carbon_price(2026, "medium")
        price_2028 = get_carbon_price(2028, "medium")
        
        self.assertGreaterEqual(interp_price, price_2026)
        self.assertLessEqual(interp_price, price_2028)


class TestTechnologyData(unittest.TestCase):
    """Test cases for technology data."""
    
    def test_technology_data_structure(self):
        """Test the structure of technology data."""
        # Check that we have data for key technologies
        expected_technologies = [
            'electric_boilers', 'high_efficiency_boilers', 
            'electric_vehicles', 'refrigerant_leak_detection'
        ]
        
        for tech in expected_technologies:
            self.assertIn(tech, TECHNOLOGY_DATA)
        
        # Check that each technology has required fields
        required_fields = [
            'capex_per_unit', 'opex_change_per_year', 'lifespan_years',
            'emission_reduction_percentage', 'applicability'
        ]
        
        for tech, data in TECHNOLOGY_DATA.items():
            for field in required_fields:
                self.assertIn(field, data)
            
            # Check that applicability is a list
            self.assertIsInstance(data['applicability'], list)
            self.assertGreater(len(data['applicability']), 0)
    
    def test_moderate_reduction_timeline(self):
        """Test the moderate reduction implementation timeline."""
        # Check that timeline has data for key sources
        expected_sources = [
            'natural_gas_boilers', 'vehicle_fleet', 
            'refrigerant_leaks', 'industrial_gas_leaks'
        ]
        
        for source in expected_sources:
            self.assertIn(source, MODERATE_REDUCTION_TIMELINE)
        
        # Check that each source has technologies with implementation percentages
        for source, techs in MODERATE_REDUCTION_TIMELINE.items():
            self.assertGreater(len(techs), 0)
            
            for tech, years in techs.items():
                # Check that we have implementation data for some years
                self.assertGreater(len(years), 0)
                
                # Check that implementation percentages are between 0 and 1
                for year, pct in years.items():
                    self.assertGreaterEqual(pct, 0.0)
                    self.assertLessEqual(pct, 1.0)
    
    def test_get_implementation_percentage(self):
        """Test getting implementation percentage for different years."""
        # Test a key technology implementation
        source = "natural_gas_boilers"
        tech = "electric_boilers"
        
        # Should start at 0 before implementation
        pct_2025 = get_implementation_percentage(source, tech, 2025)
        self.assertEqual(pct_2025, 0.0)
        
        # Should reach 0.5 (50%) by 2030
        pct_2030 = get_implementation_percentage(source, tech, 2030)
        self.assertEqual(pct_2030, 0.5)
        
        # Test interpolation for a year not explicitly defined
        interp_year = 2027
        interp_pct = get_implementation_percentage(source, tech, interp_year)
        
        # 2027 percentage should be between 2026 and 2028 percentages
        pct_2026 = get_implementation_percentage(source, tech, 2026)
        pct_2028 = get_implementation_percentage(source, tech, 2028)
        
        self.assertGreaterEqual(interp_pct, pct_2026)
        self.assertLessEqual(interp_pct, pct_2028)
    
    def test_get_applicable_technologies(self):
        """Test getting applicable technologies for a source."""
        # Test a key source
        source = "natural_gas_boilers"
        techs = get_applicable_technologies(source)
        
        # Should include electric boilers and high efficiency boilers
        self.assertIn("electric_boilers", techs)
        self.assertIn("high_efficiency_boilers", techs)
        
        # Should not include technologies for other sources
        self.assertNotIn("electric_vehicles", techs)
        
        # Test another source
        source = "vehicle_fleet"
        techs = get_applicable_technologies(source)
        
        # Should include electric vehicles
        self.assertIn("electric_vehicles", techs)
        
        # Should not include technologies for other sources
        self.assertNotIn("electric_boilers", techs)


if __name__ == '__main__':
    unittest.main()