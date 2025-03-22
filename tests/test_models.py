"""
Unit tests for the Scope 1 emissions reduction model and associated components.
"""
import unittest
import pandas as pd
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import modules to test
from models.scope1_reduction_model import Scope1ReductionModel
from models.technology_calculator import TechnologyTransitionCalculator
from models.financial_model import FinancialImpactModel
from models.time_series_utils import TimeSeriesForecaster


class TestScope1ReductionModel(unittest.TestCase):
    """Test cases for the Scope1ReductionModel class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a sample model with test parameters
        self.model = Scope1ReductionModel(
            reduction_target=0.3,
            target_year=2030,
            start_year=2025,
            carbon_price_start=30,
            carbon_price_increase=0.05,
            discount_rate=0.07
        )
        
        # Create sample historical data
        self.historical_data = pd.DataFrame({
            'year': [2020, 2021, 2022, 2023, 2024],
            'emissions': [10000, 10200, 10400, 10600, 10800],
            'natural_gas_boilers': [4000, 4080, 4160, 4240, 4320],
            'diesel_generators': [1000, 1020, 1040, 1060, 1080],
            'refrigerant_leaks': [500, 510, 520, 530, 540],
            'industrial_gas_leaks': [500, 510, 520, 530, 540],
            'vehicle_fleet': [3000, 3060, 3120, 3180, 3240],
            'off_road_equipment': [1000, 1020, 1040, 1060, 1080]
        })
        
        # Load the sample data
        self.model.load_historical_data(self.historical_data)
    
    def test_load_historical_data(self):
        """Test loading historical data."""
        self.assertIsNotNone(self.model.historical_data)
        self.assertEqual(len(self.model.historical_data), 5)
        self.assertEqual(self.model.historical_data['emissions'].iloc[-1], 10800)
    
    def test_project_bau_emissions(self):
        """Test business-as-usual emissions projection."""
        projection = self.model.project_bau_emissions(2030)
        
        # Check projection length (5 years of history + 6 years of projection)
        self.assertEqual(len(projection), 11)
        
        # Check projection includes all source categories
        source_columns = ['natural_gas_boilers', 'diesel_generators', 'refrigerant_leaks', 
                          'industrial_gas_leaks', 'vehicle_fleet', 'off_road_equipment']
        for col in source_columns:
            self.assertIn(col, projection.columns)
        
        # Check growth pattern (should continue historical trend)
        # Historical CAGR is about 2% per year
        expected_2030_emissions = 10800 * (1.02 ** 6)  # 6 years from 2024 to 2030
        actual_2030_emissions = projection[projection['year'] == 2030]['emissions'].values[0]
        
        # Allow 5% tolerance for rounding differences
        self.assertAlmostEqual(actual_2030_emissions, expected_2030_emissions, delta=expected_2030_emissions * 0.05)
    
    def test_apply_reduction_measures(self):
        """Test applying emission reduction measures."""
        # First project BAU emissions
        self.model.project_bau_emissions(2030)
        
        # Apply reduction measures
        reduced_data = self.model.apply_reduction_measures()
        
        # Check that reduced data has the expected columns
        self.assertIn('reduced_emissions', reduced_data.columns)
        for source in ['natural_gas_boilers', 'vehicle_fleet']:
            self.assertIn(f'reduced_{source}', reduced_data.columns)
        
        # Check that reductions occur in years after start_year
        for year in range(2025, 2031):
            year_data = reduced_data[reduced_data['year'] == year]
            bau_emissions = year_data['emissions'].values[0]
            reduced_emissions = year_data['reduced_emissions'].values[0]
            
            # Emissions should be reduced after 2025
            if year > 2025:
                self.assertLess(reduced_emissions, bau_emissions)
            else:
                # No reduction in the first year
                self.assertEqual(reduced_emissions, bau_emissions)
        
        # Check that target year emissions meet the reduction target
        target_year_data = reduced_data[reduced_data['year'] == 2030]
        bau_emissions = target_year_data['emissions'].values[0]
        reduced_emissions = target_year_data['reduced_emissions'].values[0]
        actual_reduction_pct = (bau_emissions - reduced_emissions) / bau_emissions
        
        # Allow 5% tolerance for implementation differences
        self.assertAlmostEqual(actual_reduction_pct, 0.3, delta=0.05)
    
    def test_calculate_financial_impact(self):
        """Test financial impact calculations."""
        # Project BAU and apply reductions
        self.model.project_bau_emissions(2030)
        reduced_data = self.model.apply_reduction_measures()
        
        # Calculate financial impact
        financial_data = self.model.calculate_financial_impact(reduced_data)
        
        # Check that financial data has the expected columns
        expected_columns = ['year', 'bau_emissions', 'reduced_emissions', 'emissions_avoided',
                           'implementation_cost', 'operational_change', 'carbon_price',
                           'carbon_cost_bau', 'carbon_cost_reduced', 'carbon_savings',
                           'net_benefit', 'npv_factor', 'discounted_net_benefit']
        
        for col in expected_columns:
            self.assertIn(col, financial_data.columns)
        
        # Check that calculations are reasonable
        for _, row in financial_data.iterrows():
            year = row['year']
            
            if year > 2025:  # After start year
                # Implementation costs should be positive
                self.assertGreater(row['implementation_cost'], 0)
                
                # Carbon savings should be proportional to emissions avoided
                expected_carbon_savings = row['emissions_avoided'] * row['carbon_price']
                self.assertAlmostEqual(row['carbon_savings'], expected_carbon_savings, delta=expected_carbon_savings * 0.01)
                
                # Net benefit calculation should be correct
                expected_net_benefit = row['carbon_savings'] - row['implementation_cost'] - row['operational_change']
                self.assertAlmostEqual(row['net_benefit'], expected_net_benefit, delta=abs(expected_net_benefit) * 0.01)
                
                # Discounted benefit calculation should be correct
                expected_discounted_benefit = row['net_benefit'] * row['npv_factor']
                self.assertAlmostEqual(row['discounted_net_benefit'], expected_discounted_benefit, delta=abs(expected_discounted_benefit) * 0.01)
    
    def test_run_scenario(self):
        """Test the full scenario run."""
        # Run the full scenario
        results = self.model.run_scenario()
        
        # Check that results contain all expected components
        expected_keys = ['scenario_name', 'reduction_target', 'target_year',
                         'historical_data', 'projection_data', 'reduced_data',
                         'financial_data', 'metrics']
        
        for key in expected_keys:
            self.assertIn(key, results)
        
        # Check metrics
        metrics = results['metrics']
        expected_metric_keys = ['reduction_achieved_tco2e', 'reduction_percentage',
                               'total_implementation_cost', 'net_present_value',
                               'return_on_investment', 'payback_years']
        
        for key in expected_metric_keys:
            self.assertIn(key, metrics)
        
        # Reduction percentage should be close to target
        self.assertAlmostEqual(metrics['reduction_percentage'], 0.3, delta=0.05)


class TestTechnologyCalculator(unittest.TestCase):
    """Test cases for the TechnologyTransitionCalculator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.calculator = TechnologyTransitionCalculator(2025, 2030)
        
        # Create sample baseline emissions
        self.baseline_emissions = {
            2025: 5000,
            2026: 5100,
            2027: 5200,
            2028: 5300,
            2029: 5400,
            2030: 5500
        }
    
    def test_calculate_emission_reductions(self):
        """Test calculation of emission reductions."""
        # Calculate reductions for electric boilers
        reductions = self.calculator.calculate_emission_reductions(
            "natural_gas_boilers", 
            self.baseline_emissions,
            ["electric_boilers"]
        )
        
        # Check that results contain expected keys
        expected_keys = ['years', 'baseline_emissions', 'reduced_emissions',
                        'reductions_by_technology']
        
        for key in expected_keys:
            self.assertIn(key, reductions)
        
        # Check that we have data for each year
        self.assertEqual(len(reductions['years']), 6)
        
        # Check that reductions increase over time
        for i in range(1, len(reductions['reduced_emissions'])):
            # Electric boilers have 100% reduction effect, so reductions should increase
            # as implementation percentage increases
            curr_emissions = reductions['reduced_emissions'][i]
            prev_emissions = reductions['reduced_emissions'][i-1]
            
            # Baseline increases by 100 each year, but reductions should increase more
            # Exception: first year has no implementation yet
            if i > 1:
                # Reduced emissions should drop despite rising baseline
                self.assertLess(curr_emissions, prev_emissions)
    
    def test_calculate_implementation_costs(self):
        """Test calculation of implementation costs."""
        # Calculate costs for electric boilers
        costs = self.calculator.calculate_implementation_costs(
            "natural_gas_boilers", 
            self.baseline_emissions,
            ["electric_boilers"]
        )
        
        # Check that results contain expected keys
        expected_keys = ['years', 'total_capex', 'total_opex_change', 'total_cost',
                         'costs_by_technology']
        
        for key in expected_keys:
            self.assertIn(key, costs)
        
        # Check that we have data for each year
        self.assertEqual(len(costs['years']), 6)
        
        # First year should have no CAPEX (implementation starts in second year)
        self.assertEqual(costs['total_capex'][0], 0)
        
        # Later years should have positive CAPEX
        for i in range(1, len(costs['total_capex'])):
            self.assertGreater(costs['total_capex'][i], 0)
    
    def test_generate_adoption_curve(self):
        """Test generation of adoption curves."""
        # Generate linear adoption curve
        linear_curve = self.calculator.generate_adoption_curve(
            "natural_gas_boilers", 
            "electric_boilers",
            0.0, 0.5, 
            "linear"
        )
        
        # Check that curve has data for each year
        self.assertEqual(len(linear_curve), 6)
        
        # Check that curve starts at 0 and ends at 0.5
        self.assertEqual(linear_curve[2025], 0.0)
        self.assertEqual(linear_curve[2030], 0.5)
        
        # Check that curve increases linearly
        for year in range(2026, 2030):
            prev_year = year - 1
            expected_increase = 0.5 / 5  # 0.5 spread over 5 years
            actual_increase = linear_curve[year] - linear_curve[prev_year]
            self.assertAlmostEqual(actual_increase, expected_increase, delta=0.001)
        
        # Test S-curve
        s_curve = self.calculator.generate_adoption_curve(
            "natural_gas_boilers", 
            "electric_boilers",
            0.0, 0.5, 
            "s_curve"
        )
        
        # S-curve should have more growth in the middle years
        middle_growth = s_curve[2028] - s_curve[2027]
        early_growth = s_curve[2026] - s_curve[2025]
        self.assertGreater(middle_growth, early_growth)


class TestFinancialModel(unittest.TestCase):
    """Test cases for the FinancialImpactModel class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.model = FinancialImpactModel(
            start_year=2025,
            end_year=2030,
            carbon_price_trajectory="medium",
            discount_rate=0.07,
            tax_rate=0.21
        )
        
        # Create sample emissions data
        self.emissions_data = {
            2025: {'bau': 5000, 'reduced': 5000},  # No reduction in first year
            2026: {'bau': 5100, 'reduced': 4950},
            2027: {'bau': 5200, 'reduced': 4800},
            2028: {'bau': 5300, 'reduced': 4600},
            2029: {'bau': 5400, 'reduced': 4350},
            2030: {'bau': 5500, 'reduced': 4000}   # ~27% reduction by 2030
        }
        
        # Create sample cost data
        self.cost_data = {
            2025: {'capex': 100000, 'opex_change': 0},      # Initial investment
            2026: {'capex': 200000, 'opex_change': -10000}, # Savings in opex
            2027: {'capex': 250000, 'opex_change': -20000},
            2028: {'capex': 300000, 'opex_change': -30000},
            2029: {'capex': 350000, 'opex_change': -40000},
            2030: {'capex': 400000, 'opex_change': -50000}
        }
    
    def test_calculate_carbon_costs(self):
        """Test calculation of carbon costs."""
        carbon_costs = self.model.calculate_carbon_costs(self.emissions_data)
        
        # Check that results have expected columns
        expected_columns = ['year', 'carbon_price', 'bau_emissions', 'reduced_emissions',
                           'bau_carbon_cost', 'reduced_carbon_cost', 'carbon_cost_savings']
        
        for col in expected_columns:
            self.assertIn(col, carbon_costs.columns)
        
        # Check calculations for a specific year
        for _, row in carbon_costs.iterrows():
            year = row['year']
            bau_emissions = self.emissions_data[year]['bau']
            reduced_emissions = self.emissions_data[year]['reduced']
            carbon_price = row['carbon_price']
            
            # Check that emissions values match
            self.assertEqual(row['bau_emissions'], bau_emissions)
            self.assertEqual(row['reduced_emissions'], reduced_emissions)
            
            # Check carbon cost calculations
            expected_bau_cost = bau_emissions * carbon_price
            expected_reduced_cost = reduced_emissions * carbon_price
            expected_savings = expected_bau_cost - expected_reduced_cost
            
            self.assertAlmostEqual(row['bau_carbon_cost'], expected_bau_cost, delta=expected_bau_cost * 0.01)
            self.assertAlmostEqual(row['reduced_carbon_cost'], expected_reduced_cost, delta=expected_reduced_cost * 0.01)
            self.assertAlmostEqual(row['carbon_cost_savings'], expected_savings, delta=expected_savings * 0.01)
    
    def test_calculate_implementation_costs(self):
        """Test calculation of implementation costs."""
        implementation_costs = self.model.calculate_implementation_costs(self.cost_data)
        
        # Check that results have expected columns
        expected_columns = ['year', 'capex', 'opex_change', 'depreciation',
                           'tax_benefit', 'net_implementation_cost']
        
        for col in expected_columns:
            self.assertIn(col, implementation_costs.columns)
        
        # Check calculations for a specific year
        for _, row in implementation_costs.iterrows():
            year = row['year']
            capex = self.cost_data[year]['capex']
            opex_change = self.cost_data[year]['opex_change']
            
            # Check that capex and opex match
            self.assertEqual(row['capex'], capex)
            self.assertEqual(row['opex_change'], opex_change)
            
            # Check depreciation calculation (straight-line over 10 years)
            expected_depreciation = capex / 10
            self.assertAlmostEqual(row['depreciation'], expected_depreciation, delta=expected_depreciation * 0.01)
            
            # Check tax benefit calculation
            if opex_change > 0:
                expected_tax_benefit = (expected_depreciation + opex_change) * 0.21
            else:
                expected_tax_benefit = expected_depreciation * 0.21
            
            self.assertAlmostEqual(row['tax_benefit'], expected_tax_benefit, delta=expected_tax_benefit * 0.01)
            
            # Check net implementation cost
            expected_net_cost = capex + opex_change - row['tax_benefit']
            self.assertAlmostEqual(row['net_implementation_cost'], expected_net_cost, delta=abs(expected_net_cost) * 0.01)
    
    def test_run_financial_analysis(self):
        """Test full financial analysis."""
        results = self.model.run_financial_analysis(self.emissions_data, self.cost_data)
        
        # Check that results contain all expected components
        expected_keys = ['carbon_costs', 'implementation_costs', 'financial_metrics',
                         'marginal_abatement_cost']
        
        for key in expected_keys:
            self.assertIn(key, results)
        
        # Check financial metrics
        metrics = results['financial_metrics']
        expected_metric_keys = ['npv', 'roi', 'payback_period', 'irr', 
                               'total_discounted_benefit', 'total_discounted_cost', 
                               'benefit_cost_ratio']
        
        for key in expected_metric_keys:
            self.assertIn(key, metrics)


class TestTimeSeriesForecaster(unittest.TestCase):
    """Test cases for the TimeSeriesForecaster class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create sample historical data
        self.historical_data = pd.DataFrame({
            'year': [2020, 2021, 2022, 2023, 2024],
            'emissions': [10000, 10200, 10404, 10612, 10824]  # 2% growth
        })
        
        # Initialize forecaster
        self.forecaster = TimeSeriesForecaster(self.historical_data)
    
    def test_fit_linear_trend(self):
        """Test fitting linear trend model."""
        model = self.forecaster.fit_linear_trend()
        
        # Check that model has expected parameters
        expected_keys = ['type', 'slope', 'intercept', 'r_squared', 'p_value', 'std_err']
        for key in expected_keys:
            self.assertIn(key, model)
        
        # Check model type
        self.assertEqual(model['type'], 'linear')
        
        # Check slope (should be around 206, which is the average yearly increase)
        self.assertAlmostEqual(model['slope'], 206, delta=10)
        
        # R-squared should be very high for this clean data
        self.assertGreater(model['r_squared'], 0.99)
    
    def test_fit_exponential_trend(self):
        """Test fitting exponential trend model."""
        model = self.forecaster.fit_exponential_trend()
        
        # Check that model has expected parameters
        expected_keys = ['type', 'growth_rate', 'initial_value', 'slope', 
                         'intercept', 'r_squared', 'p_value', 'std_err']
        for key in expected_keys:
            self.assertIn(key, model)
        
        # Check model type
        self.assertEqual(model['type'], 'exponential')
        
        # Check growth rate (should be around 0.02 or 2%)
        self.assertAlmostEqual(model['growth_rate'], 0.02, delta=0.003)
        
        # R-squared should be very high for this clean data
        self.assertGreater(model['r_squared'], 0.99)
    
    def test_forecast(self):
        """Test forecasting."""
        # Generate forecast with linear model
        forecast = self.forecaster.forecast(2025, 2030, 'linear')
        
        # Check that forecast has expected length (6 years)
        self.assertEqual(len(forecast), 6)
        
        # Check that forecast has expected columns
        expected_columns = ['year', 'emissions', 'model_type']
        for col in expected_columns:
            self.assertIn(col, forecast.columns)
        
        # Check forecast years
        expected_years = [2025, 2026, 2027, 2028, 2029, 2030]
        actual_years = forecast['year'].tolist()
        self.assertEqual(actual_years, expected_years)
        
        # Check that forecast values follow the trend
        # Last historical value is 10824, with ~206 annual increase
        expected_2025 = 10824 + 206
        actual_2025 = forecast[forecast['year'] == 2025]['emissions'].values[0]
        self.assertAlmostEqual(actual_2025, expected_2025, delta=expected_2025 * 0.05)
    
    def test_forecast_with_scenarios(self):
        """Test scenario forecasting."""
        scenarios = self.forecaster.forecast_with_scenarios(2025, 2030)
        
        # Check that we have all expected scenarios
        expected_scenarios = ['business_as_usual', 'accelerated_growth',
                             'moderate_reduction', 'aggressive_reduction']
        
        for scenario in expected_scenarios:
            self.assertIn(scenario, scenarios)
        
        # Check that each scenario has the expected data structure
        for scenario_id, scenario_df in scenarios.items():
            # Each scenario should have the same number of years
            self.assertEqual(len(scenario_df), 6)
            
            # Check that scenario has expected columns
            expected_columns = ['year', 'emissions', 'model_type', 'scenario', 'growth_modifier']
            for col in expected_columns:
                self.assertIn(col, scenario_df.columns)
        
        # Check that scenarios have different emission values
        bau_2030 = scenarios['business_as_usual'][scenarios['business_as_usual']['year'] == 2030]['emissions'].values[0]
        accelerated_2030 = scenarios['accelerated_growth'][scenarios['accelerated_growth']['year'] == 2030]['emissions'].values[0]
        reduced_2030 = scenarios['moderate_reduction'][scenarios['moderate_reduction']['year'] == 2030]['emissions'].values[0]
        
        # Accelerated should be higher than BAU
        self.assertGreater(accelerated_2030, bau_2030)
        
        # Reduced should be lower than BAU
        self.assertLess(reduced_2030, bau_2030)

if __name__ == '__main__':
    unittest.main()