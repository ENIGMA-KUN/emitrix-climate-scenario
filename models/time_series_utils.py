"""
Module for time series analysis and forecasting of emissions data.
Provides statistical methods for trend analysis and forecasting.
"""
import numpy as np
import pandas as pd
from datetime import datetime
from scipy import stats

class TimeSeriesForecaster:
    """
    Class for forecasting time series data, specifically for emissions trends.
    """
    
    def __init__(self, historical_data=None):
        """
        Initialize the forecaster with historical data.
        
        Args:
            historical_data (pandas.DataFrame): Historical emissions data with 'year' and 'emissions' columns
        """
        self.historical_data = historical_data
        self.trend_model = None
        self.forecast_data = None
        
    def load_data(self, data):
        """
        Load historical emissions data.
        
        Args:
            data (pandas.DataFrame): Historical emissions data
        """
        self.historical_data = data
        return data
    
    def fit_linear_trend(self):
        """
        Fit a linear trend model to historical data.
        
        Returns:
            dict: Dictionary with model parameters
        """
        if self.historical_data is None:
            raise ValueError("Historical data not loaded. Call load_data() first.")
        
        # Extract years and emissions
        years = self.historical_data['year'].values
        emissions = self.historical_data['emissions'].values
        
        # Fit linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(years, emissions)
        
        # Store model parameters
        self.trend_model = {
            'type': 'linear',
            'slope': slope,
            'intercept': intercept,
            'r_squared': r_value**2,
            'p_value': p_value,
            'std_err': std_err
        }
        
        return self.trend_model
    
    def fit_exponential_trend(self):
        """
        Fit an exponential trend model to historical data.
        
        Returns:
            dict: Dictionary with model parameters
        """
        if self.historical_data is None:
            raise ValueError("Historical data not loaded. Call load_data() first.")
        
        # Extract years and emissions
        years = self.historical_data['year'].values
        emissions = self.historical_data['emissions'].values
        
        # Fit exponential model (linear regression on log-transformed data)
        log_emissions = np.log(emissions)
        
        # Fit linear regression to log-transformed data
        slope, intercept, r_value, p_value, std_err = stats.linregress(years, log_emissions)
        
        # Calculate growth rate (e^slope - 1)
        growth_rate = np.exp(slope) - 1
        
        # Store model parameters
        self.trend_model = {
            'type': 'exponential',
            'growth_rate': growth_rate,
            'initial_value': np.exp(intercept),
            'slope': slope,
            'intercept': intercept,
            'r_squared': r_value**2,
            'p_value': p_value,
            'std_err': std_err
        }
        
        return self.trend_model
    
    def fit_polynomial_trend(self, degree=2):
        """
        Fit a polynomial trend model to historical data.
        
        Args:
            degree (int): Degree of the polynomial
            
        Returns:
            dict: Dictionary with model parameters
        """
        if self.historical_data is None:
            raise ValueError("Historical data not loaded. Call load_data() first.")
        
        # Extract years and emissions
        years = self.historical_data['year'].values
        emissions = self.historical_data['emissions'].values
        
        # Fit polynomial regression
        coeffs = np.polyfit(years, emissions, degree)
        
        # Calculate r-squared
        p = np.poly1d(coeffs)
        predicted = p(years)
        ss_total = np.sum((emissions - np.mean(emissions))**2)
        ss_residual = np.sum((emissions - predicted)**2)
        r_squared = 1 - (ss_residual / ss_total)
        
        # Store model parameters
        self.trend_model = {
            'type': 'polynomial',
            'degree': degree,
            'coefficients': coeffs.tolist(),
            'r_squared': r_squared
        }
        
        return self.trend_model
    
    def forecast(self, start_year, end_year, model_type='auto'):
        """
        Forecast emissions based on the fitted trend model.
        
        Args:
            start_year (int): First year to forecast
            end_year (int): Last year to forecast
            model_type (str): Type of model to use ('linear', 'exponential', 'polynomial', 'auto')
            
        Returns:
            pandas.DataFrame: DataFrame with forecasted emissions
        """
        if model_type == 'auto':
            # If no model specified, try to fit all and choose best by R²
            if self.trend_model is None:
                linear_model = self.fit_linear_trend()
                self.trend_model = linear_model  # Reset to linear
                
                exponential_model = self.fit_exponential_trend()
                polynomial_model = self.fit_polynomial_trend(degree=2)
                
                # Compare R² values
                r_squared_values = {
                    'linear': linear_model['r_squared'],
                    'exponential': exponential_model['r_squared'],
                    'polynomial': polynomial_model['r_squared']
                }
                
                # Choose best model
                best_model = max(r_squared_values, key=r_squared_values.get)
                
                # Set the trend model to the best one
                if best_model == 'linear':
                    self.trend_model = linear_model
                elif best_model == 'exponential':
                    self.trend_model = exponential_model
                else:  # polynomial
                    self.trend_model = polynomial_model
        elif model_type == 'linear':
            self.fit_linear_trend()
        elif model_type == 'exponential':
            self.fit_exponential_trend()
        elif model_type == 'polynomial':
            self.fit_polynomial_trend()
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Generate forecast years
        forecast_years = list(range(start_year, end_year + 1))
        
        # Initialize forecast data
        forecast_data = []
        
        # Generate forecasts based on model type
        if self.trend_model['type'] == 'linear':
            slope = self.trend_model['slope']
            intercept = self.trend_model['intercept']
            
            for year in forecast_years:
                forecast_emissions = slope * year + intercept
                forecast_data.append({
                    'year': year,
                    'emissions': forecast_emissions,
                    'model_type': 'linear'
                })
        
        elif self.trend_model['type'] == 'exponential':
            initial_value = self.trend_model['initial_value']
            growth_rate = self.trend_model['growth_rate']
            first_year = self.historical_data['year'].min()
            
            for year in forecast_years:
                years_since_start = year - first_year
                forecast_emissions = initial_value * ((1 + growth_rate) ** years_since_start)
                forecast_data.append({
                    'year': year,
                    'emissions': forecast_emissions,
                    'model_type': 'exponential'
                })
        
        elif self.trend_model['type'] == 'polynomial':
            coeffs = self.trend_model['coefficients']
            p = np.poly1d(coeffs)
            
            for year in forecast_years:
                forecast_emissions = p(year)
                forecast_data.append({
                    'year': year,
                    'emissions': forecast_emissions,
                    'model_type': 'polynomial'
                })
        
        # Convert to DataFrame
        self.forecast_data = pd.DataFrame(forecast_data)
        return self.forecast_data
    
    def forecast_with_scenarios(self, start_year, end_year, scenarios=None):
        """
        Generate emissions forecasts with different scenarios.
        
        Args:
            start_year (int): First year to forecast
            end_year (int): Last year to forecast
            scenarios (dict): Dictionary of scenario parameters
            
        Returns:
            dict: Dictionary with scenario forecasts
        """
        # Default scenarios if none provided
        if scenarios is None:
            scenarios = {
                'business_as_usual': {
                    'description': 'Business as Usual',
                    'growth_modifier': 1.0  # No change to trend
                },
                'accelerated_growth': {
                    'description': 'Accelerated Growth',
                    'growth_modifier': 1.5  # 50% faster growth
                },
                'moderate_reduction': {
                    'description': 'Moderate Reduction',
                    'growth_modifier': 0.7  # 30% reduction in growth rate
                },
                'aggressive_reduction': {
                    'description': 'Aggressive Reduction',
                    'growth_modifier': 0.3  # 70% reduction in growth rate
                }
            }
        
        # Generate base forecast
        base_forecast = self.forecast(start_year, end_year)
        
        # Apply scenario modifiers
        scenario_forecasts = {}
        
        for scenario_id, scenario_params in scenarios.items():
            # Copy base forecast
            scenario_df = base_forecast.copy()
            
            # Apply growth modifier
            if self.trend_model['type'] == 'linear':
                # Modify slope for linear model
                original_slope = self.trend_model['slope']
                modified_slope = original_slope * scenario_params['growth_modifier']
                
                # Recalculate emissions with modified slope
                first_year_emission = scenario_df.loc[scenario_df['year'] == start_year, 'emissions'].values[0]
                
                for i, row in scenario_df.iterrows():
                    years_since_start = row['year'] - start_year
                    scenario_df.loc[i, 'emissions'] = first_year_emission + modified_slope * years_since_start
            
            elif self.trend_model['type'] == 'exponential':
                # Modify growth rate for exponential model
                original_growth_rate = self.trend_model['growth_rate']
                modified_growth_rate = original_growth_rate * scenario_params['growth_modifier']
                
                # Recalculate emissions with modified growth rate
                first_year_emission = scenario_df.loc[scenario_df['year'] == start_year, 'emissions'].values[0]
                
                for i, row in scenario_df.iterrows():
                    years_since_start = row['year'] - start_year
                    scenario_df.loc[i, 'emissions'] = first_year_emission * ((1 + modified_growth_rate) ** years_since_start)
            
            else:  # polynomial
                # For polynomial, just scale the difference from first year
                first_year_emission = scenario_df.loc[scenario_df['year'] == start_year, 'emissions'].values[0]
                
                for i, row in scenario_df.iterrows():
                    if row['year'] > start_year:
                        original_increase = row['emissions'] - first_year_emission
                        modified_increase = original_increase * scenario_params['growth_modifier']
                        scenario_df.loc[i, 'emissions'] = first_year_emission + modified_increase
            
            # Add scenario information
            scenario_df['scenario'] = scenario_params['description']
            scenario_df['growth_modifier'] = scenario_params['growth_modifier']
            
            # Store in results dictionary
            scenario_forecasts[scenario_id] = scenario_df
        
        return scenario_forecasts
    
    def add_seasonality(self, annual_data, seasonality_pattern=None):
        """
        Add monthly seasonality to annual forecasts.
        
        Args:
            annual_data (pandas.DataFrame): Annual emissions forecast
            seasonality_pattern (dict): Monthly seasonality factors
            
        Returns:
            pandas.DataFrame: Monthly emissions forecast with seasonality
        """
        # Default seasonality pattern if none provided
        if seasonality_pattern is None:
            # Higher in winter (heating), lower in summer
            seasonality_pattern = {
                1: 1.2,   # January
                2: 1.15,  # February
                3: 1.05,  # March
                4: 0.95,  # April
                5: 0.9,   # May
                6: 0.85,  # June
                7: 0.8,   # July
                8: 0.85,  # August
                9: 0.9,   # September
                10: 0.95, # October
                11: 1.05, # November
                12: 1.15  # December
            }
        
        # Create monthly data
        monthly_data = []
        
        for _, row in annual_data.iterrows():
            year = row['year']
            annual_emissions = row['emissions']
            
            # Calculate baseline monthly emissions (without seasonality)
            baseline_monthly = annual_emissions / 12
            
            # Add monthly data with seasonality
            for month in range(1, 13):
                seasonal_factor = seasonality_pattern[month]
                monthly_emissions = baseline_monthly * seasonal_factor
                
                # Add small random variation
                random_factor = 1 + np.random.normal(0, 0.02)  # 2% random variation
                monthly_emissions *= random_factor
                
                monthly_data.append({
                    'year': year,
                    'month': month,
                    'date': datetime(year, month, 15),  # Middle of month
                    'emissions': monthly_emissions,
                    'annual_emissions': annual_emissions,
                    'scenario': row.get('scenario', 'forecast')
                })
        
        return pd.DataFrame(monthly_data)

    def calculate_trend_statistics(self):
        """
        Calculate statistics about the emissions trend.
        
        Returns:
            dict: Dictionary with trend statistics
        """
        if self.historical_data is None:
            raise ValueError("Historical data not loaded. Call load_data() first.")
        
        # Extract years and emissions
        years = self.historical_data['year'].values
        emissions = self.historical_data['emissions'].values
        
        # Calculate CAGR (Compound Annual Growth Rate)
        start_year = years[0]
        end_year = years[-1]
        start_emissions = emissions[0]
        end_emissions = emissions[-1]
        years_diff = end_year - start_year
        
        if years_diff > 0:
            cagr = (end_emissions / start_emissions) ** (1 / years_diff) - 1
        else:
            cagr = 0
        
        # Calculate year-over-year growth rates
        yoy_growth = []
        for i in range(1, len(years)):
            growth = (emissions[i] - emissions[i-1]) / emissions[i-1]
            yoy_growth.append(growth)
        
        # Calculate volatility (standard deviation of YoY growth)
        if len(yoy_growth) > 0:
            volatility = np.std(yoy_growth)
        else:
            volatility = 0
        
        # Calculate other statistics
        mean_emissions = np.mean(emissions)
        std_emissions = np.std(emissions)
        min_emissions = np.min(emissions)
        max_emissions = np.max(emissions)
        
        # Compile statistics
        stats = {
            'cagr': cagr,
            'yoy_growth': yoy_growth,
            'volatility': volatility,
            'mean_emissions': mean_emissions,
            'std_emissions': std_emissions,
            'min_emissions': min_emissions,
            'max_emissions': max_emissions
        }
        
        return stats

if __name__ == "__main__":
    # Example usage
    import pandas as pd
    
    # Create sample historical data
    historical_data = pd.DataFrame({
        'year': [2020, 2021, 2022, 2023, 2024],
        'emissions': [10000, 10300, 10500, 10800, 11000]
    })
    
    # Initialize forecaster
    forecaster = TimeSeriesForecaster(historical_data)
    
    # Fit trend model
    model = forecaster.fit_exponential_trend()
    print("Trend Model:")
    print(f"Growth Rate: {model['growth_rate']*100:.2f}% per year")
    print(f"R²: {model['r_squared']:.4f}")
    
    # Generate forecast
    forecast = forecaster.forecast(2025, 2030)
    print("\nEmissions Forecast:")
    print(forecast)
    
    # Generate scenario forecasts
    scenarios = forecaster.forecast_with_scenarios(2025, 2030)
    print("\nScenario Forecasts:")
    for scenario_id, scenario_df in scenarios.items():
        print(f"\n{scenario_id}:")
        print(scenario_df[['year', 'emissions', 'scenario']].head(3))