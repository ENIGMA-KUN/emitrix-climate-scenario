"""
Streamlit app for the Emitrix Climate Scenario Analysis Platform.
This app provides an interactive dashboard for exploring the
Moderate Reduction scenario (Scenario 2).
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import models
from models.scope1_reduction_model import Scope1ReductionModel
from models.technology_calculator import TechnologyTransitionCalculator
from models.financial_model import FinancialImpactModel

# Set page config
st.set_page_config(
    page_title="Emitrix Climate Scenario Analysis Platform",
    page_icon="🌍",
    layout="wide"
)

# App title
st.title("Emitrix Climate Scenario Analysis Platform")
st.subheader("Scope 1 Emission Reduction Pathways - Moderate Reduction Scenario")

# Sidebar for parameters
st.sidebar.header("Scenario Parameters")

# Reduction target slider
reduction_target = st.sidebar.slider(
    "Reduction Target (%)",
    min_value=10,
    max_value=50,
    value=30,
    step=5,
    help="Target percentage reduction in Scope 1 emissions by the target year."
)

# Target year slider
target_year = st.sidebar.slider(
    "Target Year",
    min_value=2026,
    max_value=2035,
    value=2030,
    step=1,
    help="Year by which to achieve the reduction target."
)

# Carbon price parameters
carbon_price_start = st.sidebar.slider(
    "Carbon Price ($/tCO2e)",
    min_value=10,
    max_value=100,
    value=30,
    step=5,
    help="Starting carbon price in dollars per ton of CO2 equivalent."
)

carbon_price_increase = st.sidebar.slider(
    "Annual Carbon Price Increase (%)",
    min_value=0,
    max_value=20,
    value=5,
    step=1,
    help="Annual percentage increase in carbon price."
)

# Financial parameters
discount_rate = st.sidebar.slider(
    "Discount Rate (%)",
    min_value=2,
    max_value=15,
    value=7,
    step=1,
    help="Discount rate used for NPV calculations."
)

# Technology mix options
st.sidebar.header("Technology Mix")
st.sidebar.markdown("Percentage of source category converted by 2030:")

electric_boilers_pct = st.sidebar.slider(
    "Electric Boilers",
    min_value=0,
    max_value=100,
    value=50,
    step=10,
    help="Percentage of natural gas boilers converted to electric by target year."
)

electric_vehicles_pct = st.sidebar.slider(
    "Electric Vehicles",
    min_value=0,
    max_value=100,
    value=30,
    step=10,
    help="Percentage of fleet converted to electric vehicles by target year."
)

leak_detection_pct = st.sidebar.slider(
    "Refrigerant Leak Detection",
    min_value=0,
    max_value=100,
    value=80,
    step=10,
    help="Percentage of refrigerant systems with leak detection by target year."
)

# Run model button
run_model = st.sidebar.button("Run Model", type="primary")

# Initialize or reset session state
if "model_run" not in st.session_state:
    st.session_state.model_run = False
    st.session_state.results = None

# Run the model if button clicked
if run_model:
    # Convert percentage inputs to decimals
    reduction_target_decimal = reduction_target / 100
    carbon_price_increase_decimal = carbon_price_increase / 100
    discount_rate_decimal = discount_rate / 100
    
    # Show spinner during calculation
    with st.spinner("Running scenario analysis..."):
        # Initialize the model
        model = Scope1ReductionModel(
            reduction_target=reduction_target_decimal,
            target_year=target_year,
            start_year=2025,
            carbon_price_start=carbon_price_start,
            carbon_price_increase=carbon_price_increase_decimal,
            discount_rate=discount_rate_decimal
        )
        
        # Load sample data (or connected to real data in production)
        model.load_sample_data()
        
        # Run the scenario
        results = model.run_scenario()
        
        # Store results in session state
        st.session_state.model_run = True
        st.session_state.results = results
    
    st.success("Scenario analysis completed!")

# Display results if model has been run
if st.session_state.model_run and st.session_state.results is not None:
    results = st.session_state.results
    
    # Extract data from results
    historical_data = pd.DataFrame(results['historical_data'])
    projection_data = pd.DataFrame(results['projection_data'])
    reduced_data = pd.DataFrame(results['reduced_data'])
    financial_data = pd.DataFrame(results['financial_data'])
    metrics = results['metrics']
    
    # Create tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs([
        "Emission Reduction Pathway", 
        "Financial Impact", 
        "Technology Transition",
        "Key Metrics"
    ])
    
    # Tab 1: Emission Reduction Pathway
    with tab1:
        st.header("Emission Reduction Pathway")
        
        # Create DataFrame for plotting
        emission_path_data = pd.DataFrame({
            'Year': list(historical_data['year']) + list(projection_data['year']),
            'Emissions (tCO2e)': list(historical_data['emissions']) + list(projection_data['emissions']),
            'Scenario': ['Historical'] * len(historical_data) + ['Business as Usual'] * len(projection_data)
        })
        
        # Add reduced emissions
        reduced_emission_data = pd.DataFrame({
            'Year': list(reduced_data['year']),
            'Emissions (tCO2e)': list(reduced_data['reduced_emissions']),
            'Scenario': ['Moderate Reduction'] * len(reduced_data)
        })
        
        emission_path_data = pd.concat([emission_path_data, reduced_emission_data])
        
        # Create plot
        fig = px.line(
            emission_path_data, 
            x='Year', 
            y='Emissions (tCO2e)', 
            color='Scenario',
            line_dash='Scenario',
            markers=True,
            color_discrete_map={
                'Historical': 'gray',
                'Business as Usual': 'red',
                'Moderate Reduction': 'green'
            }
        )
        
        # Add target line
        target_value = projection_data[projection_data['year'] == target_year]['emissions'].values[0] * (1 - reduction_target/100)
        fig.add_hline(
            y=target_value, 
            line_dash="dash", 
            line_color="blue",
            annotation_text=f"{reduction_target}% Reduction Target",
            annotation_position="bottom right"
        )
        
        # Update layout
        fig.update_layout(
            height=500,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            title=f"Emission Reduction Pathway to {target_year}",
            hovermode="x unified"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Emission sources breakdown
        st.subheader("Emission Sources Breakdown")
        
        # Extract source columns
        source_columns = [col for col in projection_data.columns 
                         if col not in ['year', 'emissions']]
        
        reduced_source_columns = [f'reduced_{col}' for col in source_columns]
        
        # Create DataFrame for plotting sources
        sources_df = pd.melt(
            reduced_data,
            id_vars=['year'],
            value_vars=reduced_source_columns,
            var_name='Source',
            value_name='Emissions'
        )
        
        # Clean up source names
        sources_df['Source'] = sources_df['Source'].str.replace('reduced_', '').str.replace('_', ' ').str.title()
        
        # Create stacked area chart
        fig_sources = px.area(
            sources_df,
            x='year',
            y='Emissions',
            color='Source',
            title=f"Emissions by Source Category - Moderate Reduction Scenario"
        )
        
        fig_sources.update_layout(
            height=400,
            hovermode="x unified",
            legend_title="Source Category"
        )
        
        st.plotly_chart(fig_sources, use_container_width=True)
        
    # Tab 2: Financial Impact
    with tab2:
        st.header("Financial Impact Analysis")
        
        # Create cost breakdown chart
        financial_cols = ['year', 'implementation_cost', 'operational_change', 'carbon_savings']
        if all(col in financial_data.columns for col in financial_cols):
            cost_data = financial_data[financial_cols].copy()
            
            # Calculate total costs/savings
            cost_data['net_impact'] = (
                cost_data['carbon_savings'] - 
                cost_data['implementation_cost'] - 
                cost_data['operational_change']
            )
            
            # Melt for plotting
            cost_data_melted = pd.melt(
                cost_data,
                id_vars=['year'],
                value_vars=['implementation_cost', 'operational_change', 'carbon_savings', 'net_impact'],
                var_name='Cost Type',
                value_name='Amount ($)'
            )
            
            # Clean up cost type names
            cost_data_melted['Cost Type'] = cost_data_melted['Cost Type'].replace({
                'implementation_cost': 'Implementation Cost',
                'operational_change': 'Operational Change',
                'carbon_savings': 'Carbon Cost Savings',
                'net_impact': 'Net Impact'
            })
            
            # Create bar chart
            fig_costs = px.bar(
                cost_data_melted,
                x='year',
                y='Amount ($)',
                color='Cost Type',
                barmode='group',
                title="Financial Impact by Year"
            )
            
            fig_costs.update_layout(
                height=400,
                hovermode="x unified",
                legend_title="Cost Type"
            )
            
            st.plotly_chart(fig_costs, use_container_width=True)
            
            # Create cumulative financial impact chart
            cost_data['cumulative_net_impact'] = cost_data['net_impact'].cumsum()
            
            fig_cumulative = px.line(
                cost_data,
                x='year',
                y='cumulative_net_impact',
                markers=True,
                title="Cumulative Financial Impact"
            )
            
            fig_cumulative.update_layout(
                height=350,
                hovermode="x unified",
                yaxis_title="Cumulative Net Impact ($)"
            )
            
            fig_cumulative.add_hline(
                y=0, 
                line_dash="dash", 
                line_color="red",
                annotation_text="Break-even point",
                annotation_position="bottom right"
            )
            
            st.plotly_chart(fig_cumulative, use_container_width=True)
            
            # Carbon price impact
            st.subheader("Carbon Price Impact")
            
            carbon_price_data = pd.DataFrame({
                'Year': list(range(2025, target_year + 1)),
                'Carbon Price ($/tCO2e)': [
                    carbon_price_start * (1 + carbon_price_increase / 100) ** (year - 2025)
                    for year in range(2025, target_year + 1)
                ]
            })
            
            fig_carbon_price = px.line(
                carbon_price_data,
                x='Year',
                y='Carbon Price ($/tCO2e)',
                markers=True,
                title="Carbon Price Trajectory"
            )
            
            fig_carbon_price.update_layout(
                height=300,
                hovermode="x unified"
            )
            
            st.plotly_chart(fig_carbon_price, use_container_width=True)
        
    # Tab 3: Technology Transition
    with tab3:
        st.header("Technology Transition Timeline")
        
        # Create technology transition chart (simplified version)
        tech_data = {
            'Year': list(range(2025, target_year + 1)),
            'Electric Boilers': [0] + [min(electric_boilers_pct * (year - 2025) / (target_year - 2025), electric_boilers_pct) for year in range(2026, target_year + 1)],
            'Electric Vehicles': [0] + [min(electric_vehicles_pct * (year - 2025) / (target_year - 2025), electric_vehicles_pct) for year in range(2026, target_year + 1)],
            'Refrigerant Leak Detection': [0] + [min(leak_detection_pct * (year - 2025) / (target_year - 2025), leak_detection_pct) for year in range(2026, target_year + 1)]
        }
        
        tech_df = pd.DataFrame(tech_data)
        
        # Melt for plotting
        tech_df_melted = pd.melt(
            tech_df,
            id_vars=['Year'],
            value_vars=['Electric Boilers', 'Electric Vehicles', 'Refrigerant Leak Detection'],
            var_name='Technology',
            value_name='Implementation Percentage'
        )
        
        # Create line chart
        fig_tech = px.line(
            tech_df_melted,
            x='Year',
            y='Implementation Percentage',
            color='Technology',
            markers=True,
            title="Technology Implementation Timeline"
        )
        
        fig_tech.update_layout(
            height=400,
            hovermode="x unified",
            yaxis=dict(ticksuffix="%")
        )
        
        st.plotly_chart(fig_tech, use_container_width=True)
        
        # Technology emission reductions
        st.subheader("Emission Reductions by Technology")
        
        # Create example data for demonstration
        tech_impact = pd.DataFrame({
            'Technology': ['Electric Boilers', 'Electric Vehicles', 'Refrigerant Leak Detection', 
                           'High-Efficiency Boilers', 'Hybrid Vehicles', 'Process Optimization'],
            'Emission Reduction (tCO2e)': [
                metrics['reduction_achieved_tco2e'] * 0.4,  # 40% from electric boilers
                metrics['reduction_achieved_tco2e'] * 0.25,  # 25% from electric vehicles
                metrics['reduction_achieved_tco2e'] * 0.15,  # 15% from leak detection
                metrics['reduction_achieved_tco2e'] * 0.1,   # 10% from high-efficiency boilers
                metrics['reduction_achieved_tco2e'] * 0.05,  # 5% from hybrid vehicles
                metrics['reduction_achieved_tco2e'] * 0.05   # 5% from process optimization
            ],
            'Cost per tCO2e ($)': [220, 180, 90, 120, 150, 200]
        })
        
        # Sort by cost effectiveness
        tech_impact = tech_impact.sort_values('Cost per tCO2e ($)')
        
        # Create bubble chart
        fig_impact = px.scatter(
            tech_impact,
            x='Cost per tCO2e ($)',
            y='Emission Reduction (tCO2e)',
            size='Emission Reduction (tCO2e)',
            color='Technology',
            title="Technology Impact and Cost Effectiveness"
        )
        
        fig_impact.update_layout(
            height=400,
            hovermode="closest"
        )
        
        st.plotly_chart(fig_impact, use_container_width=True)
        
    # Tab 4: Key Metrics
    with tab4:
        st.header("Key Performance Indicators")
        
        # Create columns for metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="Total Emission Reduction",
                value=f"{metrics['reduction_percentage']*100:.1f}%",
                delta=f"{metrics['reduction_percentage']*100 - reduction_target:.1f}% vs target"
            )
            
            st.metric(
                label="Total Implementation Cost",
                value=f"${metrics['total_implementation_cost']:,.0f}"
            )
            
        with col2:
            st.metric(
                label="Net Present Value (NPV)",
                value=f"${metrics['net_present_value']:,.0f}"
            )
            
            st.metric(
                label="Return on Investment (ROI)",
                value=f"{metrics['return_on_investment']*100:.1f}%"
            )
            
        with col3:
            st.metric(
                label="Payback Period",
                value=f"{metrics['payback_years']:.1f} years"
            )
            
            st.metric(
                label="Marginal Abatement Cost",
                value=f"${metrics['marginal_abatement_cost']:.0f}/tCO2e"
            )
        
        # Additional metrics table
        st.subheader("Detailed Metrics")
        
        detailed_metrics = pd.DataFrame({
            'Metric': [
                'BAU Emissions in Target Year',
                'Reduced Emissions in Target Year',
                'Absolute Reduction from BAU',
                'Annual Implementation Cost',
                'Total Carbon Savings',
                'Total Operational Changes'
            ],
            'Value': [
                f"{metrics['target_year_bau_emissions']:,.1f} tCO2e",
                f"{metrics['target_year_reduced_emissions']:,.1f} tCO2e",
                f"{metrics['reduction_achieved_tco2e']:,.1f} tCO2e",
                f"${metrics['annual_implementation_cost']:,.0f}",
                f"${metrics['total_carbon_savings']:,.0f}",
                f"${metrics['total_operational_change']:,.0f}"
            ]
        })
        
        st.table(detailed_metrics)
        
        # Scenario summary
        st.subheader("Scenario Summary")
        
        st.markdown(f"""
        The **Moderate Reduction** scenario achieves a **{metrics['reduction_percentage']*100:.1f}%** reduction in Scope 1 emissions by {target_year} 
        through a balanced approach to technology implementation and operational changes.
        
        Key technology transitions include:
        - Converting **{electric_boilers_pct}%** of natural gas boilers to electric boilers
        - Transitioning **{electric_vehicles_pct}%** of the vehicle fleet to electric vehicles
        - Implementing refrigerant leak detection in **{leak_detection_pct}%** of cooling systems
        
        The financial analysis shows a positive NPV of **${metrics['net_present_value']:,.0f}** with a payback period of 
        **{metrics['payback_years']:.1f} years**, indicating this emission reduction pathway is economically viable 
        under the assumed carbon price starting at **${carbon_price_start}** per tCO2e with an annual increase of **{carbon_price_increase}%**.
        """)
        
        if metrics['return_on_investment'] > 0:
            st.success(f"This scenario is financially beneficial with a positive ROI of {metrics['return_on_investment']*100:.1f}%.")
        else:
            st.warning(f"This scenario has a negative ROI of {metrics['return_on_investment']*100:.1f}%. Consider adjusting parameters or technology mix.")

else:
    # Display instructions if model hasn't been run yet
    st.info("👈 Adjust the scenario parameters in the sidebar and click 'Run Model' to see the results.")
    
    st.markdown("""
    ### Emitrix Climate Scenario Analysis Platform
    
    This tool helps organizations model and evaluate Scope 1 emissions reduction pathways.
    
    #### Key Features:
    - Model emissions reduction scenarios
    - Evaluate financial impacts of carbon pricing
    - Analyze technology transition pathways
    - Generate key performance metrics
    
    #### Getting Started:
    1. Set your reduction target and target year
    2. Adjust carbon price parameters
    3. Define your technology implementation mix
    4. Click 'Run Model' to see the results
    """)

# Footer
st.markdown("---")
st.caption("Emitrix Climate Scenario Analysis Platform - Developed for emissions reduction pathway modeling and financial impact analysis")