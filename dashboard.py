"""
Browser-Based Dashboard for Taxi Claims Analysis
Task 3: Interactive dashboard for presenting insights and recommendations

Run with: streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from scipy import stats

# Page configuration
st.set_page_config(
    page_title="Taxi Claims Analysis Dashboard",
    page_icon="🚕",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .insight-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
    .recommendation-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep']

# Load data
@st.cache_data
def load_data():
    """Load and cache the processed taxi data"""
    df = pd.read_parquet('master.parquet')
    return df

# Load data
df = load_data()

# Sidebar for navigation
st.sidebar.title("🚕 Navigation")
page = st.sidebar.radio(
    "Go to:",
    ["Executive Summary", "Monthly Trends", "Hourly Patterns", "Geographic Insights", "Cost Reduction Opportunities", "Thought Process & Disclaimer"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info(
    "This dashboard analyzes taxi claim data to identify cost-saving opportunities "
    "and assess spending patterns."
)

# ============================================================================
# PAGE 1: EXECUTIVE SUMMARY
# ============================================================================
if page == "Executive Summary":
    st.markdown('<p class="main-header">🚕 Taxi Claims Analysis Dashboard</p>', unsafe_allow_html=True)
    # Key metrics
    st.markdown("## 📊 Key Metrics")
    col1, col2, col3, col4 = st.columns(4)

    total_trips = len(df)
    total_spend = df['taxi_fare'].sum()
    avg_fare = df['taxi_fare'].mean()
    avg_distance = df['distance_km'].mean()

    with col1:
        st.metric("Total Trips", f"{total_trips:,}")
    with col2:
        st.metric("Total Spend", f"${total_spend:,.2f}")
    with col3:
        st.metric("Average Fare", f"${avg_fare:.2f}")
    with col4:
        st.metric("Average Distance", f"{avg_distance:.2f} km")

    st.markdown("---")

    # Key findings
    st.markdown("## 🔍 Key Findings")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <h3>✅ Spending is NOT Increasing</h3>
        <ul>
        <li>Monthly spend shows a <strong>flat to slightly declining trend</strong></li>
        <li>Cost per ride has been <strong>stable</strong> throughout the period</li>
        <li>No statistical evidence of concerning spending growth</li>
        </ul>
        """, unsafe_allow_html=True)

        st.markdown("""
        <h3>🕖 7PM Spike: Presenteeism Issue</h3>
        <ul>
        <li><strong>Massive spike</strong> in rides immediately after 6-7PM cutoff</li>
        <li>Suggests officers may be staying late to claim taxi reimbursement</li>
        <li>Spend increases dramatically in first 30 minutes past 7PM</li>
        </ul>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <h3>🍽️ Lunch Hour Trips</h3>
        <ul>
        <li><strong>17% of all trips</strong> involve Bendemeer Food Centre during lunch</li>
        <li>Indicates limited food options near workplace</li>
        <li>Opportunity for lunch shuttle service</li>
        </ul>
        """, unsafe_allow_html=True)

        st.markdown("""
        <h3>🚗 Ride Pooling Opportunities</h3>
        <ul>
        <li>High concentration of trips from <strong>MOM Service Center</strong></li>
        <li>Many officers traveling to same destinations at same times</li>
        <li>Top route: MOM Service Center → MOM HQ (1PM)</li>
        </ul>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Recommendations
    st.markdown("## 💡 Recommendations")

    # Calculate savings for waterfall chart
    df_valid_exec = df[df['pickup_postal'].notna() & df['destination_postal'].notna()].copy()
    df_valid_exec['trip_pair'] = df_valid_exec['pickup_postal'] + ' → ' + df_valid_exec['destination_postal']

    # Get routes from MOM Service Center
    trip_pair_counts = df_valid_exec['trip_pair'].value_counts()
    top_10_pairs = trip_pair_counts.head(10)
    routes_from_339946 = [pair for pair in top_10_pairs.index if pair.startswith('339946')]
    df_339946_exec = df_valid_exec[df_valid_exec['trip_pair'].isin(routes_from_339946)]
    hourly_route_costs_exec = df_339946_exec.groupby(['derived_trip_start_hour', 'trip_pair'])['taxi_fare'].sum().reset_index()

    # Calculate savings
    reduction_by_hour = {13: 0.5, 19: 0.33, 20: 0.25}
    base_cost_by_hour = (
        hourly_route_costs_exec[hourly_route_costs_exec["derived_trip_start_hour"].isin(reduction_by_hour.keys())]
        .groupby("derived_trip_start_hour")["taxi_fare"]
        .sum()
    )

    descriptions = {
        13: "Pooling rides from MOM Svc Ctr at 1 PM",
        19: "Pooling rides from MOM Svc Ctr at 7 PM",
        20: "Pooling rides from MOM Svc Ctr at 8 PM"
    }

    savings_data_exec = []
    for hour in base_cost_by_hour.index:
        savings_data_exec.append({
            'Initiative': descriptions[hour],
            'Current Cost': base_cost_by_hour.loc[hour],
            'Reduction %': f"{reduction_by_hour[hour]*100:.0f}%",
            'Estimated Savings': base_cost_by_hour.loc[hour] * reduction_by_hour[hour]
        })

    # Add presenteeism savings
    spend_at_19 = df[df['derived_trip_start_hour'] == 19]['taxi_fare'].sum()
    reduction_7_15pm = 0.30
    savings_7_15pm = spend_at_19 * reduction_7_15pm

    savings_data_exec.append({
        'Initiative': 'Cutting back presenteeism 7-7:15pm',
        'Current Cost': spend_at_19,
        'Reduction %': '30%',
        'Estimated Savings': savings_7_15pm
    })

    savings_df_exec = pd.DataFrame(savings_data_exec)
    total_savings_exec = savings_df_exec['Estimated Savings'].sum()
    total_spend_exec = df['taxi_fare'].sum()

    st.markdown("""
    <h3>Estimated Savings: <strong>~10% of Total Spend</strong></h3>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Waterfall chart
    current_total = total_spend_exec
    final_total = current_total - total_savings_exec

    categories = ['Current Total'] + savings_df_exec['Initiative'].tolist() + ['Final Total']
    values = [current_total] + [-s for s in savings_df_exec['Estimated Savings'].tolist()] + [final_total]

    # Calculate measure for waterfall
    measure = ['absolute'] + ['relative'] * len(savings_df_exec) + ['total']

    fig_waterfall = go.Figure(go.Waterfall(
        orientation="v",
        measure=measure,
        x=categories,
        y=values,
        text=[f'${v:,.0f}' for v in values],
        textposition="outside",
        connector={"line": {"color": "rgb(63, 63, 63)"}},
        increasing={"marker": {"color": "green"}},
        decreasing={"marker": {"color": "green"}},
        totals={"marker": {"color": "orange"}}
    ))

    fig_waterfall.update_layout(
        title="10% Cost Reduction Through Ride Pooling and Reducing Presenteeism",
        yaxis_title="Amount ($)",
        height=500,
        showlegend=False
    )

    st.plotly_chart(fig_waterfall, use_container_width=True)

    st.markdown("""
    <h3>1. Ride Pooling Program</h3>
    <ul>
    <li><strong>Target:</strong> Common routes from MOM Service Center, especially at 1PM (to HQ) and 7-8PM (to MRT stations)</li>
    <li><strong>Expected Savings:</strong> $1,183</li>
    <li><strong>Implementation:</strong> 
        <ul>
        <li>Introduce guidance on pooling of trips wherever sensible: such as trips from the same department.</li>
        <li>To match unaffiliated officers for pooling, consider a designated meeting zone at MOM Service Centre driveway for all rides heading to common destinations.</li>
        <li>For trips to HQ at 1pm, consider a fixed light shuttle service.</li>
        </ul>
    </li>
    </ul>

    <h3>2. Address Presenteeism</h3>
    <ul>
    <li><strong>Target:</strong> Reduce 7-7:15PM trips by 30%</li>
    <li><strong>Expected Savings:</strong> $418</li>
    <li><strong>Implementation:</strong>
        <ul>
        <li>Manager communication encouraging timely departure</li>
        <li>Explore tiered reimbursement rates (e.g., 50% at 6:30PM, 75% at 6:45PM, 100% at 7PM)</li>
        </ul>
    </li>
    </ul>

    <h3>3. Lunch Shuttle Service (Not modelled here/To be explored)</h3>
    <ul>
    <li><strong>Target:</strong> A reduction of 10-20% should be possible here, but more consultation with staff should be done to accurately treat the issue. 
                The signal is less clear than for trips originating from MOM Service Centre; <br>there could be multiple factors leading to concentrated trips at Bendemeer beyond lunch crush. </li>
    <li><strong>Expected Impact:</strong> Reduction in lunch-hour trips</li>
    <li><strong>Implementation:</strong> A possible shuttle service during lunch hours (~10AM-2PM)</li>
    </ul>
    """, unsafe_allow_html=True)

# ============================================================================
# PAGE 2: MONTHLY TRENDS
# ============================================================================
elif page == "Monthly Trends":
    st.markdown("# 📈 Spending Trends Over Time")

    tab1, tab2 = st.tabs(["Monthly Total Spend", "Cost Per Ride"])

    with tab1:
        st.markdown("## Total Spend by Month")

        # Calculate monthly spend
        monthly_spend = (
            df.groupby('derived_month', as_index=False)['taxi_fare']
              .sum()
              .rename(columns={'derived_month': 'Month', 'taxi_fare': 'Total Spend'})
              .sort_values('Month')
        )

        # Create figure
        fig = go.Figure()

        # Add bar chart
        fig.add_trace(go.Bar(
            x=monthly_spend['Month'],
            y=monthly_spend['Total Spend'],
            name='Monthly Spend',
            marker_color='steelblue'
        ))

        # Add trend line
        x = np.arange(len(monthly_spend))
        y = monthly_spend['Total Spend'].to_numpy()
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

        fig.add_trace(go.Scatter(
            x=monthly_spend['Month'],
            y=slope * x + intercept,
            mode='lines',
            name='Trend Line',
            line=dict(color='orange', width=3)
        ))

        fig.update_layout(
            xaxis=dict(
                tickmode='array',
                tickvals=monthly_spend['Month'],
                ticktext=[month_names[int(m)-1] for m in monthly_spend['Month']]
            ),
            yaxis_title='Total Spend ($)',
            xaxis_title='Month (2015)',
            title='Monthly Taxi Spending - No Increasing Trend',
            hovermode='x unified',
            height=500
        )

        st.plotly_chart(fig, use_container_width=True)

        st.info("📌 The trend line shows a slight decreasing slope, but it is **not statistically significant** (p > 0.05)")

        # Calculate statistics
        median_spend = monthly_spend['Total Spend'].median()
        std_spend = monthly_spend['Total Spend'].std()
        min_spend = monthly_spend['Total Spend'].min()
        max_spend = monthly_spend['Total Spend'].max()
        min_month = monthly_spend.loc[monthly_spend['Total Spend'].idxmin(), 'Month']
        max_month = monthly_spend.loc[monthly_spend['Total Spend'].idxmax(), 'Month']

        # First row of metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Trend Slope*", f"${slope:.2f}/month")
        with col2:
            st.metric("Median", f"${median_spend:,.2f}")
        with col3:
            st.metric("Standard Deviation", f"${std_spend:,.2f}")

        # Second row of metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("P-value*", f"{p_value:.4f}")
        with col2:
            st.metric("Min", f"${min_spend:,.2f} ({month_names[int(min_month)-1]})")
        with col3:
            st.metric("Max", f"${max_spend:,.2f} ({month_names[int(max_month)-1]})")

    with tab2:
        st.markdown("## Cost Per Ride Over Time")

        # Calculate monthly statistics
        monthly_stats = df.groupby('derived_month').agg({
            'taxi_fare': ['mean', 'median',
                          lambda x: x.quantile(0.1),
                          lambda x: x.quantile(0.9)]
        }).reset_index()

        monthly_stats.columns = ['Month', 'Average', 'Median', '10th Percentile', '90th Percentile']

        # Create figure
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=monthly_stats['Month'],
            y=monthly_stats['Average'],
            mode='lines+markers',
            name='Average',
            line=dict(color='#1f77b4', width=2)
        ))

        fig.add_trace(go.Scatter(
            x=monthly_stats['Month'],
            y=monthly_stats['Median'],
            mode='lines+markers',
            name='Median',
            line=dict(color='#ff7f0e', width=2)
        ))

        fig.add_trace(go.Scatter(
            x=monthly_stats['Month'],
            y=monthly_stats['10th Percentile'],
            mode='lines+markers',
            name='10th Percentile',
            line=dict(color='#2ca02c', width=2, dash='dash')
        ))

        fig.add_trace(go.Scatter(
            x=monthly_stats['Month'],
            y=monthly_stats['90th Percentile'],
            mode='lines+markers',
            name='90th Percentile',
            line=dict(color='#d62728', width=2, dash='dash')
        ))

        fig.update_layout(
            xaxis=dict(
                tickmode='array',
                tickvals=monthly_stats['Month'],
                ticktext=[month_names[int(m)-1] for m in monthly_stats['Month']]
            ),
            yaxis_title='Fare per Ride ($)',
            xaxis_title='Month (2015)',
            title='Cost of Rides Has Been Stable',
            hovermode='x unified',
            height=500
        )

        st.plotly_chart(fig, use_container_width=True)

        # Regression on median
        x = np.arange(len(monthly_stats))
        y = monthly_stats['Median'].to_numpy()
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

        st.info("📌 The median fare shows stable costs over time (p > 0.05 = not significant)")

        # Calculate statistics
        median_fare = monthly_stats['Median'].median()
        std_fare = monthly_stats['Median'].std()
        min_fare = monthly_stats['Median'].min()
        max_fare = monthly_stats['Median'].max()
        min_month = monthly_stats.loc[monthly_stats['Median'].idxmin(), 'Month']
        max_month = monthly_stats.loc[monthly_stats['Median'].idxmax(), 'Month']

        # First row of metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Trend Slope*", f"${slope:.4f}/month")
        with col2:
            st.metric("Median", f"${median_fare:.2f}")
        with col3:
            st.metric("Standard Deviation", f"${std_fare:.2f}")

        # Second row of metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("P-value*", f"{p_value:.4f}")
        with col2:
            st.metric("Min", f"${min_fare:.2f} ({month_names[int(min_month)-1]})")
        with col3:
            st.metric("Max", f"${max_fare:.2f} ({month_names[int(max_month)-1]})")

# ============================================================================
# PAGE 3: HOURLY PATTERNS
# ============================================================================
elif page == "Hourly Patterns":
    st.markdown("# ⏰ Trip Patterns by Time of Day")

    st.markdown("## Total Spend by Hour of Day")

    # Calculate hourly spend
    hourly_spend = (
        df.groupby('derived_trip_start_hour', as_index=False)['taxi_fare']
          .sum()
          .rename(columns={'derived_trip_start_hour': 'Hour', 'taxi_fare': 'Total Spend'})
          .sort_values('Hour')
    )

    # Categorize hours
    def categorize_hours(hour):
        if 19 <= hour <= 23 or 0 <= hour <= 6:
            return 'Beyond Office Hours'
        return 'Office Hours'

    hourly_spend['Category'] = hourly_spend['Hour'].apply(categorize_hours)

    # Create figure
    fig = go.Figure()

    office_hours = hourly_spend[hourly_spend['Category'] == 'Office Hours']
    beyond_hours = hourly_spend[hourly_spend['Category'] == 'Beyond Office Hours']

    fig.add_trace(go.Bar(
        x=office_hours['Hour'],
        y=office_hours['Total Spend'],
        name='Office Hours (7am-6pm)',
        marker_color='steelblue'
    ))

    fig.add_trace(go.Bar(
        x=beyond_hours['Hour'],
        y=beyond_hours['Total Spend'],
        name='Beyond Office Hours',
        marker_color='orange'
    ))

    # Add vertical line at 7pm
    fig.add_vline(x=18.5, line_dash="dash", line_color="red", line_width=2,
                 annotation_text="Before/After 7PM", annotation_position="top")

    fig.update_layout(
        xaxis=dict(
            tickmode='array',
            tickvals=list(range(24)),
            ticktext=[f'{h}:00' for h in range(24)]
        ),
        yaxis_title='Total Spend ($)',
        xaxis_title='Hour of Day',
        title='Massive Spike After 7PM - Evidence of Presenteeism',
        hovermode='x unified',
        height=500,
        barmode='group'
    )

    st.plotly_chart(fig, use_container_width=True)

    st.warning("""
    **Key Insight:** Notice the dramatic spike between the 6-7PM boundary. This suggests
    possible **presenteeism** where officers stay just past 7PM to qualify for taxi reimbursement.
    """)

    # Add spacing
    st.markdown("---")

    st.markdown("## Detailed 7PM Spike Analysis (15-minute intervals)")

    # Create 15-minute bins for 6-8pm
    df_local = df.copy()
    df_local['minute'] = pd.to_datetime(df_local['derived_trip_start_datetime']).dt.minute
    df_local['minute_bin'] = (df_local['minute'] // 15) * 15
    df_local['hour'] = df_local['derived_trip_start_hour']

    df_6_to_8 = df_local[df_local["hour"].isin([18, 19])]

    fifteen_min_spend = (
        df_6_to_8.groupby(["hour", "minute_bin"], as_index=False)["taxi_fare"]
        .sum()
    )

    def create_time_label(hour, minute):
        hour = int(hour)
        minute = int(minute)
        period = "AM" if hour < 12 else "PM"
        display_hour = hour % 12
        if display_hour == 0:
            display_hour = 12
        return f"{display_hour}:{minute:02d} {period}"

    fifteen_min_spend["time_label"] = fifteen_min_spend.apply(
        lambda row: create_time_label(row["hour"], row["minute_bin"]), axis=1
    )

    fifteen_min_spend = fifteen_min_spend.sort_values(["hour", "minute_bin"])

    # Color based on category
    colors = ['steelblue' if h < 19 else 'orange' for h in fifteen_min_spend['hour']]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=fifteen_min_spend['time_label'],
        y=fifteen_min_spend['taxi_fare'],
        marker_color=colors,
        text=fifteen_min_spend['taxi_fare'].round(2),
        textposition='outside',
        texttemplate='$%{text}'
    ))

    # Add vertical line between 6:45pm and 7:00pm
    fig.add_vline(x=3.5, line_dash="dash", line_color="red", line_width=2,
                 annotation_text="7:00 PM", annotation_position="top")

    fig.update_layout(
        xaxis_title='Time',
        yaxis_title='Total Spend ($)',
        title='Evidence of Policy Exploitation - Spend Jumps Immediately After 7PM',
        height=500,
        showlegend=False
    )

    st.plotly_chart(fig, use_container_width=True)

    st.error("""
    **🚨 Critical Finding:** Spend increases dramatically in the first 30 minutes past 7PM.
    This strongly suggests officers are timing their trips to qualify for reimbursement rather
    than based on genuine overtime needs.
    """)

    st.markdown("### Recommended Actions")
    st.markdown("""
    1. **Encourage early departure** - Managers should actively encourage staff to return home at 6PM unless overtime is absolutely necessary
    2. **Tiered reimbursement** - Instead of all-or-nothing at 7PM:
       - 50% reimbursement after 6:30PM
       - 75% reimbursement after 6:45PM
       - 100% reimbursement after 7:00PM
    3. **Flat-rate caps** - Limit maximum reimbursement for trips taken 6:30-7:00PM to prevent subsidizing expensive prime-time rides
    """)

    # Add spacing
    st.markdown("---")

    st.markdown("## Interactive Hour Analysis")
    st.markdown("To examine other time-windows, adjust the time slider here.")

    # Time slider for selecting start hour (0-22, allowing for 2-hour window)
    selected_start_hour = st.slider(
        "Select Start Hour (2-hour window)",
        min_value=0,
        max_value=22,
        value=19,
        format="%d:00"
    )

    # Create 15-minute bins for selected 2-hour window
    df_interactive = df.copy()
    df_interactive['minute'] = pd.to_datetime(df_interactive['derived_trip_start_datetime']).dt.minute
    df_interactive['minute_bin'] = (df_interactive['minute'] // 15) * 15
    df_interactive['hour'] = df_interactive['derived_trip_start_hour']

    # Filter for selected 2-hour window
    end_hour = selected_start_hour + 1
    df_selected = df_interactive[df_interactive["hour"].isin([selected_start_hour, end_hour])]

    # Create helper function for time labels
    def create_time_label_interactive(hour, minute):
        hour = int(hour)
        minute = int(minute)
        period = "AM" if hour < 12 else "PM"
        display_hour = hour % 12
        if display_hour == 0:
            display_hour = 12
        return f"{display_hour}:{minute:02d} {period}"

    # Create complete set of all possible 15-minute intervals for 2-hour window
    all_intervals = []
    for h in [selected_start_hour, end_hour]:
        for m in [0, 15, 30, 45]:
            all_intervals.append({
                'hour': h,
                'minute_bin': m,
                'time_label': create_time_label_interactive(h, m),
                'taxi_fare': 0.0
            })

    complete_df = pd.DataFrame(all_intervals)

    # Merge with actual data if it exists
    if len(df_selected) > 0:
        actual_spend = (
            df_selected.groupby(["hour", "minute_bin"], as_index=False)["taxi_fare"]
            .sum()
        )

        # Update complete_df with actual values where they exist
        for _, row in actual_spend.iterrows():
            mask = (complete_df['hour'] == row['hour']) & (complete_df['minute_bin'] == row['minute_bin'])
            complete_df.loc[mask, 'taxi_fare'] = row['taxi_fare']

    # Sort by hour and minute_bin to ensure correct order
    complete_df = complete_df.sort_values(['hour', 'minute_bin'])

    # Dynamic coloring based on office hours (7am-6pm = office, rest = beyond)
    colors_interactive = []
    for h in complete_df['hour']:
        if 7 <= h <= 18:
            colors_interactive.append('steelblue')
        else:
            colors_interactive.append('orange')

    fig_interactive = go.Figure()

    fig_interactive.add_trace(go.Bar(
        x=complete_df['time_label'],
        y=complete_df['taxi_fare'],
        marker_color=colors_interactive,
        text=complete_df['taxi_fare'].round(2),
        textposition='outside',
        texttemplate='$%{text}'
    ))

    # Add vertical line at the hour mark (between first and second hour)
    hour_mark_period = "AM" if end_hour < 12 else "PM"
    hour_mark_display = end_hour % 12 if end_hour % 12 != 0 else 12
    fig_interactive.add_vline(
        x=3.5,
        line_dash="dash",
        line_color="red",
        line_width=2,
        annotation_text=f"{hour_mark_display}:00 {hour_mark_period}",
        annotation_position="top"
    )

    # Create descriptive title
    start_period = "AM" if selected_start_hour < 12 else "PM"
    start_display = selected_start_hour % 12 if selected_start_hour % 12 != 0 else 12
    end_display = end_hour % 12 if end_hour % 12 != 0 else 12
    end_period = "AM" if end_hour < 12 else "PM"

    fig_interactive.update_layout(
        xaxis_title='Time (15-minute intervals)',
        yaxis_title='Total Spend ($)',
        title=f'Spending Pattern: {start_display}:00 {start_period} - {end_display}:59 {end_period}',
        height=500,
        showlegend=False
    )

    st.plotly_chart(fig_interactive, use_container_width=True)

    # Display metrics
    total_spend_window = complete_df['taxi_fare'].sum()
    total_trips_window = len(df_selected)
    avg_fare_window = total_spend_window / total_trips_window if total_trips_window > 0 else 0

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Trips", f"{total_trips_window:,}")
    with col2:
        st.metric("Total Spend", f"${total_spend_window:.2f}")
    with col3:
        st.metric("Average Fare", f"${avg_fare_window:.2f}")

# ============================================================================
# PAGE 4: GEOGRAPHIC INSIGHTS
# ============================================================================
elif page == "Geographic Insights":
    st.markdown("# 🗺️ Geographic Trip Distribution")

    st.markdown("""
    This section analyzes where trips originate and end, revealing patterns that can guide
    ride pooling initiatives. Dot size indicates trip concentration at each location.
    """)

    # Filter valid coordinates
    df_valid = df[
        df['pickup_latitude'].notna() &
        df['pickup_longitude'].notna() &
        df['destination_latitude'].notna() &
        df['destination_longitude'].notna()
    ].copy()

    # Hour selector
    selected_hour = st.slider("Select Hour to View", 0, 23, 19, format="%d:00")

    hour_df = df_valid[df_valid['derived_trip_start_hour'] == selected_hour]

    # Helper function to aggregate nearby locations
    def aggregate_locations(lat_list, lon_list, precision=3):
        """
        Aggregate nearby locations and count occurrences
        precision=3 means ~100m grouping
        """
        if len(lat_list) == 0:
            return [], [], []

        locations = pd.DataFrame({
            'lat': lat_list,
            'lon': lon_list
        })
        locations['lat_round'] = locations['lat'].round(precision)
        locations['lon_round'] = locations['lon'].round(precision)

        # Count occurrences at each location
        agg = locations.groupby(['lat_round', 'lon_round']).size().reset_index(name='count')

        return agg['lat_round'].tolist(), agg['lon_round'].tolist(), agg['count'].tolist()

    # Helper function to scale marker sizes
    def scale_size(counts, min_size=8, max_size=50, global_max=None):
        """
        Scale counts to marker sizes using square root for better perception
        """
        if not counts or len(counts) == 0:
            return []

        counts_array = np.array(counts)
        sqrt_counts = np.sqrt(counts_array)
        sqrt_min = np.sqrt(1)
        sqrt_max = np.sqrt(global_max) if global_max else np.sqrt(max(counts))

        if sqrt_max == sqrt_min:
            return [min_size] * len(counts)

        normalized = (sqrt_counts - sqrt_min) / (sqrt_max - sqrt_min)
        return (normalized * (max_size - min_size) + min_size).tolist()

    # Check if there are trips at this hour
    if len(hour_df) > 0:
        # Aggregate pickup and destination locations
        pickup_lat, pickup_lon, pickup_counts = aggregate_locations(
            hour_df['pickup_latitude'].tolist(),
            hour_df['pickup_longitude'].tolist(),
            precision=3
        )

        dest_lat, dest_lon, dest_counts = aggregate_locations(
            hour_df['destination_latitude'].tolist(),
            hour_df['destination_longitude'].tolist(),
            precision=3
        )

        # Find global max for consistent scaling
        all_counts = pickup_counts + dest_counts
        global_max_count = max(all_counts) if all_counts else 1

        # Calculate sizes
        pickup_sizes = scale_size(pickup_counts, global_max=global_max_count)
        dest_sizes = scale_size(dest_counts, global_max=global_max_count)

        # Calculate map center
        center_lat = df_valid[['pickup_latitude', 'destination_latitude']].values.mean()
        center_lon = df_valid[['pickup_longitude', 'destination_longitude']].values.mean()

        # Create unified map
        fig = go.Figure()

        # PICKUP LOCATIONS - OUTLINE (RED) - larger, opaque
        if pickup_lat:
            fig.add_trace(go.Scattermapbox(
                lat=pickup_lat,
                lon=pickup_lon,
                mode='markers',
                marker=dict(
                    size=[s * 1.2 for s in pickup_sizes],
                    color='rgb(220, 100, 20)',
                    opacity=0.8
                ),
                hoverinfo='skip',
                showlegend=False
            ))

            # PICKUP LOCATIONS - FILL (RED) - inner fill
            fig.add_trace(go.Scattermapbox(
                lat=pickup_lat,
                lon=pickup_lon,
                mode='markers',
                marker=dict(
                    size=pickup_sizes,
                    color='rgba(255, 150, 80, 0.25)',
                    opacity=0.25
                ),
                text=[f'{c} pickups' for c in pickup_counts],
                hovertemplate='<b>Pickup Location</b><br>%{text}<br><extra></extra>',
                name='🟠 Origins',
                showlegend=True
            ))

        # DESTINATION LOCATIONS - OUTLINE (BLUE) - larger, opaque
        if dest_lat:
            fig.add_trace(go.Scattermapbox(
                lat=dest_lat,
                lon=dest_lon,
                mode='markers',
                marker=dict(
                    size=[s * 1.2 for s in dest_sizes],
                    color='rgb(20, 100, 220)',
                    opacity=0.8
                ),
                hoverinfo='skip',
                showlegend=False
            ))

            # DESTINATION LOCATIONS - FILL (BLUE) - inner fill
            fig.add_trace(go.Scattermapbox(
                lat=dest_lat,
                lon=dest_lon,
                mode='markers',
                marker=dict(
                    size=dest_sizes,
                    color='rgba(80, 150, 255, 0.25)',
                    opacity=0.25
                ),
                text=[f'{c} dropoffs' for c in dest_counts],
                hovertemplate='<b>Destination</b><br>%{text}<br><extra></extra>',
                name='🔵 Destinations',
                showlegend=True
            ))

        fig.update_layout(
            mapbox=dict(
                style='carto-positron',
                center=dict(lat=center_lat, lon=center_lon),
                zoom=10.5
            ),
            title=dict(
                text=f'Taxi Activity at {selected_hour}:00 | 🟠 Origins  🔵 Destinations | {len(hour_df)} trips',
                x=0.5,
                xanchor='center',
                font=dict(size=16)
            ),
            height=700,
            margin=dict(l=0, r=0, t=50, b=0),
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor="rgba(255, 255, 255, 0.8)"
            )
        )

        st.plotly_chart(fig, use_container_width=True)

        # Display metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Trips", f"{len(hour_df):,}")
        with col2:
            st.metric("Unique Pickup Locations", f"{len(pickup_lat):,}")
        with col3:
            st.metric("Unique Destinations", f"{len(dest_lat):,}")

        # Add contextual explanation
        st.markdown("### 🔍 What the Map Shows")
        st.markdown("""
        **At 7pm** you can see that virtually all the trips originate from the **MOM Service Center**. The destinations
        are heavily clustered in Bedok, Simei and Tampines MRT. Given neither of these three destinations show up in
        any other hour, it's safe to assume that these are probably officers returning home, lending weight to our
        **presenteeism hypothesis**.

        **During lunch hours (10-2pm)** there's quite a bit of traffic going to and leaving from **Bendemeer Food Centre**.
        Assessing the adequacy of officer's access to food options near their place of work may be prudent. It's common
        for many government organs to have a lunch shuttle to drive down cost of travel and recoup employee time.

        The geo scatter plot gives us a feel for hotspots but that's only part of the story. What we also want is to
        look for **common location pairs**. If there is a high concentration of specific trips within the same hour,
        that's a major opportunity for savings via trip pooling or maybe even a shuttle. The best place to look is
        trips originating from MOM Service Center.

        **Use the slider below to explore different hours, then see the detailed breakdowns in the tabs further down.**
        """)

    else:
        st.info(f"No trips at {selected_hour}:00")

    # Create tabs for detailed analysis
    st.markdown("---")
    st.markdown("## 📍 Detailed Geographic Analysis")

    tab1, tab2 = st.tabs(["🍽️ Lunch Hour Patterns", "🚗 Trip Pair Analysis"])

    with tab1:
        st.markdown("### Bendemeer Food Centre - Lunch Hour Concentration")

        # Filter lunch hour trips (10am-2pm)
        lunch_hour_df = df_valid[
            (df_valid['derived_trip_start_hour'] >= 10) &
            (df_valid['derived_trip_start_hour'] <= 14)
        ].copy()

        # Classify trips as Bendemeer or Other
        lunch_hour_df['is_bendemeer'] = (
            (lunch_hour_df['pickup_postal'] == '330025') |
            (lunch_hour_df['destination_postal'] == '330025')
        )

        # Count by hour and category
        hourly_breakdown = lunch_hour_df.groupby(['derived_trip_start_hour', 'is_bendemeer']).size().reset_index(name='trip_count')

        # Cost by hour and category
        hourly_cost_breakdown = lunch_hour_df.groupby(['derived_trip_start_hour', 'is_bendemeer'])['taxi_fare'].sum().reset_index(name='total_cost')

        # Pivot to get Bendemeer and Other columns
        pivot_breakdown = hourly_breakdown.pivot(index='derived_trip_start_hour', columns='is_bendemeer', values='trip_count').fillna(0)

        # Pivot for cost data
        pivot_cost_breakdown = hourly_cost_breakdown.pivot(index='derived_trip_start_hour', columns='is_bendemeer', values='total_cost').fillna(0)

        # Rename columns for clarity (trip count)
        if True in pivot_breakdown.columns and False in pivot_breakdown.columns:
            pivot_breakdown = pivot_breakdown.rename(columns={True: 'Bendemeer', False: 'Other'})
        elif True in pivot_breakdown.columns:
            pivot_breakdown = pivot_breakdown.rename(columns={True: 'Bendemeer'})
            pivot_breakdown['Other'] = 0
        elif False in pivot_breakdown.columns:
            pivot_breakdown = pivot_breakdown.rename(columns={False: 'Other'})
            pivot_breakdown['Bendemeer'] = 0
        else:
            pivot_breakdown['Bendemeer'] = 0
            pivot_breakdown['Other'] = 0

        # Rename columns for clarity (cost)
        if True in pivot_cost_breakdown.columns and False in pivot_cost_breakdown.columns:
            pivot_cost_breakdown = pivot_cost_breakdown.rename(columns={True: 'Bendemeer', False: 'Other'})
        elif True in pivot_cost_breakdown.columns:
            pivot_cost_breakdown = pivot_cost_breakdown.rename(columns={True: 'Bendemeer'})
            pivot_cost_breakdown['Other'] = 0
        elif False in pivot_cost_breakdown.columns:
            pivot_cost_breakdown = pivot_cost_breakdown.rename(columns={False: 'Other'})
            pivot_cost_breakdown['Bendemeer'] = 0
        else:
            pivot_cost_breakdown['Bendemeer'] = 0
            pivot_cost_breakdown['Other'] = 0

        # Create side-by-side charts
        col_chart1, col_chart2 = st.columns(2)

        with col_chart1:
            fig_lunch = go.Figure()

            # Add Bendemeer trips (orange)
            if 'Bendemeer' in pivot_breakdown.columns:
                fig_lunch.add_trace(go.Bar(
                    x=pivot_breakdown.index,
                    y=pivot_breakdown['Bendemeer'],
                    name='Bendemeer Food Centre',
                    marker_color="orange"
                ))

            # Add Other trips (grey)
            if 'Other' in pivot_breakdown.columns:
                fig_lunch.add_trace(go.Bar(
                    x=pivot_breakdown.index,
                    y=pivot_breakdown['Other'],
                    name='Other Destinations',
                    marker_color='#CCCCCC'
                ))

            fig_lunch.update_layout(
                barmode='stack',
                xaxis=dict(
                    tickmode='array',
                    tickvals=[10, 11, 12, 13, 14],
                    ticktext=['10:00', '11:00', '12:00', '13:00', '14:00']
                ),
                xaxis_title='Hour of Day',
                yaxis_title='Number of Trips',
                title='Trip Counts - Bendemeer Dominates',
                height=400,
                hovermode='x unified',
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )

            st.plotly_chart(fig_lunch, use_container_width=True)

        with col_chart2:
            fig_cost = go.Figure()

            # Add Bendemeer costs (orange)
            if 'Bendemeer' in pivot_cost_breakdown.columns:
                fig_cost.add_trace(go.Bar(
                    x=pivot_cost_breakdown.index,
                    y=pivot_cost_breakdown['Bendemeer'],
                    name='Bendemeer Food Centre',
                    marker_color="orange"
                ))

            # Add Other costs (grey)
            if 'Other' in pivot_cost_breakdown.columns:
                fig_cost.add_trace(go.Bar(
                    x=pivot_cost_breakdown.index,
                    y=pivot_cost_breakdown['Other'],
                    name='Other Destinations',
                    marker_color='#CCCCCC'
                ))

            fig_cost.update_layout(
                barmode='stack',
                xaxis=dict(
                    tickmode='array',
                    tickvals=[10, 11, 12, 13, 14],
                    ticktext=['10:00', '11:00', '12:00', '13:00', '14:00']
                ),
                xaxis_title='Hour of Day',
                yaxis_title='Total Cost ($)',
                title='Cost - Bendemeer Lunch Expenses',
                height=400,
                hovermode='x unified',
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )

            st.plotly_chart(fig_cost, use_container_width=True)

        # Calculate metrics for display
        bendemeer_trips = df_valid[
            (df_valid["pickup_postal"] == '330025') |
            (df_valid["destination_postal"] == '330025')
        ].copy()

        lunch_trips = bendemeer_trips[
            (bendemeer_trips['derived_trip_start_hour'] >= 10) &
            (bendemeer_trips['derived_trip_start_hour'] <= 14)
        ]

        total_trips = len(df_valid)
        bendemeer_lunch_count = len(lunch_trips)
        proportion = (bendemeer_lunch_count / total_trips) * 100
        total_cost = lunch_trips['taxi_fare'].sum()
        cost_per_ride = lunch_trips['taxi_fare'].mean()
        avg_distance = lunch_trips['distance_km'].mean()

        # Calculate monthly average cost
        num_months = df_valid['derived_month'].nunique()  # Number of months in dataset
        monthly_avg_cost = total_cost / num_months if num_months > 0 else 0

        # First row of metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Lunch Trips to/from Bendemeer", f"{bendemeer_lunch_count:,}")
        with col2:
            st.metric("Total Cost of Bendemeer Trips", f"${total_cost:,.2f}")
        with col3:
            st.metric("Avg Cost per Ride", f"${cost_per_ride:.2f}")

        # Second row of metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Percentage of All Trips", f"{proportion:.1f}%")
        with col2:
            st.metric("Avg Distance", f"{avg_distance:.2f} km")
        with col3:
            st.metric("Monthly Avg Cost", f"${monthly_avg_cost:.2f}")

        st.warning("""
        **Key Finding:** 17% of all trips involve Bendemeer Food Centre (postal 330025) during lunch hours (10am-2pm).
        This suggests limited food options near the workplace. Consider implementing a **lunch shuttle service**
        to reduce these costs.
        """)

    with tab2:
        st.markdown("### Top Routes from MOM Service Center (339946)")

        st.write("These are the trips originating from MOM Service Center (postal 339946) that were in the 10 most common trips across the whole dataset.")

        st.info("""
                Of the 10 most common trips in the dataset, **7** of them start from the MOM Service Center (339946).
                """)

        # Location name mapping
        location_names = {
            '059764': 'MOM HQ',
            '649845': 'Lakeside MRT',
            '609601': 'Jurong East MRT',
            '129580': 'Clementi MRT',
            '529888': 'Simei MRT',
            '529510': 'Tampines MRT',
            '467360': 'Bedok MRT'
        }

        # Create trip pair column
        df_pairs = df_valid.copy()
        df_pairs['trip_pair'] = df_pairs['pickup_postal'] + ' → ' + df_pairs['destination_postal']

        # Get top trip pairs
        trip_pair_counts = df_pairs['trip_pair'].value_counts()
        top_10_pairs = trip_pair_counts.head(10)

        # Filter for routes from MOM Service Center (339946)
        routes_from_339946 = [pair for pair in top_10_pairs.index if pair.startswith('339946')]

        # Create route analysis table
        route_data = []
        for route in routes_from_339946:
            route_trips = df_pairs[df_pairs['trip_pair'] == route]
            dest_postal = route.replace('339946 → ', '')
            location_name = location_names.get(dest_postal, '')

            route_data.append({
                'Destination Postal': dest_postal,
                'Location': location_name,
                'Trips': len(route_trips),
                'Total Cost': route_trips['taxi_fare'].sum(),
                'Avg Fare': route_trips['taxi_fare'].mean(),
                'Peak Hour': route_trips['derived_trip_start_hour'].mode()[0] if len(route_trips) > 0 else 0
            })

        route_df = pd.DataFrame(route_data).sort_values('Total Cost', ascending=False)

        st.dataframe(
            route_df.style.format({
                'Trips': '{:,}',
                'Total Cost': '${:,.2f}',
                'Avg Fare': '${:.2f}'
            }),
            use_container_width=True
        )

        # Prepare data for both charts
        # Categorize trips: routes from 339946 individually, or "Others"
        def categorize_trip(trip_pair):
            # Handle NA/NaN values
            if pd.isna(trip_pair):
                return 'Others (non-339946)'
            if trip_pair in routes_from_339946:
                return trip_pair
            else:
                return 'Others (non-339946)'

        df_pairs['trip_category'] = df_pairs['trip_pair'].apply(categorize_trip)

        # Function to format labels with location names
        def format_label(trip_pair):
            if trip_pair == 'Others (non-339946)':
                return 'Others'
            # Extract destination postal code
            dest_postal = trip_pair.replace('339946 → ', '')
            location_name = location_names.get(dest_postal, '')
            if location_name:
                return f"{dest_postal} | {location_name}"
            else:
                return dest_postal

        # Colorblind-friendly palette
        colorblind_palette = [
            '#0173B2',  # Blue
            '#DE8F05',  # Orange
            '#029E73',  # Green/Teal
            '#CC78BC',  # Purple/Pink
            '#CA9161',  # Brown/Tan
            '#56B4E9',  # Sky Blue
            '#E69F00',  # Amber
        ]

        # Create side-by-side charts
        st.markdown("### 📊 Trips and Costs from MOM Service Center by Hour")

        col_chart1, col_chart2 = st.columns(2)

        # LEFT CHART: Trip Counts
        with col_chart1:
            # Group by hour and trip category - COUNT trips
            hourly_trips = df_pairs.groupby(['derived_trip_start_hour', 'trip_category']).size().unstack(fill_value=0)

            # Reorder columns
            if 'Others (non-339946)' in hourly_trips.columns:
                column_order = routes_from_339946 + ['Others (non-339946)']
            else:
                column_order = routes_from_339946
            column_order = [col for col in column_order if col in hourly_trips.columns]
            hourly_trips = hourly_trips[column_order]

            # Assign colors
            colors_trips = []
            for i, col in enumerate(column_order):
                if col == 'Others (non-339946)':
                    colors_trips.append('#CCCCCC')
                else:
                    colors_trips.append(colorblind_palette[i % len(colorblind_palette)])

            fig_trips = go.Figure()
            for i, col in enumerate(hourly_trips.columns):
                fig_trips.add_trace(go.Bar(
                    x=hourly_trips.index,
                    y=hourly_trips[col],
                    name=format_label(col),
                    marker_color=colors_trips[i],
                    showlegend=True
                ))

            fig_trips.update_layout(
                barmode='stack',
                xaxis_title='Hour of Day',
                yaxis_title='Number of Trips',
                title='Trip Counts by Hour',
                height=500,
                hovermode='x unified',
                legend=dict(
                    orientation="v",
                    yanchor="top",
                    y=1,
                    xanchor="left",
                    x=1.02,
                    font=dict(size=9)
                )
            )

            st.plotly_chart(fig_trips, use_container_width=True)

        # RIGHT CHART: Costs
        with col_chart2:
            # Group by hour and trip category - SUM costs
            hourly_costs = df_pairs.groupby(['derived_trip_start_hour', 'trip_category'])['taxi_fare'].sum().unstack(fill_value=0)

            # Reorder columns (same as trips)
            hourly_costs = hourly_costs[column_order]

            # Assign colors (same as trips)
            colors_costs = colors_trips

            fig_costs = go.Figure()
            for i, col in enumerate(hourly_costs.columns):
                fig_costs.add_trace(go.Bar(
                    x=hourly_costs.index,
                    y=hourly_costs[col],
                    name=format_label(col),
                    marker_color=colors_costs[i],
                    showlegend=True
                ))

            fig_costs.update_layout(
                barmode='stack',
                xaxis_title='Hour of Day',
                yaxis_title='Total Cost ($)',
                title='Cost by Hour',
                height=500,
                hovermode='x unified',
                legend=dict(
                    orientation="v",
                    yanchor="top",
                    y=1,
                    xanchor="left",
                    x=1.02,
                    font=dict(size=9)
                )
            )

            st.plotly_chart(fig_costs, use_container_width=True)

        # Display summary metrics
        df_339946 = df_pairs[df_pairs['trip_pair'].isin(routes_from_339946)]

        # Calculate metrics
        total_routes = len(routes_from_339946)
        total_trips_339946 = len(df_339946)
        total_cost_339946 = df_339946['taxi_fare'].sum()
        num_months = df_valid['derived_month'].nunique()
        total_all_trips = len(df_valid)

        cost_per_month = total_cost_339946 / num_months if num_months > 0 else 0
        trips_per_month = total_trips_339946 / num_months if num_months > 0 else 0
        pct_of_all_trips = (total_trips_339946 / total_all_trips) * 100 if total_all_trips > 0 else 0

        # First row of metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Routes from MOM SC", f"{total_routes}")
        with col2:
            st.metric("Total Trips on These Routes", f"{total_trips_339946:,}")
        with col3:
            st.metric("Total Cost", f"${total_cost_339946:,.2f}")

        # Second row of metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Monthly Avg Cost", f"${cost_per_month:,.2f}")
        with col2:
            st.metric("Trips per Month", f"{trips_per_month:.1f}")
        with col3:
            st.metric("% of All Trips", f"{pct_of_all_trips:.1f}%")

        st.error("""
        **Key Finding:** High concentration of trips from MOM Service Center (postal 339946) to a few destinations at specific times.
        Top route is to MOM HQ (059764) with peak at 1PM. Multiple evening routes to MRT stations (7-8PM). There is strong
        potential for **ride pooling** initiatives.
        """)

# ============================================================================
# PAGE 5: COST OPPORTUNITIES
# ============================================================================
elif page == "Cost Reduction Opportunities":
    st.markdown("# 💰 Cost Reduction Opportunities")

    st.markdown("""
    This page allows you to model different cost-saving scenarios by adjusting the reduction percentages
    for each initiative. The waterfall chart and savings breakdown will update automatically.
    """)

    # Calculate base costs (needed for waterfall)
    # Filter valid postal codes
    df_valid = df[df['pickup_postal'].notna() & df['destination_postal'].notna()].copy()
    df_valid['trip_pair'] = df_valid['pickup_postal'] + ' → ' + df_valid['destination_postal']

    # Get top trip pairs
    trip_pair_counts = df_valid['trip_pair'].value_counts()
    top_10_pairs = trip_pair_counts.head(10)

    # Routes from MOM Service Center (339946)
    routes_from_339946 = [pair for pair in top_10_pairs.index if pair.startswith('339946')]

    # Calculate hourly costs
    df_339946 = df_valid[df_valid['trip_pair'].isin(routes_from_339946)]
    hourly_route_costs = df_339946.groupby(['derived_trip_start_hour', 'trip_pair'])['taxi_fare'].sum().reset_index()

    # Get base costs by hour
    base_cost_by_hour = (
        hourly_route_costs[hourly_route_costs["derived_trip_start_hour"].isin([13, 19, 20])]
        .groupby("derived_trip_start_hour")["taxi_fare"]
        .sum()
    )

    # Get presenteeism base cost
    spend_at_19 = df[df['derived_trip_start_hour'] == 19]['taxi_fare'].sum()
    total_spend = df['taxi_fare'].sum()

    st.markdown("---")

    # Initialize session state with defaults if not exists
    if 'reduction_1pm' not in st.session_state:
        st.session_state.reduction_1pm = 50
    if 'reduction_7pm' not in st.session_state:
        st.session_state.reduction_7pm = 33
    if 'reduction_8pm' not in st.session_state:
        st.session_state.reduction_8pm = 25
    if 'reduction_presenteeism' not in st.session_state:
        st.session_state.reduction_presenteeism = 30

    # Calculate savings based on current session state values
    reduction_by_hour = {
        13: st.session_state.reduction_1pm / 100,
        19: st.session_state.reduction_7pm / 100,
        20: st.session_state.reduction_8pm / 100
    }

    descriptions = {
        13: "Pooling rides from MOM Svc Ctr at 1 PM",
        19: "Pooling rides from MOM Svc Ctr at 7 PM",
        20: "Pooling rides from MOM Svc Ctr at 8 PM"
    }

    savings_data = []
    for hour in [13, 19, 20]:
        if hour in base_cost_by_hour.index:
            current_cost = base_cost_by_hour.loc[hour]
            reduction_pct = reduction_by_hour[hour]
            estimated_savings = current_cost * reduction_pct

            savings_data.append({
                'Initiative': descriptions[hour],
                'Original Cost': current_cost,
                'Reduction %': f"{reduction_pct*100:.0f}%",
                'Estimated Savings': estimated_savings
            })

    # Add presenteeism savings
    reduction_7_15pm = st.session_state.reduction_presenteeism / 100
    savings_7_15pm = spend_at_19 * reduction_7_15pm

    savings_data.append({
        'Initiative': 'Cutting back presenteeism 7-7:15pm',
        'Original Cost': spend_at_19,
        'Reduction %': f"{reduction_7_15pm*100:.0f}%",
        'Estimated Savings': savings_7_15pm
    })

    savings_df = pd.DataFrame(savings_data)

    total_savings = savings_df['Estimated Savings'].sum()
    savings_percentage = (total_savings / total_spend) * 100

    # Waterfall chart at top
    st.markdown("## 💧 Savings Waterfall")

    current_total = total_spend
    final_total = current_total - total_savings

    categories = ['Current Total'] + savings_df['Initiative'].tolist() + ['Final Total']
    values = [current_total] + [-s for s in savings_df['Estimated Savings'].tolist()] + [final_total]

    # Calculate measure for waterfall
    measure = ['absolute'] + ['relative'] * len(savings_df) + ['total']

    fig = go.Figure(go.Waterfall(
        orientation="v",
        measure=measure,
        x=categories,
        y=values,
        text=[f'${v:,.0f}' for v in values],
        textposition="outside",
        connector={"line": {"color": "rgb(63, 63, 63)"}},
        increasing={"marker": {"color": "green"}},
        decreasing={"marker": {"color": "green"}},
        totals={"marker": {"color": "orange"}}
    ))

    fig.update_layout(
        title=f"{savings_percentage:.1f}% Cost Reduction Through Ride Pooling and Reducing Presenteeism",
        yaxis_title="Amount ($)",
        height=500,
        showlegend=False
    )

    st.plotly_chart(fig, use_container_width=True)

    # Display summary metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Current Total Spend", f"${total_spend:,.2f}")
    with col2:
        st.metric("Total Estimated Savings", f"${total_savings:,.2f}")
    with col3:
        st.metric("Percentage Reduction", f"{savings_percentage:.1f}%")

    st.markdown("---")

    # Interactive parameter controls in 2x2 grid
    st.markdown("## 🎛️ Adjust Savings Scenarios")
    st.caption("Use the sliders below to model different cost-saving scenarios. The chart and table update automatically.")

    # Row 1: 1PM and 7PM pooling
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("#### 1. Ride Pooling at 1PM: MOM Service Center → MOM HQ")
        st.metric("Original Cost", f"${base_cost_by_hour.get(13, 0):.2f}")

        reduction_1pm = st.slider(
            "Cost Reduction %",
            min_value=0,
            max_value=100,
            value=50,
            step=5,
            key="reduction_1pm",
            help="Percentage reduction in taxi costs through ride pooling at 1PM"
        )

        # Explanation for 1PM pooling
        if reduction_1pm == 50:
            explanation = "**2 officers per taxi** on average"
        elif reduction_1pm == 33:
            explanation = "**1.5 officers per taxi** on average"
        elif reduction_1pm == 67:
            explanation = "**3 officers per taxi** on average"
        elif reduction_1pm == 25:
            explanation = "**1.33 officers per taxi** on average"
        elif reduction_1pm == 0:
            explanation = "**No pooling** - each officer takes separate taxi"
        else:
            if reduction_1pm > 0:
                avg_officers = 1 / (1 - reduction_1pm / 100)
                explanation = f"Approximately **{avg_officers:.2f} officers per taxi**"
            else:
                explanation = "**No pooling**"

        st.info(f"💡 {reduction_1pm}% reduction = {explanation}")

    with col_right:
        st.markdown("#### 2. Ride Pooling at 7PM: MOM Service Center → MRT Stations")
        st.metric("Original Cost", f"${base_cost_by_hour.get(19, 0):.2f}")

        reduction_7pm = st.slider(
            "Cost Reduction %",
            min_value=0,
            max_value=100,
            value=33,
            step=5,
            key="reduction_7pm",
            help="Percentage reduction in taxi costs through ride pooling at 7PM"
        )

        # Explanation for 7PM pooling
        if reduction_7pm == 50:
            explanation = "**2 officers per taxi** on average"
        elif reduction_7pm == 33:
            explanation = "**1.5 officers per taxi** on average"
        elif reduction_7pm == 67:
            explanation = "**3 officers per taxi** on average"
        elif reduction_7pm == 25:
            explanation = "**1.33 officers per taxi** on average"
        elif reduction_7pm == 0:
            explanation = "**No pooling** - each officer takes separate taxi"
        else:
            if reduction_7pm > 0:
                avg_officers = 1 / (1 - reduction_7pm / 100)
                explanation = f"Approximately **{avg_officers:.2f} officers per taxi**"
            else:
                explanation = "**No pooling**"

        st.info(f"💡 {reduction_7pm}% reduction = {explanation}")

    # Row 2: 8PM pooling and presenteeism
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("#### 3. Ride Pooling at 8PM: MOM Service Center → MRT Stations")
        st.metric("Original Cost", f"${base_cost_by_hour.get(20, 0):.2f}")

        reduction_8pm = st.slider(
            "Cost Reduction %",
            min_value=0,
            max_value=100,
            value=25,
            step=5,
            key="reduction_8pm",
            help="Percentage reduction in taxi costs through ride pooling at 8PM"
        )

        # Explanation for 8PM pooling
        if reduction_8pm == 50:
            explanation = "**2 officers per taxi** on average"
        elif reduction_8pm == 33:
            explanation = "**1.5 officers per taxi** on average"
        elif reduction_8pm == 67:
            explanation = "**3 officers per taxi** on average"
        elif reduction_8pm == 25:
            explanation = "**1.33 officers per taxi** on average"
        elif reduction_8pm == 0:
            explanation = "**No pooling** - each officer takes separate taxi"
        else:
            if reduction_8pm > 0:
                avg_officers = 1 / (1 - reduction_8pm / 100)
                explanation = f"Approximately **{avg_officers:.2f} officers per taxi**"
            else:
                explanation = "**No pooling**"

        st.info(f"💡 {reduction_8pm}% reduction = {explanation}")

    with col_right:
        st.markdown("#### 4. Reducing Presenteeism: 7:00-7:15PM trip window")
        st.metric("Original Cost", f"${spend_at_19:.2f}")

        reduction_presenteeism = st.slider(
            "Trip Reduction %",
            min_value=0,
            max_value=100,
            value=30,
            step=5,
            key="reduction_presenteeism",
            help="Percentage reduction in trips during the 7-7:15PM window"
        )

        # Explanation for presenteeism
        if reduction_presenteeism == 0:
            explanation = "**💡No change** to current behavior"
        elif reduction_presenteeism < 20:
            explanation = "**💡Mild change** in behavior"
        elif reduction_presenteeism < 40:
            explanation = "**💡Moderate change** in behavior"
        elif reduction_presenteeism < 60:
            explanation = "**💡Significant change** in behavior"
        else:
            explanation = "**💡Major change** in behavior"

        st.info(f"{explanation}")

    st.warning("""
    **Note on Presenteeism:** This assumes some officers stay late primarily to qualify for taxi reimbursement.
    Requires cultural/policy changes like manager encouragement or tiered reimbursement rates.
    """)

    st.markdown("---")

    # Display savings breakdown table
    st.markdown("## 📊 Savings Breakdown")

    st.dataframe(
        savings_df.style.format({
            'Original Cost': '${:,.2f}',
            'Estimated Savings': '${:,.2f}'
        }),
        use_container_width=True
    )

    # Final recommendations
    st.markdown("---")
    st.markdown("""
    <h2>🎯 Summary of Recommendations</h2>

    <h3>1. Ride Pooling Program</h3>
    <ul>
    <li><strong>Target:</strong> Common routes from MOM Service Center, especially at 1PM (to HQ) and 7-8PM (to MRT stations)</li>
    <li><strong>Expected Savings:</strong> $1,183</li>
    <li><strong>Implementation:</strong> 
        <ul>
        <li>Introduce guidance on pooling of trips wherever sensible: such as trips from the same department.</li>
        <li>To match unaffiliated officers for pooling, consider a designated meeting zone at MOM Service Centre driveway for all rides heading to common destinations.</li>
        <li>For trips to HQ at 1pm, consider a fixed light shuttle service.</li>
        </ul>
    </li>
    </ul>

    <h3>2. Address Presenteeism</h3>
    <ul>
    <li><strong>Target:</strong> Reduce 7-7:15PM trips by 30%</li>
    <li><strong>Expected Savings:</strong> $418</li>
    <li><strong>Implementation:</strong>
        <ul>
        <li>Manager communication encouraging timely departure</li>
        <li>Explore tiered reimbursement rates (e.g., 50% at 6:30PM, 75% at 6:45PM, 100% at 7PM)</li>
        </ul>
    </li>
    </ul>

    <h3>3. Lunch Shuttle Service (Not modelled here/To be explored)</h3>
    <ul>
    <li><strong>Target:</strong> A reduction of 10-20% should be possible here, but more consultation with staff should be done to accurately treat the issue. 
                The signal is less clear than for trips originating from MOM Service Centre; <br>there could be multiple factors leading to concentrated trips at Bendemeer beyond lunch crush. </li>
    <li><strong>Expected Impact:</strong> Reduction in lunch-hour trips</li>
    <li><strong>Implementation:</strong> A possible shuttle service during lunch hours (~10AM-2PM)</li>
    </ul>

    <h3>Total Impact: ~10% reduction in taxi spending</h3>

    <p><strong>Note:</strong> The savings multipliers selected here erred on the conservative side. This was keeping in mind that there is fine balance between economising and destroying value. 
                <br> For instance, if we mandated that all rides from MOM Svc Centre to HQ/MRTs had to have at least two officers in them we would realise a much higher savings. 
                <br> But this would unduly hobble senior officers or those with urgent taskings from appropriately taking a solo cab.
                </p>
    """, unsafe_allow_html=True)

# ============================================================================
# PAGE 6: THOUGHT PROCESS & DISCLAIMER
# ============================================================================
elif page == "Thought Process & Disclaimer":
    st.markdown("### Thought Process")

    st.markdown("""
    You might have notice that in this analysis I did not focus on division/departments or individual officers (via the card numbers).
    This is because there isn't enough context here to understand what each of these departments do, let alone individuals. Thus, even if we saw a 80/20 pattern in spend where a few
    departments or officers accounted for the majority of spend, there would be no way for me to tell if that spend was justified or not. Making recommendations in that situation would be irresponsible.

    Instead, I focused on broad based patterns where milder assumptions were needed to draw conclusions. For instance, the surge in trips taken just after 7pm is a well known behaviour whenever you have a
    policy that relies on a cutoff time. Likewise, if we see a concentration of trips emmanating from a main office to a few key destinations, it is reasonable to assume that pooling rides would be possible.
    """)

    st.markdown("### Disclaimer")

    st.markdown("""
    If this weren't a take home assignment, I would be strongly **pushing the case that taxi spend doesn't need to be curtailed at all**. If you take into account the average wage of a public service officer (and even then take a discount on that), you'll quickly realise that the value of officers time is plainly ahead of these taxi fares.

    To make that case, I would use the OneMap API to calculate the time savings for each of these trips **X** the average (or even 25th percentile) hourly wage **X** multiplier (accounting for the surplus value generated by workers for the employer)

    I'm quite confident the **value recaptured by saving travel time easily exceeds cost of fares**. This relies on a few heuristics:
    - (1) point-to-point transport in Singapore is cheap vis-a-vis wages and for our development stage
    - (2) the public service in SG is very lean by global standards and has become much leaner over the years -> pushing up expected value each worker creates with their time
    - (3) many of these trips are taken during hours where there is not alternative, thus they are an entitlement and factor into implicit wages (you can't cut these back no matter what; cutting back amounts to a stealth wage cut)

    In sum, cutting back on taxi spend is usually **bad value for money/false economy**.
    """)

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #666;'>⚙️ Sean Lim / 02/2026 / MOM OA</p>",
    unsafe_allow_html=True
)
