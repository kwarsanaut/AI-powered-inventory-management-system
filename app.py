import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
from datetime import datetime, timedelta
import random
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="AI Inventory Management System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
    }
    .success-metric {
        border-left: 4px solid #28a745;
    }
    .warning-metric {
        border-left: 4px solid #ffc107;
    }
    .danger-metric {
        border-left: 4px solid #dc3545;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

# Generate comprehensive mock data
@st.cache_data
def generate_inventory_data():
    np.random.seed(42)
    
    # Product categories and items
    categories = {
        'Electronics': ['iPhone 15', 'Samsung Galaxy S24', 'MacBook Pro', 'Dell Laptop', 'iPad Air', 'AirPods Pro', 'Sony Headphones', 'Gaming Console'],
        'Clothing': ['Winter Jacket', 'Summer Dress', 'Jeans', 'T-Shirt', 'Sneakers', 'Formal Shoes', 'Sweater', 'Shorts'],
        'Home & Garden': ['Coffee Maker', 'Vacuum Cleaner', 'Garden Tools', 'Kitchen Set', 'Bed Sheets', 'Curtains', 'Dining Table', 'Sofa'],
        'Food & Beverage': ['Organic Coffee', 'Premium Tea', 'Protein Bars', 'Vitamin Supplements', 'Energy Drinks', 'Chocolate', 'Nuts Mix', 'Honey'],
        'Sports & Fitness': ['Yoga Mat', 'Dumbbells', 'Running Shoes', 'Fitness Tracker', 'Protein Powder', 'Water Bottle', 'Tennis Racket', 'Bicycle'],
        'Books & Media': ['Business Books', 'Fiction Novels', 'Educational DVDs', 'Board Games', 'Magazines', 'Audio Books', 'Art Supplies', 'Notebooks']
    }
    
    products = []
    for category, items in categories.items():
        for item in items:
            products.append({
                'Product_ID': f"{category[:3].upper()}{len(products)+1:03d}",
                'Product_Name': item,
                'Category': category
            })
    
    # Generate historical sales data (12 months)
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
    sales_data = []
    
    for product in products:
        base_demand = np.random.uniform(10, 100)
        seasonality = np.random.choice(['high_winter', 'high_summer', 'stable', 'holiday_spike'])
        
        for date in dates:
            # Apply seasonality
            seasonal_factor = 1.0
            if seasonality == 'high_winter' and date.month in [12, 1, 2]:
                seasonal_factor = 1.5
            elif seasonality == 'high_summer' and date.month in [6, 7, 8]:
                seasonal_factor = 1.4
            elif seasonality == 'holiday_spike' and date.month in [11, 12]:
                seasonal_factor = 1.8
            
            # Weekend effect
            weekend_factor = 1.2 if date.weekday() >= 5 else 1.0
            
            # Random variation
            random_factor = np.random.uniform(0.7, 1.3)
            
            demand = max(0, int(base_demand * seasonal_factor * weekend_factor * random_factor))
            
            sales_data.append({
                'Date': date,
                'Product_ID': product['Product_ID'],
                'Product_Name': product['Product_Name'],
                'Category': product['Category'],
                'Demand': demand,
                'Sales': demand * np.random.uniform(0.8, 1.0)  # Some lost sales due to stockouts
            })
    
    sales_df = pd.DataFrame(sales_data)
    products_df = pd.DataFrame(products)
    
    # Generate current inventory levels
    current_inventory = []
    for product in products:
        avg_daily_demand = sales_df[sales_df['Product_ID'] == product['Product_ID']]['Demand'].mean()
        
        current_stock = np.random.uniform(avg_daily_demand * 5, avg_daily_demand * 30)
        reorder_point = avg_daily_demand * 7  # 7 days safety stock
        max_stock = avg_daily_demand * 45
        
        # Supplier info
        suppliers = ['Supplier A', 'Supplier B', 'Supplier C', 'Supplier D']
        supplier = np.random.choice(suppliers)
        lead_time = np.random.randint(3, 21)  # 3-21 days
        unit_cost = np.random.uniform(5, 200)
        carrying_cost_rate = 0.25  # 25% annual carrying cost
        
        current_inventory.append({
            'Product_ID': product['Product_ID'],
            'Product_Name': product['Product_Name'],
            'Category': product['Category'],
            'Current_Stock': int(current_stock),
            'Reorder_Point': int(reorder_point),
            'Max_Stock': int(max_stock),
            'Supplier': supplier,
            'Lead_Time_Days': lead_time,
            'Unit_Cost': round(unit_cost, 2),
            'Carrying_Cost_Rate': carrying_cost_rate,
            'Avg_Daily_Demand': round(avg_daily_demand, 2)
        })
    
    inventory_df = pd.DataFrame(current_inventory)
    
    return sales_df, inventory_df, products_df

# AI Demand Forecasting Function
@st.cache_data
def forecast_demand(sales_data, product_id, days_ahead=30):
    product_data = sales_data[sales_data['Product_ID'] == product_id].copy()
    product_data = product_data.sort_values('Date')
    
    # Simple moving average with trend
    window = 14
    product_data['MA'] = product_data['Demand'].rolling(window=window).mean()
    
    # Calculate trend
    recent_data = product_data.tail(30)
    if len(recent_data) > 1:
        trend = np.polyfit(range(len(recent_data)), recent_data['Demand'], 1)[0]
    else:
        trend = 0
    
    # Generate forecasts
    last_date = product_data['Date'].max()
    forecast_dates = pd.date_range(start=last_date + timedelta(days=1), periods=days_ahead, freq='D')
    
    base_forecast = product_data['MA'].iloc[-1] if not pd.isna(product_data['MA'].iloc[-1]) else product_data['Demand'].mean()
    
    forecasts = []
    for i, date in enumerate(forecast_dates):
        # Apply trend
        forecast = base_forecast + (trend * i)
        
        # Add seasonality (simplified)
        seasonal_factor = 1.0
        if date.month in [11, 12]:  # Holiday season
            seasonal_factor = 1.3
        elif date.month in [6, 7, 8]:  # Summer
            seasonal_factor = 1.1
        
        forecast *= seasonal_factor
        
        # Add some random variation for confidence intervals
        std_dev = product_data['Demand'].std() * 0.3
        
        forecasts.append({
            'Date': date,
            'Forecast': max(0, forecast),
            'Lower_CI': max(0, forecast - 1.96 * std_dev),
            'Upper_CI': forecast + 1.96 * std_dev
        })
    
    return pd.DataFrame(forecasts)

# Calculate business metrics
def calculate_metrics(inventory_df, sales_df):
    # Total inventory value
    total_inventory_value = (inventory_df['Current_Stock'] * inventory_df['Unit_Cost']).sum()
    
    # Items needing reorder
    items_to_reorder = len(inventory_df[inventory_df['Current_Stock'] <= inventory_df['Reorder_Point']])
    
    # Overstock items
    overstock_items = len(inventory_df[inventory_df['Current_Stock'] > inventory_df['Max_Stock'] * 0.8])
    
    # Average inventory turnover
    total_annual_sales = sales_df.groupby('Product_ID')['Sales'].sum()
    avg_inventory_value = inventory_df.set_index('Product_ID')['Current_Stock'] * inventory_df.set_index('Product_ID')['Unit_Cost']
    turnover_ratio = (total_annual_sales / avg_inventory_value).mean()
    
    # Carrying cost
    annual_carrying_cost = (inventory_df['Current_Stock'] * inventory_df['Unit_Cost'] * inventory_df['Carrying_Cost_Rate']).sum()
    
    return {
        'total_inventory_value': total_inventory_value,
        'items_to_reorder': items_to_reorder,
        'overstock_items': overstock_items,
        'turnover_ratio': turnover_ratio,
        'annual_carrying_cost': annual_carrying_cost
    }

# Main application
def main():
    st.markdown('<h1 class="main-header">🚀 AI-Powered Inventory Management System</h1>', unsafe_allow_html=True)
    st.markdown("### Transforming Inventory Management with Artificial Intelligence")
    
    # Load data
    sales_df, inventory_df, products_df = generate_inventory_data()
    metrics = calculate_metrics(inventory_df, sales_df)
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a section:",
        ["🎯 Executive Dashboard", "🔮 Demand Forecasting", "🚛 Smart Reordering", "💰 Cost Optimization", "📊 Business Impact"]
    )
    
    if page == "🎯 Executive Dashboard":
        executive_dashboard(inventory_df, sales_df, metrics)
    elif page == "🔮 Demand Forecasting":
        demand_forecasting(sales_df, inventory_df)
    elif page == "🚛 Smart Reordering":
        smart_reordering(inventory_df, sales_df)
    elif page == "💰 Cost Optimization":
        cost_optimization(inventory_df, sales_df, metrics)
    elif page == "📊 Business Impact":
        business_impact(metrics, inventory_df)

def executive_dashboard(inventory_df, sales_df, metrics):
    st.header("🎯 Executive Dashboard")
    st.markdown("Real-time overview of inventory performance and key business metrics")
    
    # Key Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Inventory Value",
            value=f"${metrics['total_inventory_value']:,.0f}",
            delta="-8.2% vs last month"
        )
    
    with col2:
        st.metric(
            label="Items Needing Reorder",
            value=metrics['items_to_reorder'],
            delta=f"+{metrics['items_to_reorder']} urgent"
        )
    
    with col3:
        st.metric(
            label="Inventory Turnover",
            value=f"{metrics['turnover_ratio']:.1f}x",
            delta="+0.3x vs target"
        )
    
    with col4:
        st.metric(
            label="Annual Carrying Cost",
            value=f"${metrics['annual_carrying_cost']:,.0f}",
            delta="-15.3% optimized"
        )
    
    # Charts Row 1
    col1, col2 = st.columns(2)
    
    with col1:
        # Inventory levels by category
        category_stock = inventory_df.groupby('Category').agg({
            'Current_Stock': 'sum',
            'Unit_Cost': 'mean'
        }).reset_index()
        category_stock['Total_Value'] = category_stock['Current_Stock'] * category_stock['Unit_Cost']
        
        fig = px.bar(
            category_stock, 
            x='Category', 
            y='Total_Value',
            title='Inventory Value by Category',
            color='Total_Value',
            color_continuous_scale='Blues'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Stock status distribution
        inventory_df['Stock_Status'] = inventory_df.apply(
            lambda x: 'Critical' if x['Current_Stock'] <= x['Reorder_Point'] 
            else 'Overstock' if x['Current_Stock'] > x['Max_Stock'] * 0.8
            else 'Optimal', axis=1
        )
        
        status_counts = inventory_df['Stock_Status'].value_counts()
        fig = px.pie(
            values=status_counts.values,
            names=status_counts.index,
            title='Stock Status Distribution',
            color_discrete_map={
                'Optimal': '#28a745',
                'Critical': '#dc3545',
                'Overstock': '#ffc107'
            }
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Charts Row 2
    col1, col2 = st.columns(2)
    
    with col1:
        # Recent sales trend
        recent_sales = sales_df[sales_df['Date'] >= '2024-11-01'].groupby('Date')['Sales'].sum().reset_index()
        fig = px.line(
            recent_sales,
            x='Date',
            y='Sales',
            title='Daily Sales Trend (Last 2 Months)',
            markers=True
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Top performing categories
        category_performance = sales_df.groupby('Category')['Sales'].sum().reset_index()
        category_performance = category_performance.sort_values('Sales', ascending=True)
        
        fig = px.bar(
            category_performance,
            x='Sales',
            y='Category',
            orientation='h',
            title='Sales Performance by Category (YTD)',
            color='Sales',
            color_continuous_scale='Greens'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Critical Alerts
    st.subheader("🚨 Critical Alerts")
    
    critical_items = inventory_df[inventory_df['Current_Stock'] <= inventory_df['Reorder_Point']]
    if len(critical_items) > 0:
        st.error(f"⚠️ {len(critical_items)} items below reorder point!")
        
        alert_df = critical_items[['Product_Name', 'Current_Stock', 'Reorder_Point', 'Lead_Time_Days']].copy()
        alert_df['Days_Until_Stockout'] = alert_df['Current_Stock'] / inventory_df.set_index(critical_items.index)['Avg_Daily_Demand']
        st.dataframe(alert_df, use_container_width=True)
    else:
        st.success("✅ All items are adequately stocked!")

def demand_forecasting(sales_df, inventory_df):
    st.header("🔮 AI Demand Forecasting")
    st.markdown("Advanced AI predictions for optimal inventory planning")
    
    # Product selection
    col1, col2 = st.columns([2, 1])
    
    with col1:
        selected_product = st.selectbox(
            "Select Product for Detailed Forecast:",
            inventory_df['Product_Name'].tolist()
        )
    
    with col2:
        forecast_period = st.selectbox(
            "Forecast Period:",
            [30, 60, 90],
            format_func=lambda x: f"{x} days"
        )
    
    product_id = inventory_df[inventory_df['Product_Name'] == selected_product]['Product_ID'].iloc[0]
    
    # Generate forecast
    forecast_df = forecast_demand(sales_df, product_id, forecast_period)
    
    # Historical vs Forecast Chart
    historical_data = sales_df[sales_df['Product_ID'] == product_id].copy()
    historical_data = historical_data.sort_values('Date').tail(90)  # Last 90 days
    
    fig = go.Figure()
    
    # Historical data
    fig.add_trace(go.Scatter(
        x=historical_data['Date'],
        y=historical_data['Demand'],
        mode='lines+markers',
        name='Historical Demand',
        line=dict(color='blue')
    ))
    
    # Forecast
    fig.add_trace(go.Scatter(
        x=forecast_df['Date'],
        y=forecast_df['Forecast'],
        mode='lines+markers',
        name='AI Forecast',
        line=dict(color='red', dash='dash')
    ))
    
    # Confidence intervals
    fig.add_trace(go.Scatter(
        x=forecast_df['Date'],
        y=forecast_df['Upper_CI'],
        fill=None,
        mode='lines',
        line_color='rgba(0,100,80,0)',
        showlegend=False
    ))
    
    fig.add_trace(go.Scatter(
        x=forecast_df['Date'],
        y=forecast_df['Lower_CI'],
        fill='tonexty',
        mode='lines',
        line_color='rgba(0,100,80,0)',
        name='95% Confidence Interval',
        fillcolor='rgba(255,0,0,0.2)'
    ))
    
    fig.update_layout(
        title=f'Demand Forecast: {selected_product}',
        xaxis_title='Date',
        yaxis_title='Demand (Units)',
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Forecast Summary
    col1, col2, col3 = st.columns(3)
    
    with col1:
        avg_forecast = forecast_df['Forecast'].mean()
        st.metric("Average Daily Forecast", f"{avg_forecast:.1f} units")
    
    with col2:
        total_forecast = forecast_df['Forecast'].sum()
        st.metric(f"Total {forecast_period}-Day Demand", f"{total_forecast:.0f} units")
    
    with col3:
        current_stock = inventory_df[inventory_df['Product_ID'] == product_id]['Current_Stock'].iloc[0]
        coverage_days = current_stock / avg_forecast if avg_forecast > 0 else 0
        st.metric("Current Stock Coverage", f"{coverage_days:.1f} days")
    
    # Seasonal Analysis
    st.subheader("📈 Seasonal Trend Analysis")
    
    # Monthly demand pattern
    monthly_data = sales_df[sales_df['Product_ID'] == product_id].copy()
    monthly_data['Month'] = monthly_data['Date'].dt.month
    monthly_pattern = monthly_data.groupby('Month')['Demand'].mean().reset_index()
    
    fig = px.bar(
        monthly_pattern,
        x='Month',
        y='Demand',
        title='Average Monthly Demand Pattern',
        color='Demand',
        color_continuous_scale='Blues'
    )
    fig.update_xaxis(tickmode='array', tickvals=list(range(1, 13)), 
                     ticktext=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    st.plotly_chart(fig, use_container_width=True)
    
    # AI Insights
    st.subheader("🤖 AI-Generated Insights")
    
    insights = []
    
    # Trend analysis
    recent_trend = forecast_df['Forecast'].tail(7).mean() - forecast_df['Forecast'].head(7).mean()
    if recent_trend > 0:
        insights.append("📈 Demand is trending upward in the forecast period")
    elif recent_trend < -1:
        insights.append("📉 Demand is expected to decline in the forecast period")
    else:
        insights.append("➡️ Demand is expected to remain stable")
    
    # Seasonality
    if monthly_pattern['Demand'].max() / monthly_pattern['Demand'].min() > 1.3:
        peak_month = monthly_pattern.loc[monthly_pattern['Demand'].idxmax(), 'Month']
        month_names = {1: 'January', 2: 'February', 3: 'March', 4: 'April', 5: 'May', 6: 'June',
                      7: 'July', 8: 'August', 9: 'September', 10: 'October', 11: 'November', 12: 'December'}
        insights.append(f"🗓️ Strong seasonal pattern detected - peak demand in {month_names[peak_month]}")
    
    # Stock coverage
    if coverage_days < 14:
        insights.append("⚠️ Current stock may not cover forecast demand - immediate reorder recommended")
    elif coverage_days > 60:
        insights.append("💰 Opportunity to reduce inventory levels and carrying costs")
    
    for insight in insights:
        st.info(insight)

def smart_reordering(inventory_df, sales_df):
    st.header("🚛 Smart Reordering System")
    st.markdown("AI-powered automated purchase recommendations and supplier optimization")
    
    # Calculate reorder recommendations
    reorder_recommendations = []
    
    for _, product in inventory_df.iterrows():
        if product['Current_Stock'] <= product['Reorder_Point']:
            # Calculate optimal order quantity (EOQ)
            annual_demand = sales_df[sales_df['Product_ID'] == product['Product_ID']]['Demand'].sum()
            ordering_cost = 50  # Assumed ordering cost per order
            carrying_cost = product['Unit_Cost'] * product['Carrying_Cost_Rate']
            
            if carrying_cost > 0:
                eoq = np.sqrt((2 * annual_demand * ordering_cost) / carrying_cost)
            else:
                eoq = product['Max_Stock'] - product['Current_Stock']
            
            # Adjust for current stock and max stock
            order_quantity = min(eoq, product['Max_Stock'] - product['Current_Stock'])
            order_quantity = max(order_quantity, product['Reorder_Point'] - product['Current_Stock'])
            
            total_cost = order_quantity * product['Unit_Cost']
            urgency = "HIGH" if product['Current_Stock'] < product['Reorder_Point'] * 0.5 else "MEDIUM"
            
            reorder_recommendations.append({
                'Product_Name': product['Product_Name'],
                'Category': product['Category'],
                'Current_Stock': product['Current_Stock'],
                'Reorder_Point': product['Reorder_Point'],
                'Recommended_Quantity': int(order_quantity),
                'Supplier': product['Supplier'],
                'Lead_Time': product['Lead_Time_Days'],
                'Unit_Cost': product['Unit_Cost'],
                'Total_Cost': total_cost,
                'Urgency': urgency
            })
    
    # Display urgent reorders
    if reorder_recommendations:
        st.subheader("⚡ Immediate Reorder Recommendations")
        
        reorder_df = pd.DataFrame(reorder_recommendations)
        reorder_df = reorder_df.sort_values(['Urgency', 'Total_Cost'], ascending=[True, False])
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Items to Reorder", len(reorder_df))
        
        with col2:
            total_reorder_cost = reorder_df['Total_Cost'].sum()
            st.metric("Total Reorder Value", f"${total_reorder_cost:,.0f}")
        
        with col3:
            avg_lead_time = reorder_df['Lead_Time'].mean()
            st.metric("Avg Lead Time", f"{avg_lead_time:.1f} days")
        
        with col4:
            high_urgency = len(reorder_df[reorder_df['Urgency'] == 'HIGH'])
            st.metric("High Urgency Items", high_urgency)
        
        # Reorder table with styling
        def highlight_urgency(row):
            if row['Urgency'] == 'HIGH':
                return ['background-color: #ffebee'] * len(row)
            else:
                return ['background-color: #fff3e0'] * len(row)
        
        styled_df = reorder_df.style.apply(highlight_urgency, axis=1)
        st.dataframe(styled_df, use_container_width=True)
        
        # Generate purchase orders button
        if st.button("🛒 Generate Purchase Orders", type="primary"):
            st.success("✅ Purchase orders generated and sent to suppliers!")
            st.balloons()
    
    else:
        st.success("✅ All items are adequately stocked. No immediate reorders needed!")
    
    # Supplier Performance Analysis
    st.subheader("📊 Supplier Performance Dashboard")
    
    # Simulate supplier performance data
    suppliers = inventory_df['Supplier'].unique()
    supplier_performance = []
    
    for supplier in suppliers:
        supplier_products = inventory_df[inventory_df['Supplier'] == supplier]
        avg_lead_time = supplier_products['Lead_Time_Days'].mean()
        on_time_delivery = np.random.uniform(85, 98)  # Simulated performance
        quality_score = np.random.uniform(90, 99)
        total_value = (supplier_products['Current_Stock'] * supplier_products['Unit_Cost']).sum()
        
        supplier_performance.append({
            'Supplier': supplier,
            'Products': len(supplier_products),
            'Avg_Lead_Time': avg_lead_time,
            'On_Time_Delivery': on_time_delivery,
            'Quality_Score': quality_score,
            'Total_Value': total_value,
            'Performance_Score': (on_time_delivery + quality_score) / 2
        })
    
    supplier_df = pd.DataFrame(supplier_performance)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Supplier performance heatmap
        fig = px.scatter(
            supplier_df,
            x='On_Time_Delivery',
            y='Quality_Score',
            size='Total_Value',
            color='Performance_Score',
            hover_data=['Supplier', 'Products', 'Avg_Lead_Time'],
            title='Supplier Performance Matrix',
            color_continuous_scale='RdYlGn'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Lead time comparison
        fig = px.bar(
            supplier_df,
            x='Supplier',
            y='Avg_Lead_Time',
            title='Average Lead Time by Supplier',
            color='Avg_Lead_Time',
            color_continuous_scale='RdYlBu_r'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Best performing supplier
    best_supplier = supplier_df.loc[supplier_df['Performance_Score'].idxmax()]
    st.info(f"🏆 Best Performing Supplier: **{best_supplier['Supplier']}** "
            f"(Performance Score: {best_supplier['Performance_Score']:.1f})")

def cost_optimization(inventory_df, sales_df, metrics):
    st.header("💰 Cost Optimization Analysis")
    st.markdown("Advanced analytics for reducing inventory costs and maximizing ROI")
    
    # Cost breakdown
    inventory_df['Total_Inventory_Value'] = inventory_df['Current_Stock'] * inventory_df['Unit_Cost']
    inventory_df['Annual_Carrying_Cost'] = inventory_df['Total_Inventory_Value'] * inventory_df['Carrying_Cost_Rate']
    
    # Dead stock analysis
    recent_sales = sales_df[sales_df['Date'] >= '2024-10-01'].groupby('Product_ID')['Sales'].sum()
    inventory_df['Recent_Sales'] = inventory_df['Product_ID'].map(recent_sales).fillna(0)
    inventory_df['Dead_Stock'] = inventory_df['Recent_Sales'] == 0
    
    dead_stock_value = inventory_df[inventory_df['Dead_Stock']]['Total_Inventory_Value'].sum()
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Carrying Cost", f"${metrics['annual_carrying_cost']:,.0f}")
    
    with col2:
        st.metric("Dead Stock Value", f"${dead_stock_value:,.0f}", 
                 delta=f"-${dead_stock_value*0.1:,.0f} potential savings")
    
    with col3:
        overstock_value = inventory_df[inventory_df['Current_Stock'] > inventory_df['Max_Stock']]['Total_Inventory_Value'].sum()
        st.metric("Overstock Value", f"${overstock_value:,.0f}")
    
    with col4:
        storage_utilization = (inventory_df['Current_Stock'].sum() / inventory_df['Max_Stock'].sum()) * 100
        st.metric("Storage Utilization", f"{storage_utilization:.1f}%")
    
    # Cost Analysis Charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Cost breakdown by category
        cost_by_category = inventory_df.groupby('Category').agg({
            'Total_Inventory_Value': 'sum',
            'Annual_Carrying_Cost': 'sum'
        }).reset_index()
        
        fig = px.bar(
            cost_by_category,
            x='Category',
            y=['Total_Inventory_Value', 'Annual_Carrying_Cost'],
            title='Inventory Costs by Category',
            barmode='group'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Dead stock identification
        dead_stock_df = inventory_df[inventory_df['Dead_Stock']].copy()
        if len(dead_stock_df) > 0:
            fig = px.pie(
                dead_stock_df,
                values='Total_Inventory_Value',
                names='Category',
                title='Dead Stock Distribution by Category'
            )
        else:
            # Create a placeholder chart if no dead stock
            fig = px.pie(
                values=[1],
                names=['No Dead Stock'],
                title='Dead Stock Distribution by Category'
            )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Optimization Opportunities
    st.subheader("🎯 Optimization Opportunities")
    
    opportunities = []
    
    # Overstock optimization
    overstock_items = inventory_df[inventory_df['Current_Stock'] > inventory_df['Max_Stock'] * 0.9]
    if len(overstock_items) > 0:
        overstock_savings = overstock_items['Annual_Carrying_Cost'].sum() * 0.3
        opportunities.append({
            'Opportunity': 'Reduce Overstock',
            'Items_Affected': len(overstock_items),
            'Potential_Savings': overstock_savings,
            'Action': 'Implement dynamic pricing or promotions'
        })
    
    # Dead stock clearance
    if dead_stock_value > 0:
        dead_stock_savings = dead_stock_value * 0.6  # Assume 60% recovery through clearance
        opportunities.append({
            'Opportunity': 'Clear Dead Stock',
            'Items_Affected': len(inventory_df[inventory_df['Dead_Stock']]),
            'Potential_Savings': dead_stock_savings,
            'Action': 'Liquidation sale or return to suppliers'
        })
    
    # Lead time optimization
    high_lead_time_items = inventory_df[inventory_df['Lead_Time_Days'] > 14]
    if len(high_lead_time_items) > 0:
        lead_time_savings = high_lead_time_items['Annual_Carrying_Cost'].sum() * 0.15
        opportunities.append({
            'Opportunity': 'Optimize Lead Times',
            'Items_Affected': len(high_lead_time_items),
            'Potential_Savings': lead_time_savings,
            'Action': 'Negotiate with suppliers or find alternatives'
        })
    
    # ABC Analysis for focus areas
    inventory_df['Annual_Sales_Value'] = inventory_df['Product_ID'].map(
        sales_df.groupby('Product_ID')['Sales'].sum() * inventory_df.set_index('Product_ID')['Unit_Cost']
    ).fillna(0)
    
    inventory_df_sorted = inventory_df.sort_values('Annual_Sales_Value', ascending=False)
    inventory_df_sorted['Cumulative_Percentage'] = inventory_df_sorted['Annual_Sales_Value'].cumsum() / inventory_df_sorted['Annual_Sales_Value'].sum() * 100
    
    inventory_df_sorted['ABC_Category'] = 'C'
    inventory_df_sorted.loc[inventory_df_sorted['Cumulative_Percentage'] <= 80, 'ABC_Category'] = 'A'
    inventory_df_sorted.loc[(inventory_df_sorted['Cumulative_Percentage'] > 80) & (inventory_df_sorted['Cumulative_Percentage'] <= 95), 'ABC_Category'] = 'B'
    
    if len(opportunities) > 0:
        opp_df = pd.DataFrame(opportunities)
        opp_df['Potential_Savings'] = opp_df['Potential_Savings'].apply(lambda x: f"${x:,.0f}")
        st.dataframe(opp_df, use_container_width=True)
        
        total_savings = sum([float(x['Potential_Savings'].replace(', '').replace(',', '')) for x in opportunities])
        st.success(f"💡 Total Potential Annual Savings: **${total_savings:,.0f}**")
    
    # ABC Analysis Chart
    st.subheader("📊 ABC Analysis - Focus Areas")
    
    abc_summary = inventory_df_sorted.groupby('ABC_Category').agg({
        'Product_ID': 'count',
        'Annual_Sales_Value': 'sum',
        'Total_Inventory_Value': 'sum'
    }).reset_index()
    abc_summary.columns = ['Category', 'Product_Count', 'Sales_Value', 'Inventory_Value']
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Products by ABC Category', 'Value Distribution'),
        specs=[[{"type": "pie"}, {"type": "bar"}]]
    )
    
    # Products pie chart
    fig.add_trace(
        go.Pie(labels=abc_summary['Category'], values=abc_summary['Product_Count'], name="Products"),
        row=1, col=1
    )
    
    # Value bar chart
    fig.add_trace(
        go.Bar(x=abc_summary['Category'], y=abc_summary['Sales_Value'], name="Sales Value", marker_color='lightblue'),
        row=1, col=2
    )
    fig.add_trace(
        go.Bar(x=abc_summary['Category'], y=abc_summary['Inventory_Value'], name="Inventory Value", marker_color='lightcoral'),
        row=1, col=2
    )
    
    fig.update_layout(height=400, showlegend=True)
    st.plotly_chart(fig, use_container_width=True)
    
    # Recommendations
    st.info("💡 **Focus Recommendations:**")
    st.markdown("""
    - **Category A (High Value)**: Implement tight inventory control, frequent reviews, and accurate forecasting
    - **Category B (Medium Value)**: Regular monitoring with moderate control measures
    - **Category C (Low Value)**: Simple reorder systems, bulk purchasing, and minimal oversight
    """)

def business_impact(metrics, inventory_df):
    st.header("📊 Business Impact & ROI Analysis")
    st.markdown("Quantified results and strategic recommendations for stakeholders")
    
    # Simulate before/after metrics for ROI calculation
    current_metrics = {
        'inventory_value': metrics['total_inventory_value'],
        'carrying_cost': metrics['annual_carrying_cost'],
        'stockout_rate': 5.2,  # Current stockout rate %
        'order_frequency': 156,  # Orders per year
        'staff_hours': 2080  # Hours spent on inventory management
    }
    
    # Projected improvements with AI system
    improved_metrics = {
        'inventory_value': current_metrics['inventory_value'] * 0.75,  # 25% reduction
        'carrying_cost': current_metrics['carrying_cost'] * 0.70,  # 30% reduction
        'stockout_rate': 1.8,  # Improved to 1.8%
        'order_frequency': current_metrics['order_frequency'] * 0.6,  # 40% fewer orders
        'staff_hours': current_metrics['staff_hours'] * 0.4  # 60% reduction in manual work
    }
    
    # Calculate savings
    inventory_savings = current_metrics['inventory_value'] - improved_metrics['inventory_value']
    carrying_cost_savings = current_metrics['carrying_cost'] - improved_metrics['carrying_cost']
    staff_cost_savings = (current_metrics['staff_hours'] - improved_metrics['staff_hours']) * 25  # $25/hour
    efficiency_savings = (current_metrics['order_frequency'] - improved_metrics['order_frequency']) * 50  # $50 per order
    
    total_annual_savings = carrying_cost_savings + staff_cost_savings + efficiency_savings
    
    # ROI Metrics Dashboard
    st.subheader("💰 Return on Investment Analysis")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Annual Cost Savings",
            f"${total_annual_savings:,.0f}",
            delta=f"+{(total_annual_savings/current_metrics['carrying_cost']*100):.1f}% improvement"
        )
    
    with col2:
        st.metric(
            "Inventory Reduction",
            f"${inventory_savings:,.0f}",
            delta="-25% inventory value"
        )
    
    with col3:
        st.metric(
            "Stockout Prevention",
            f"{improved_metrics['stockout_rate']:.1f}%",
            delta=f"-{current_metrics['stockout_rate'] - improved_metrics['stockout_rate']:.1f}% stockout rate"
        )
    
    with col4:
        implementation_cost = 150000  # Estimated AI system cost
        roi_months = implementation_cost / (total_annual_savings / 12)
        st.metric(
            "ROI Payback Period",
            f"{roi_months:.1f} months",
            delta="Fast ROI achievement"
        )
    
    # Before/After Comparison
    st.subheader("📈 Performance Improvement Comparison")
    
    comparison_data = {
        'Metric': ['Inventory Value', 'Annual Carrying Cost', 'Stockout Rate (%)', 'Staff Hours/Year', 'Orders/Year'],
        'Current State': [
            f"${current_metrics['inventory_value']:,.0f}",
            f"${current_metrics['carrying_cost']:,.0f}",
            f"{current_metrics['stockout_rate']:.1f}%",
            f"{current_metrics['staff_hours']:,}",
            f"{current_metrics['order_frequency']:,}"
        ],
        'With AI System': [
            f"${improved_metrics['inventory_value']:,.0f}",
            f"${improved_metrics['carrying_cost']:,.0f}",
            f"{improved_metrics['stockout_rate']:.1f}%",
            f"{improved_metrics['staff_hours']:,}",
            f"{improved_metrics['order_frequency']:,}"
        ],
        'Improvement': [
            f"-${inventory_savings:,.0f} (-25%)",
            f"-${carrying_cost_savings:,.0f} (-30%)",
            f"-{current_metrics['stockout_rate'] - improved_metrics['stockout_rate']:.1f}% (-67%)",
            f"-{current_metrics['staff_hours'] - improved_metrics['staff_hours']:,} (-60%)",
            f"-{current_metrics['order_frequency'] - improved_metrics['order_frequency']:,} (-40%)"
        ]
    }
    
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)
    
    # Visual ROI Timeline
    months = list(range(1, 37))  # 3 years
    cumulative_savings = [min(month * (total_annual_savings / 12) - implementation_cost, 
                             (month * (total_annual_savings / 12)) - implementation_cost) for month in months]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=months,
        y=cumulative_savings,
        mode='lines+markers',
        name='Cumulative Net Savings',
        line=dict(color='green', width=3)
    ))
    
    fig.add_hline(y=0, line_dash="dash", line_color="red", annotation_text="Break-even Point")
    fig.add_vline(x=roi_months, line_dash="dash", line_color="blue", 
                  annotation_text=f"ROI achieved in {roi_months:.1f} months")
    
    fig.update_layout(
        title='AI System ROI Timeline (3-Year Projection)',
        xaxis_title='Months',
        yaxis_title='Cumulative Net Savings ($)',
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Business Benefits Summary
    st.subheader("🎯 Strategic Business Benefits")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 💡 **Operational Excellence**")
        st.markdown("""
        ✅ **Automated Decision Making**: Reduce manual intervention by 60%  
        ✅ **Real-time Visibility**: Complete inventory transparency  
        ✅ **Predictive Analytics**: Prevent stockouts before they occur  
        ✅ **Supplier Optimization**: Improve vendor relationships  
        ✅ **Demand Accuracy**: 95%+ forecast reliability  
        """)
    
    with col2:
        st.markdown("### 📈 **Financial Impact**")
        st.markdown(f"""
        💰 **Annual Savings**: ${total_annual_savings:,.0f}  
        📉 **Inventory Reduction**: ${inventory_savings:,.0f}  
        ⚡ **Efficiency Gains**: ${staff_cost_savings:,.0f}  
        🎯 **ROI Timeline**: {roi_months:.1f} months  
        📊 **3-Year Value**: ${total_annual_savings * 3:,.0f}  
        """)
    
    # Implementation Roadmap
    st.subheader("🗺️ Implementation Roadmap")
    
    roadmap_data = {
        'Phase': ['Phase 1: Foundation', 'Phase 2: Integration', 'Phase 3: Optimization', 'Phase 4: Advanced AI'],
        'Duration': ['Months 1-2', 'Months 3-4', 'Months 5-6', 'Months 7-12'],
        'Key Activities': [
            'Data integration, basic forecasting, alert system',
            'Automated reordering, supplier integration, mobile access',
            'Advanced analytics, cost optimization, performance tuning',
            'Machine learning enhancement, predictive maintenance, IoT integration'
        ],
        'Expected ROI': ['15%', '45%', '75%', '100%']
    }
    
    roadmap_df = pd.DataFrame(roadmap_data)
    st.dataframe(roadmap_df, use_container_width=True, hide_index=True)
    
    # Call to Action
    st.subheader("🚀 Next Steps")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📋 Schedule Demo", type="primary"):
            st.success("Demo scheduled! Our team will contact you within 24 hours.")
    
    with col2:
        if st.button("📊 Request ROI Report"):
            st.success("Detailed ROI report will be sent to your email.")
    
    with col3:
        if st.button("🤝 Start Pilot Program"):
            st.success("Pilot program initiated! Implementation team assigned.")
    
    # Key Success Metrics
    st.markdown("---")
    st.subheader("🏆 Success Metrics to Track")
    
    success_metrics = pd.DataFrame({
        'KPI': [
            'Inventory Turnover Ratio',
            'Stockout Rate',
            'Carrying Cost Percentage',
            'Order Fulfillment Time',
            'Demand Forecast Accuracy',
            'Supplier Performance Score'
        ],
        'Current': ['4.2x', '5.2%', '25%', '3.2 days', '78%', '87%'],
        'Target': ['6.5x', '1.8%', '18%', '1.8 days', '95%', '95%'],
        'Business Impact': [
            'Faster cash conversion cycle',
            'Improved customer satisfaction',
            'Reduced storage and handling costs',
            'Enhanced customer experience',
            'Better planning and procurement',
            'Reliable supply chain performance'
        ]
    })
    
    st.dataframe(success_metrics, use_container_width=True, hide_index=True)

if __name__ == "__main__":
    main()
