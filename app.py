import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder
import holidays
import matplotlib.pyplot as plt
import seaborn as sns

# Set page config
st.set_page_config(
    page_title="Store Sales Forecast",
    page_icon="🛒",
    layout="wide"
)

# App title and description
st.title("🛒 Store Sales Time Series Forecasting")
st.markdown("""
This app predicts store sales using an XGBoost model trained on historical data.
""")

# Sidebar for user inputs
with st.sidebar:
    st.header("Model Configuration")
    st.markdown("Configure the forecasting model parameters")
    
    # Model parameters
    n_estimators = st.slider("Number of estimators", 100, 2000, 1000, 100)
    learning_rate = st.slider("Learning rate", 0.001, 0.1, 0.01, 0.001)
    
    st.header("Data Options")
    show_raw_data = st.checkbox("Show raw data", value=False)
    show_eda = st.checkbox("Show exploratory analysis", value=False)

# Load data function with caching
@st.cache_data
def load_data():
    train = pd.read_csv('train.csv', parse_dates=['date'])
    test = pd.read_csv('test.csv', parse_dates=['date'])
    stores = pd.read_csv('stores.csv')
    oil = pd.read_csv('oil.csv', parse_dates=['date'])
    
    # Merge data
    train = pd.merge(train, stores, on='store_nbr', how='left')
    test = pd.merge(test, stores, on='store_nbr', how='left')
    
    # Process oil data
    oil['dcoilwtico'] = oil['dcoilwtico'].replace('.', np.nan).astype(float)
    oil['dcoilwtico'] = oil.set_index('date')['dcoilwtico'].interpolate(method='time').values
    
    train = pd.merge(train, oil, on='date', how='left')
    test = pd.merge(test, oil, on='date', how='left')
    
    # Add holiday features
    ec_holidays = holidays.CountryHoliday('EC', years=range(2013, 2018))
    train['is_holiday'] = train['date'].apply(lambda x: x in ec_holidays)
    test['is_holiday'] = test['date'].apply(lambda x: x in ec_holidays)
    
    # Add time features
    for df in [train, test]:
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        df['day_of_week'] = df['date'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    # Encode family
    le = LabelEncoder()
    train['family'] = le.fit_transform(train['family'])
    test['family'] = le.transform(test['family'])
    
    return train, test

# Load data
train, test = load_data()

# Show raw data if selected
if show_raw_data:
    st.subheader("Raw Data Preview")
    tab1, tab2 = st.tabs(["Training Data", "Test Data"])
    
    with tab1:
        st.dataframe(train.head())
    
    with tab2:
        st.dataframe(test.head())

# EDA section
if show_eda:
    st.subheader("Exploratory Data Analysis")
    
    # Sales distribution
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    sns.histplot(train['sales'], bins=50, kde=True, ax=ax1)
    ax1.set_title("Sales Distribution")
    st.pyplot(fig1)
    
    # Sales over time
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    train.groupby('date')['sales'].sum().plot(ax=ax2)
    ax2.set_title("Total Sales Over Time")
    st.pyplot(fig2)

# Model training and prediction
st.subheader("Model Training & Prediction")

# Define features
features = ['store_nbr', 'family', 'onpromotion', 'dcoilwtico', 
            'year', 'month', 'day', 'day_of_week', 'is_weekend', 
            'is_holiday', 'cluster']

X = train[features]
y = train['sales']

# Train model
if st.button("Train Model"):
    with st.spinner("Training model..."):
        model = XGBRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=42
        )
        model.fit(X, y)
        
        # Make predictions
        X_test = test[features]
        test['sales'] = model.predict(X_test)
        
        # Show some predictions
        st.success("Model training completed!")
        
        # Display sample predictions
        st.dataframe(test[['date', 'store_nbr', 'family', 'sales']].head())
        
        # Plot predictions for a sample store
        sample_store = test['store_nbr'].sample(1).values[0]
        store_data = test[test['store_nbr'] == sample_store]
        
        fig3, ax3 = plt.subplots(figsize=(12, 6))
        store_data.groupby('date')['sales'].sum().plot(ax=ax3)
        ax3.set_title(f"Predicted Sales for Store {sample_store}")
        st.pyplot(fig3)
        
        # Download predictions
        csv = test[['id', 'sales']].to_csv(index=False)
        st.download_button(
            label="Download Predictions",
            data=csv,
            file_name='sales_predictions.csv',
            mime='text/csv'
        )

# Footer
st.markdown("---")
st.markdown("""
**Note:** This app uses an XGBoost model for time series forecasting of store sales.
Adjust the parameters in the sidebar to optimize the model.
""")
