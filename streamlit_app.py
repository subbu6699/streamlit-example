import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Function to load and cache the data
@st.cache
def load_data(data):
    return pd.read_csv(data)

# Function to plot interactive scatter plot
def plot_interactive(data, x, y):
    fig = px.scatter(data_frame=data, x=x, y=y, title=f'Scatter plot of {x} vs {y}')
    return fig

# Function to perform and show linear regression
def linear_regression(data, features, target):
    X = data[features]
    y = data[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return model, mse, r2, X_train, X_test, y_train, y_test, y_pred

# Title of the web app
st.title('Advanced Data Exploration App')

# Sidebar for configuration
st.sidebar.title('Configuration')
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    data = load_data(uploaded_file)

    # Show the raw data
    if st.sidebar.checkbox('Show raw data', False):
        st.subheader('Raw Data')
        st.write(data)

    # Select columns to plot
    st.sidebar.subheader('Data Visualization')
    all_columns = data.columns.tolist()
    selected_columns = st.sidebar.multiselect('Select columns to plot', all_columns, default=all_columns[:2])

    # Plotting using seaborn in main area
    if len(selected_columns) == 2:
        st.subheader('Seaborn Pair Plot')
        sns.pairplot(data[selected_columns])
        st.pyplot()

        # Interactive Plotly plot
        st.subheader('Interactive Plot by Plotly')
        st.plotly_chart(plot_interactive(data, selected_columns[0], selected_columns[1]))

    # Machine Learning Section
    st.sidebar.subheader('Machine Learning')
    if st.sidebar.checkbox('Apply Linear Regression', False):
        # Select features and target for regression
        features = st.sidebar.multiselect('Select features for regression', all_columns, default=all_columns[:len(all_columns)-1])
        target = st.sidebar.selectbox('Select target for regression', all_columns, index=len(all_columns)-1)

        if len(features) > 0 and target:
            model, mse, r2, X_train, X_test, y_train, y_test, y_pred = linear_regression(data, features, target)

            st.subheader('Linear Regression Results')
            st.write('Mean Squared Error:', mse)
            st.write('R^2 Score:', r2)

            st.subheader('Regression Coefficients')
            coefficients = pd.DataFrame(model.coef_, features, columns=['Coefficient'])
            st.write(coefficients)

            st.subheader('Predictions vs Actuals')
            comparison = pd.DataFrame({'Actual': y_test, 'Prediction': y_pred})
            st.write(comparison)

            st.subheader('Interactive Prediction Plot')
            fig = px.scatter(x=y_test, y=y_pred, labels={'x': 'Actual', 'y': 'Prediction'}, title='Actual vs Prediction')
            fig.add_scatter(x=[y.min(), y.max()], y=[y.min(), y.max()], mode='lines', showlegend=False)
            st.plotly_chart(fig)
else:
    st.write("Upload a CSV file to get started")
