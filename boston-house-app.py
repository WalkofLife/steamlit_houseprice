import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import fetch_california_housing
st.write("""
# Boston House Price Prediction App
""")
st.write('---')

# Loads the boston data
boston = fetch_california_housing()
input_dataframe = pd.DataFrame(boston.data, columns=boston.feature_names)
input_dataframe = input_dataframe[:100]
X  =  input_dataframe[['HouseAge', 'AveRooms', 'AveBedrms', 'Population']]
Y = input_dataframe[['MedInc']]
# print(boston.feature_names)
#Sliderbar
st.sidebar.header('Specify Input parameters')

def user_input_features():
    Houseage = st.sidebar.slider('HouseAge', X.HouseAge.min(), X.HouseAge.max(), X.HouseAge.mean())
    AveRooms = st.sidebar.slider('AveRooms', X.AveRooms.min(), X.AveRooms.max(), X.AveRooms.mean())
    AveBedrms = st.sidebar.slider('Average Bedrooms',X.AveBedrms.min(), X.AveBedrms.max(), X.AveBedrms.mean())
    Population = st.sidebar.slider('Population', X.Population.min(), X.Population.max(), X.Population.mean())

    data = pd.DataFrame({
        'HouseAge' : Houseage,
        'AveRooms' : AveRooms,
        'AveBedrms' : AveBedrms,
        'Population' : Population

    }, index = [0])
    return data

df = user_input_features()

# Main panel
# Printing the Specified Input Parameters
st.header('Specified Input Parameters')
st.write(df)
st.write('----')

model = RandomForestRegressor()
model.fit(X, Y)

prediction = model.predict(df)

st.header('Prediction of MedInc')
st.write(prediction)
st.write('--')

# Feature Importance by SHAP
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

st.header('Feature Importance')
plt.title('Feature importance based n SHap values ')
fig, ax = plt.subplots() # Create a figure object
# Create the SHAP summary plot
shap.summary_plot(shap_values, X, show=False, plot_type="dot")
# Display the plot in Streamlit
st.pyplot(fig, bbox_inches='tight')

plt.title('Feature importance based n SHap values - bar ')
fig, ax = plt.subplots()
shap.summary_plot(shap_values, X, plot_type='bar')
st.pyplot(fig, bbox_inches='tight')
