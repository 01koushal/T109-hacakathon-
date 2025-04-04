import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.api import ExponentialSmoothing

st.set_page_config(page_title="Water Forecast Dashboard", layout="wide")

st.title("💧 Water Forecasting & Reservoir Assessment")

uploaded_file = st.file_uploader("Upload water usage CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("Data uploaded successfully!")

    st.subheader("📊 Raw Data Preview")
    st.dataframe(df)

    st.subheader("📈 Total Water Usage Over Time")
    fig, ax = plt.subplots()
    sns.lineplot(data=df, x="Year", y="Total_Usage", marker="o", ax=ax)
    ax.set_title("Total Water Usage Over the Years")
    ax.set_ylabel("Water Usage (in million cubic meters)")
    st.pyplot(fig)

    st.subheader("🔮 Forecast Future Water Requirements")

    df.set_index("Year", inplace=True)
    model = ExponentialSmoothing(df["Total_Usage"], trend='add', seasonal=None)
    model_fit = model.fit()
    forecast = model_fit.forecast(5)

    forecast_df = forecast.reset_index()
    forecast_df.columns = ["Year", "Predicted_Usage"]
    st.write("Forecast for the next 5 years:")
    st.dataframe(forecast_df)

    st.subheader("📉 Forecast Plot")
    fig2, ax2 = plt.subplots()
    df["Total_Usage"].plot(label="Historical", ax=ax2)
    forecast.plot(label="Forecast", ax=ax2)
    ax2.set_xlabel("Year")
    ax2.set_ylabel("Water Usage")
    ax2.set_title("Forecasted Water Usage")
    ax2.legend()
    st.pyplot(fig2)

else:
    st.info("Please upload a valid CSV file with water usage data.")
