# app.py
import streamlit as st
st.set_page_config(page_title="SimuLad", layout="wide", page_icon="🌱")
import pandas as pd
import plotly.express as px

from alternative_models import forecast_arima, forecast_prophet
from ai_integration import generate_summary
from experts import add_expert_message, generate_expert_response, get_conversation_log

st.title("SimuLad")

# --- Load Data ---
@st.cache_data
def load_merged_data(filepath="merged_data.csv"):
    data = pd.read_csv(filepath, parse_dates=["DateTime"])
    return data

merged_data = load_merged_data()

# Sidebar: Page selection
page = st.sidebar.radio("Select Page", ["Visualizations", "Forecasting", "Expert Collaboration"])

# --------------------------
# FORECASTING PAGE
# --------------------------
if page == "Forecasting":
    st.header("Forecasting")
    # Let the user choose between forecasting for a single ecosystem or comparing two.
    forecast_type = st.sidebar.radio("Forecast Type", ["Single Ecosystem", "Compare Ecosystems"])
    
    st.sidebar.subheader("Forecast Model")
    forecast_model = st.sidebar.selectbox("Select Forecast Model", ["ARIMA", "Prophet"])
    st.sidebar.subheader("Simulation Settings")
    temp_adjust = st.sidebar.slider("Temperature Change (°F)", -5.0, 5.0, 0.0, 0.5)
    wind_adjust = st.sidebar.slider("Wind Speed Change (m/s)", -5.0, 5.0, 0.0, 0.5)
    run_simulation = st.sidebar.button("Run Simulation")
    
    if forecast_type == "Single Ecosystem":
        selected_ecosystem = st.sidebar.selectbox("Select Ecosystem", merged_data["Location"].unique().tolist())
        df_ecosystem = merged_data[merged_data["Location"] == selected_ecosystem]
        sensor_cols = [col for col in df_ecosystem.columns if col not in ["DateTime", "Location"]]
        if not sensor_cols:
            st.error(f"No sensor data found for {selected_ecosystem}.")
        else:
            df_sim = df_ecosystem[["DateTime", "Location"] + sensor_cols]
            st.write(f"Data Preview for {selected_ecosystem}:", df_sim.head())
            # Prepare simulation data: interpolate missing sensor values.
            raw_simulation_data = df_sim[["DateTime"] + sensor_cols]
            simulation_data = raw_simulation_data.copy()
            simulation_data = simulation_data.set_index("DateTime")
            simulation_data[sensor_cols] = simulation_data[sensor_cols].interpolate(method='time').ffill().bfill()
            simulation_data = simulation_data.reset_index()
            st.write("Simulation data shape after interpolation:", simulation_data.shape)
            
            if simulation_data.shape[0] < 5:
                st.error("Not enough data available for forecasting. Please check your dataset.")
            else:
                forecast_df = None
                if run_simulation:
                    if forecast_model == "ARIMA":
                        st.info("Forecasting with ARIMA model...")
                        try:
                            forecast_df = forecast_arima(simulation_data, order=(1,1,1), steps=24)
                        except Exception as e:
                            st.error(f"ARIMA forecast failed: {e}")
                    elif forecast_model == "Prophet":
                        st.info("Forecasting with Prophet model...")
                        try:
                            forecast_df = forecast_prophet(simulation_data, steps=24)
                        except Exception as e:
                            st.error(f"Prophet forecast failed: {e}")
                    if forecast_df is not None:
                        st.subheader("Forecast for Next 24 Hours")
                        st.line_chart(forecast_df)
                        simulation_text = (f"In {selected_ecosystem}, Temperature adjusted by {temp_adjust}°F and Wind Speed by {wind_adjust} m/s. "
                                           f"Forecast using {forecast_model} shows the impact on related variables.")
                        summary = generate_summary(simulation_text)
                        st.subheader("AI-Generated Simulation Summary")
                        st.write(summary)
                        
    elif forecast_type == "Compare Ecosystems":
        st.subheader("Compare Forecasts for Two Ecosystems")
        ecos = merged_data["Location"].unique().tolist()
        ecosystem1 = st.sidebar.selectbox("Select Ecosystem 1", ecos, key="eco1")
        ecosystem2 = st.sidebar.selectbox("Select Ecosystem 2", ecos, key="eco2")
        if ecosystem1 == ecosystem2:
            st.error("Please select two different ecosystems for comparison.")
        else:
            # Prepare data for each ecosystem.
            df1 = merged_data[merged_data["Location"] == ecosystem1]
            df2 = merged_data[merged_data["Location"] == ecosystem2]
            sensor_cols1 = [col for col in df1.columns if col not in ["DateTime", "Location"]]
            sensor_cols2 = [col for col in df2.columns if col not in ["DateTime", "Location"]]
            if not sensor_cols1 or not sensor_cols2:
                st.error("One or both ecosystems lack sensor data.")
            else:
                st.write(f"Data Preview for {ecosystem1}:", df1.head())
                st.write(f"Data Preview for {ecosystem2}:", df2.head())
                def prepare_data(df, sensors):
                    raw = df[["DateTime"] + sensors]
                    sim = raw.copy().set_index("DateTime")
                    sim[sensors] = sim[sensors].interpolate(method='time').ffill().bfill()
                    return sim.reset_index()
                sim1 = prepare_data(df1, sensor_cols1)
                sim2 = prepare_data(df2, sensor_cols2)
                st.write(f"Simulation data shape for {ecosystem1}: {sim1.shape}")
                st.write(f"Simulation data shape for {ecosystem2}: {sim2.shape}")
                if sim1.shape[0] < 5 or sim2.shape[0] < 5:
                    st.error("Not enough data available for one or both ecosystems. Please check your dataset.")
                else:
                    forecast1 = forecast2 = None
                    if run_simulation:
                        if forecast_model == "ARIMA":
                            st.info("Forecasting with ARIMA model for both ecosystems...")
                            try:
                                forecast1 = forecast_arima(sim1, order=(1,1,1), steps=24)
                                forecast2 = forecast_arima(sim2, order=(1,1,1), steps=24)
                            except Exception as e:
                                st.error(f"ARIMA forecast failed: {e}")
                        elif forecast_model == "Prophet":
                            st.info("Forecasting with Prophet model for both ecosystems...")
                            try:
                                forecast1 = forecast_prophet(sim1, steps=24)
                                forecast2 = forecast_prophet(sim2, steps=24)
                            except Exception as e:
                                st.error(f"Prophet forecast failed: {e}")
                        if forecast1 is not None and forecast2 is not None:
                            st.subheader("Forecast for Next 24 Hours")
                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown(f"**Forecast for {ecosystem1}**")
                                st.line_chart(forecast1)
                            with col2:
                                st.markdown(f"**Forecast for {ecosystem2}**")
                                st.line_chart(forecast2)
                            # Compare the two forecasts using LLM.
                            compare_prompt = (
                                f"Compare the following forecasts for two ecosystems:\n\n"
                                f"Forecast for {ecosystem1}:\n{forecast1.to_string(index=True)}\n\n"
                                f"Forecast for {ecosystem2}:\n{forecast2.to_string(index=True)}\n\n"
                                "Provide a detailed comparison analysis highlighting key differences, potential causes, "
                                "and actionable recommendations based on these forecasts."
                            )
                            comparison = generate_summary(compare_prompt)
                            st.subheader("LLM Comparison of Forecasts")
                            st.write(comparison)

# --------------------------
# EXPERT COLLABORATION PAGE
# --------------------------
elif page == "Expert Collaboration":
    st.header("Expert Collaboration (AI Experts)")
    st.markdown("Simulated expert discussion among AI specialists analyzing the current ecosystem data.")
    expert_model = st.sidebar.selectbox("Select Expert Model", ["gemma3", "deepseek-r1", "llama3.3", "mistral", "phi3"])
    # Do not display any raw data summary.
    data_summary_input = ""
    log = get_conversation_log()
    if log:
        for entry in log:
            st.markdown(f"**{entry['timestamp']} - {entry['expert']}**: {entry['message']}")
    else:
        st.info("No expert messages yet.")
    if st.button("Generate Expert Discussion"):
        add_expert_message("Temperature Expert", "Based on the latest sensor data, I observe a subtle upward trend in temperature.")
        context_temp = ("Temperature Expert: I observe a subtle upward trend in temperature that could affect other variables. "
                        "Please provide an in-depth analysis of the potential impacts on the ecosystem.")
        generate_expert_response("Temperature Expert", context_temp, data_summary=data_summary_input, model_choice=expert_model)
        
        add_expert_message("Humidity Expert", "Based on the temperature trends, I expect corresponding changes in humidity levels.")
        context_humidity = ("Humidity Expert: Considering the observed temperature trends and their potential influence on moisture, "
                            "please analyze how humidity levels might change and suggest actionable recommendations.")
        generate_expert_response("Humidity Expert", context_humidity, data_summary=data_summary_input, model_choice=expert_model)
        
        add_expert_message("Wind Speed Expert", "Observations indicate fluctuations in wind speed which may affect dispersion patterns.")
        context_wind = ("Wind Speed Expert: Given the variability in wind speed from the dataset, please evaluate how these fluctuations "
                        "might influence overall ecosystem dynamics, particularly pollutant dispersion or microclimate effects.")
        generate_expert_response("Wind Speed Expert", context_wind, data_summary=data_summary_input, model_choice=expert_model)
        
        st.success("Expert discussion updated.")
        log = get_conversation_log()
        for entry in log:
            st.markdown(f"**{entry['timestamp']} - {entry['expert']}**: {entry['message']}")

# --------------------------
# VISUALIZATIONS PAGE
# --------------------------
elif page == "Visualizations":
    st.header("Visualizations")
    viz_page = st.sidebar.radio("Select Visualization", ["Metric Variation Over Time", "Correlation Heatmap"])
    # For visualizations, define ecosystem and sensor_cols here.
    selected_ecosystem = st.sidebar.selectbox("Select Ecosystem", merged_data["Location"].unique().tolist(), key="vizEco")
    df_ecosystem = merged_data[merged_data["Location"] == selected_ecosystem]
    sensor_cols = [col for col in df_ecosystem.columns if col not in ["DateTime", "Location"]]
    df_sim = df_ecosystem[["DateTime", "Location"] + sensor_cols]
    
    if viz_page == "Correlation Heatmap":
        st.subheader("Sensor Data Correlation")
        corr_matrix = df_sim[sensor_cols].corr()
        fig_corr = px.imshow(corr_matrix, text_auto=True, aspect="auto", title="Correlation Heatmap")
        st.plotly_chart(fig_corr, use_container_width=True)
    elif viz_page == "Metric Variation Over Time":
        st.subheader("Metric Variation Over Time")
        selected_metric = st.selectbox("Select Metric", sensor_cols)
        fig_line = px.line(df_sim, x="DateTime", y=selected_metric, title=f"{selected_metric} Over Time")
        st.plotly_chart(fig_line, use_container_width=True)
