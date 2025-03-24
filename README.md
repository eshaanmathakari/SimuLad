# SimuLad

SimuLad is an interactive ecosystem simulation and collaborative forecasting application that leverages sensor data from environments such as Rainforest, Ocean, Desert, and LEO‑W. The application uses multiple forecasting models (ARIMA and Prophet) and integrates local LLMs via Ollama for generating expert analyses and comparisons.

## Features

- **Data Visualization:**  
  View interactive visualizations including time-series plots and correlation heatmaps of sensor metrics.

- **Forecasting:**  
  Run forecasts for a selected ecosystem using ARIMA or Prophet. Compare forecasts across two different ecosystems side-by-side and have an LLM provide a detailed comparison analysis.

- **Expert Collaboration:**  
  Simulate a discussion among domain experts (Temperature, Humidity, Wind Speed, etc.) using local LLM models (choose from gemma3, deepseek‑r1, llama3.3, mistral, phi3). Experts generate actionable insights based on the sensor data and provided context.

## Demo Video of the Application 
[Simulad Demo for HackArizona 2025](https://youtu.be/k-h7qteMtAA)

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/simulad.git
   cd simulad

2. **Create and activate a virtual environment (optional but recommended):**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install the required dependencies:**

```bash
pip install -r requirements.txt
```
**Note-** Ensure you have Ollama installed and the desired local LLM models (e.g., gemma3, deepseek‑r1, llama3.3, mistral, phi3) available.

## Usage
### 1. **Prepare the Data:**
Place your merged sensor dataset as merged_data.csv in the project root. This CSV should contain a DateTime column, a Location column, and sensor metric columns. You can process the raw data provided using the `data-processing.py`

### 2. **Run the Application:**

```bash
streamlit run app.py
```
### 3. **Navigate Through Pages:**

**a.** Visualizations: Explore metric variations and correlation heatmaps.

**b.** Forecasting:
- Choose "Single Ecosystem" to forecast for one environment or "Compare Ecosystems" to forecast for two environments and compare their forecasts.
- Adjust simulation parameters (temperature and wind speed changes) and select your forecast model (ARIMA or Prophet).

**c.** Expert Collaboration:
- Select an expert LLM model.
- Generate expert discussions where simulated experts (Temperature, Humidity, Wind Speed) provide actionable insights based on the data.

## Future Scope
- Deep Embedded Agentic AI:
Develop an architecture where all expert agents can communicate and debate to determine the most reliable analysis automatically.

- Dynamic Dashboard:
Enhance the dashboard with more scientific visualizations (e.g., anomaly detection, multivariate trend analysis) and incorporate real-time data fetching and processing.

- Model Ensemble:
Explore combining forecasts from multiple models (VAR, ARIMA, Prophet) using ensemble methods to improve accuracy.

- User Feedback Integration:
Implement mechanisms for users to rate expert analyses and forecast comparisons to continuously fine-tune the LLM prompts and models.

