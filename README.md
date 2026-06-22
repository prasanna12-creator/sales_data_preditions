# sales_data_preditions
# 📈 Predictive Sales Forecasting Dashboard

An interactive, data-driven web application built with Python and Streamlit that leverages machine learning to forecast future revenue. This project ingests historical sales data, preprocesses it into a continuous time-series, evaluates model accuracy, and projects a 30-day future sales trend using the Holt-Winters Exponential Smoothing algorithm.

🚀 Key Features

Automated Data Preprocessing: Automatically loads local CSV files, parses complex date formats, drops invalid rows, and resamples transactional data into a continuous daily timeline (filling missing days with zero sales).

Train/Test Split & Evaluation: Automatically isolates the last 30 days of historical data to test the model's accuracy, calculating the Root Mean Squared Error (RMSE) to grade prediction performance.

Time-Series Machine Learning: Utilizes the statsmodels library to apply an additive Holt-Winters Exponential Smoothing model, capturing historical trends to forecast future revenue.

Interactive Forecasting Visualizations: Uses Plotly Graph Objects to render a dynamic, interactive line chart comparing historical actuals (solid blue line) against the 30-day future prediction (dotted orange line).

Raw Data Inspection: Includes an expandable data table for users to inspect the exact dollar amounts forecasted for upcoming dates.

🛠️ Tech Stack

Python: Core programming language.

Streamlit: Web framework for the interactive user interface.

Pandas & NumPy: Data manipulation, datetime parsing, and mathematical aggregations.

Statsmodels: Implementation of the Holt-Winters time-series forecasting algorithm.

Scikit-Learn: Evaluation metrics (Mean Squared Error).

Plotly: High-performance, interactive data visualization.

📊 Data Requirements

This pipeline is designed to process standard sales data. For the code to run out-of-the-box, your CSV file must contain at least the following two columns:

Sale_Date: The date the transaction occurred (e.g., DD-MM-YYYY or YYYY-MM-DD).

Sales_Amount: The total revenue or monetary value of that specific transaction.

💻 Installation & Setup

1. Clone or Download the Repository
Save the forecast_app.py script to your local machine.

2. Install Dependencies
Open your terminal or PowerShell and run the following command to install the required Python libraries:

pip install streamlit pandas numpy scikit-learn statsmodels plotly


3. Configure Your Local Data Path
Open forecast_app.py and locate the file_path variable near the top of the script. Update this to the exact folder on your computer where your CSV file is stored:

file_path = r"C:\Path\To\Your\Data_Folder"


▶️ How to Run

Because this is a Streamlit web application, do not run it like a standard Python script (doing so will result in a missing ScriptRunContext error).

Instead, open your terminal or PowerShell, and execute the application using the streamlit run command followed by the path to your file:

streamlit run "c:/path/to/your/forecast_app.py"


Note: If your folder path contains spaces, make sure to wrap the entire path in quotation marks.

A local web server will spin up, and the predictive dashboard will automatically open in a new tab in your default web browser.
