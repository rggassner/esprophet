# ELK Time-Series Anomaly Detector (Prophet Regression)

A Python tool that performs seasonal regression analysis on Elasticsearch logs to detect anomalies. It uses **Facebook Prophet** to model daily and weekly patterns, generating visual alerts (PNGs) when traffic exceeds statistical thresholds.

## Features

* **Seasonal Analysis:** Automatically accounts for daily and weekly traffic cycles.
* **Decoupled Architecture:** Configuration and Elasticsearch queries are stored in external files.
* **Visual Evidence:** Generates annotated PNG plots for every detected anomaly.
* **Quiet Alerting:** Includes a percentage-based buffer and volume floor to reduce false positives.
* **Multi-Entities Support:** Dynamically processes the top $N$ entities (domains, hosts, or IPs) found in your logs.

## Installation

1. **Clone the repository:**
   
   ```
   git clone https://github.com/rggassner/esprophet
   cd esprophet
   ```

2.  **Create a virtual environment and activate it:**
    
    ```
    python3 -m venv venv
    source venv/bin/activate
    ```
    
3.  **Install dependencies:**
    
    ```
    pip install -r requirements.txt
    ```
    
## Configuration

### 1\. Environment Variables (`.env`)

Create a `.env` file in the root directory to map your specific Elasticsearch fields:

### 2\. Query Template (`query_template.json`)

The script injects your `.env` settings into this template. You can modify the filters here without touching the Python code.

## How it Works

The script follows a 4-step pipeline:

1.  **Ingest:** Pulls 30 days of hourly aggregated data from Elasticsearch.
2.  **Train:** Fits a Prophet model using the first 29 days, enabling daily and weekly seasonality (Fourier Order\=10).
3.  **Predict:** Generates a "Preview" (Trend) and a 99% confidence interval (Min/Max) for the last 24 hours.
4.  **Evaluate:** If the observed data exceeds the `Max_Threshold * BUFFER_PERCENT`, an anomaly is flagged and a PNG is saved to the output directory.


## Example Output

When an anomaly is detected, a plot is generated showing:

-   **Black Line:** Actual observed traffic.
-   **Blue Shaded Area:** The calculated "Normal Range."
-   **Red Dots:** Statistical outliers that breached the buffered threshold.
-   **Summary Box:** Total document count for the analysis period.
