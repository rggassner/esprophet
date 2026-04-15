#!venv/bin/python3
#pylint: disable=broad-exception-caught
import os
import json
from datetime import datetime, timezone
import pandas as pd
from prophet import Prophet
from elasticsearch import Elasticsearch, helpers
import urllib3
import matplotlib.pyplot as plt
from dotenv import load_dotenv

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
load_dotenv()

MIN_DOCS = int(os.getenv("MIN_DOCS", "150000"))
ES_HOST = os.getenv("ES_HOST")
ES_USER = os.getenv("ES_USER")
ES_PASS = os.getenv("ES_PASS")
INDEX_PATTERN = os.getenv("INDEX_PATTERN", "logs-waf-prod*")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./alerts_plots")
BUFFER_PERCENT = float(os.getenv("BUFFER_PERCENT", "1.20"))
INTERVAL_WIDTH = float(os.getenv("INTERVAL_WIDTH", "0.99"))
TIMESTAMP = os.getenv("TIMESTAMP_FIELD", "@timestamp")
GRAIN = os.getenv("GRAIN_FIELD", "dest_host")
ANALYSIS_START = os.getenv("ANALYSIS_START", "now-30d")
ANALYSIS_END = os.getenv("ANALYSIS_END", "now-1h")
FIXED_INTERVAL = os.getenv("FIXED_INTERVAL", "1h")
AGG_SIZE = os.getenv("AGG_SIZE", "200")
MINIMUM_SAMPLES = int(os.getenv("MINIMUM_SAMPLES", "168"))
PLOT_UPPER = os.getenv("PLOT_UPPER_ALERTS", "True") == "True"
PLOT_LOWER = os.getenv("PLOT_LOWER_ALERTS", "False") == "True"
ENABLE_ES_INGEST = os.getenv("ENABLE_ES_INGEST", "True") == "True"
RESULTS_INDEX = os.getenv("RESULTS_INDEX", "anomaly-predictions")
PREDICT_DAYS = int(os.getenv("PREDICT_FUTURE_DAYS", "7"))
GENERATE_PLOTS = os.getenv("GENERATE_PLOTS", "False") == "True"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

es = Elasticsearch(
    [ES_HOST],
    basic_auth=(ES_USER, ES_PASS),
    verify_certs=False,
    ssl_show_warn=False
)

def ingest_to_elastic(df_forecast, df_actual, entity_name):
    """
    Ingest forecast results and observed data into Elasticsearch as structured
    anomaly detection documents.

    This function merges Prophet forecast output with actual observed values,
    evaluates anomaly conditions, and performs bulk indexing into the configured
    Elasticsearch index. Each document represents a single timestamped data point
    for a given entity, enriched with prediction bounds and anomaly flags.

    Processing Steps:
        1. Merge forecast (yhat, bounds) with actual observations (y).
        2. Generate a unique document ID per entity and timestamp to prevent
           duplication and ensure idempotency.
        3. Evaluate anomaly conditions:
            - Upper anomaly: observed > yhat_upper * BUFFER_PERCENT
            - Lower anomaly: observed < yhat_lower
        4. Construct normalized documents with forecast, thresholds, and metadata.
        5. Bulk ingest documents into Elasticsearch using helpers.bulk().

    Args:
        df_forecast (pandas.DataFrame):
            Prophet forecast DataFrame containing:
                - 'ds' (datetime): Timestamp
                - 'yhat' (float): Predicted value
                - 'yhat_lower' (float): Lower confidence bound
                - 'yhat_upper' (float): Upper confidence bound

        df_actual (pandas.DataFrame):
            Observed data containing:
                - 'ds' (datetime): Timestamp
                - 'y' (float/int): Observed value

        entity_name (str):
            Identifier for the entity (e.g., hostname, service name).
            Used for document partitioning and ID generation.

    Global Dependencies:
        es (Elasticsearch): Initialized Elasticsearch client
        RESULTS_INDEX (str): Target index for anomaly documents
        INDEX_PATTERN (str): Source index pattern for traceability
        GRAIN (str): Field representing the aggregation grain
        BUFFER_PERCENT (float): Multiplier for upper anomaly sensitivity

    Output:
        None. Documents are indexed directly into Elasticsearch.

    Raises:
        Exception:
            Catches and logs bulk ingestion failures without interrupting
            the overall processing pipeline.

    Notes:
        - Document IDs are normalized to avoid invalid characters by replacing
          dots and spaces with underscores.
        - Missing observed values (NaN) result in anomaly flags being set to False.
        - Boolean fields are explicitly cast to Python bool to ensure proper
          Elasticsearch mapping (avoiding numpy.bool_ issues).
        - Designed for high-throughput ingestion using Elasticsearch bulk API.
    """
    # Merge actuals (y) into forecast (yhat)
    merged = df_forecast.merge(df_actual[['ds', 'y']], on='ds', how='left')

    actions = []
    for _, row in merged.iterrows():
        # Create a unique ID that prevents collisions across different grains/entities
        timestamp_str = row['ds'].strftime('%Y-%m-%dT%H:%M:%SZ')
        # Format: grain_entity_timestamp (e.g., dest_host_example_com_2023-10-01T10:00:00Z)
        doc_id = f"{GRAIN}_{entity_name}_{timestamp_str}".replace(".", "_").replace(" ", "_")

        # Anomaly logic based on your global constants
        is_upper = False
        is_lower = False

        if pd.notnull(row['y']):
            # Cast to bool to ensure Elastic sees 'boolean' type, not 'numpy.bool_'
            is_upper = bool(row['y'] > (row['yhat_upper'] * BUFFER_PERCENT))
            is_lower = bool(row['y'] < row['yhat_lower'])

        doc = {
            "_index": RESULTS_INDEX,
            "_id": doc_id,
            "@timestamp": row['ds'],
            "source_index": INDEX_PATTERN,  # Standardized name for your INDEX_PATTERN
            "source_grain": GRAIN,          # Standardized name for your GRAIN_FIELD
            "entity_name": entity_name,
            "observed": float(row['y']) if pd.notnull(row['y']) else None,
            "preview": float(row['yhat']),
            "min_threshold": float(row['yhat_lower']),
            "max_threshold": float(row['yhat_upper']),
            "is_upper_anomaly": is_upper,
            "is_lower_anomaly": is_lower,
            "model_run_at": datetime.now(timezone.utc)
        }
        actions.append(doc)

    try:
        if actions:
            helpers.bulk(es, actions)
    except Exception as e:
        print(f"Failed to bulk ingest for {entity_name}: {e}")

def load_templated_query():
    """
    Load and render the Elasticsearch query template by replacing placeholder
    variables with runtime configuration values.

    This function reads a JSON query template from 'query_template.json' and
    dynamically injects environment-specific parameters such as time range,
    aggregation fields, and thresholds. The resulting query is returned as a
    Python dictionary ready to be executed via the Elasticsearch client.

    Template Placeholders Replaced:
        {{TIMESTAMP_FIELD}}  → Field used for time-based aggregation
        {{GRAIN_FIELD}}      → Field used for entity grouping (e.g., dest_host)
        {{MIN_DOCS}}         → Minimum document threshold for filtering
        {{ANALYSIS_START}}   → Start of the analysis time window
        {{ANALYSIS_END}}     → End of the analysis time window
        {{FIXED_INTERVAL}}   → Aggregation interval (e.g., 1h)
        {{AGG_SIZE}}         → Number of top entities to retrieve

    Returns:
        dict:
            Parsed Elasticsearch query body with all placeholders resolved,
            ready to be passed to es.search().

    Raises:
        FileNotFoundError:
            If 'query_template.json' is not found in the working directory.
        json.JSONDecodeError:
            If the template file contains invalid JSON.
        Exception:
            For any unexpected errors during file reading or processing.

    Notes:
        - All replacement values are sourced from environment variables.
        - This function assumes placeholders exist exactly as defined; missing
          or misspelled placeholders will not raise errors but may result in
          malformed queries.
        - Designed to decouple query structure from runtime configuration,
          making tuning and reuse easier across environments.
    """
    with open('query_template.json', 'r', encoding='utf-8') as f:
        query_str = f.read()
    query_str = query_str.replace("{{TIMESTAMP_FIELD}}", TIMESTAMP)
    query_str = query_str.replace("{{GRAIN_FIELD}}", GRAIN)
    query_str = query_str.replace("{{MIN_DOCS}}", os.getenv("MIN_DOCS"))
    query_str = query_str.replace("{{ANALYSIS_START}}", os.getenv("ANALYSIS_START"))
    query_str = query_str.replace("{{ANALYSIS_END}}", os.getenv("ANALYSIS_END"))
    query_str = query_str.replace("{{FIXED_INTERVAL}}", os.getenv("FIXED_INTERVAL"))
    query_str = query_str.replace("{{AGG_SIZE}}", os.getenv("AGG_SIZE"))
    return json.loads(query_str)

def generate_plot(df, forecast, entity_name):
    """
    Generate and save a time-series visualization comparing observed data
    against Prophet forecast predictions, including anomaly highlighting.

    This function produces a PNG plot per entity, showing:
    - Actual observed values over time
    - Forecasted trend (yhat)
    - Expected value range (yhat_lower to yhat_upper) as a confidence band
    - Optional anomaly markers for values outside expected bounds

    Visualization Features:
        - Upper anomalies (spikes) are flagged when observed values exceed
          yhat_upper multiplied by BUFFER_PERCENT.
        - Lower anomalies (drops) are flagged when observed values fall below
          yhat_lower.
        - A summary statistic (total document count) is displayed on the plot.
        - Output is saved to OUTPUT_DIR using a sanitized entity-based filename.

    Args:
        df (pandas.DataFrame):
            Historical observed data with the following columns:
                - 'ds' (datetime): Timestamp of the observation
                - 'y' (float/int): Observed value (e.g., document count)

        forecast (pandas.DataFrame):
            Prophet forecast output containing:
                - 'ds' (datetime): Timestamp
                - 'yhat' (float): Predicted value
                - 'yhat_lower' (float): Lower confidence bound
                - 'yhat_upper' (float): Upper confidence bound

        entity_name (str):
            Identifier for the entity being plotted (e.g., hostname, service).
            Used in the plot title and output filename.

    Global Dependencies:
        OUTPUT_DIR (str): Directory where plots are saved
        BUFFER_PERCENT (float): Multiplier for upper anomaly sensitivity
        PLOT_UPPER (bool): Toggle for plotting upper anomalies
        PLOT_LOWER (bool): Toggle for plotting lower anomalies

    Output:
        Saves a PNG file named "<entity_name>.png" in OUTPUT_DIR.

    Notes:
        - The function does not return any value.
        - Datetime values are expected to be timezone-naive and aligned
          between df and forecast.
        - Designed for debugging and visual validation of anomaly detection
          behavior in the forecasting pipeline.
    """
    plt.figure(figsize=(18, 6))
    total_docs = df['y'].sum()

    # Base Lines
    plt.plot(df['ds'], df['y'], color='black', label='Observed (Actual)', alpha=0.5, linewidth=1)
    plt.plot(forecast['ds'], forecast['yhat'], color='blue', label='Preview (Trend)', linewidth=2)

    # Shaded Corridor
    plt.fill_between(forecast['ds'],
                     forecast['yhat_lower'].clip(lower=0),
                     forecast['yhat_upper'],
                     color='blue', alpha=0.2, label='Normal Range (Min/Max)')

    merged = df.merge(forecast[['ds', 'yhat_lower', 'yhat_upper']], on='ds')

    # --- Conditional Upper Plotting ---
    if PLOT_UPPER:
        upper_anomalies = merged[merged['y'] > (merged['yhat_upper'] * BUFFER_PERCENT)]
        if not upper_anomalies.empty:
            plt.scatter(upper_anomalies['ds'], upper_anomalies['y'],
                        color='red', label='Upper Alert', zorder=5)

    # --- Conditional Lower Plotting ---
    if PLOT_LOWER:
        # Note: We usually don't use a buffer for drops (a drop is a drop)
        lower_anomalies = merged[merged['y'] < merged['yhat_lower']]
        if not lower_anomalies.empty:
            plt.scatter(lower_anomalies['ds'], lower_anomalies['y'],
                        color='orange', label='Lower Alert', zorder=5)

    # Stats and Formatting
    stats_text = f"Total Docs: {total_docs:,}"
    plt.gca().text(0.98, 0.95, stats_text, transform=plt.gca().transAxes,
                   fontsize=10, fontweight='bold', verticalalignment='top',
                   horizontalalignment='right',
                   bbox={
                    "boxstyle": "round",
                    "facecolor": "white",
                    "alpha": 0.8,
                    "edgecolor": "gray"
                    }
                   )

    plt.title(f"{entity_name} (Sensitivity: {BUFFER_PERCENT})")
    plt.xlabel("Time (UTC)")
    plt.ylabel("Counts per Hour")
    plt.legend(loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.7)

    filename = entity_name.replace(".", "_") + ".png"
    plt.savefig(os.path.join(OUTPUT_DIR, filename), bbox_inches='tight')
    plt.close()

def run_analysis():
    """
    Orchestrates the end-to-end anomaly detection pipeline for all target entities.

    The process follows a five-step lifecycle:
    1.  **Ingestion:** Executes a templated Elasticsearch aggregation to fetch 
        time-series data for the top N entities (defined by GRAIN_FIELD).
    2.  **Preprocessing:** Converts raw ES buckets into a Pandas DataFrame and 
        validates against MINIMUM_SAMPLES to ensure statistical relevance.
    3.  **Modeling:** Fits a Facebook Prophet model with hourly granularity, 
        accounting for daily and weekly seasonality patterns.
    4.  **Forecasting:** Generates a historical "preview" and a future projection 
        extending PREDICT_DAYS into the future.
    5.  **Output:** Standardizes results into the RESULTS_INDEX via bulk ingestion 
        and optionally renders debug plots if an anomaly is detected in the 
        trailing 24-hour window.

    Global Settings (via .env):
        INDEX_PATTERN (str): The source log index to query.
        MINIMUM_SAMPLES (int): Minimum hourly data points required to train.
        PREDICT_DAYS (int): How many days to forecast beyond the current time.
        ENABLE_ES_INGEST (bool): Toggle for pushing results back to Elasticsearch.
        GENERATE_PLOTS (bool): Toggle for saving PNG visualizations.

    Raises:
        Exception: Catches and logs per-entity failures during modeling or 
            ingestion to prevent a single failure from halting the entire batch.
    """
    query = load_templated_query()
    res = es.search(index=os.getenv("INDEX_PATTERN"), body=query)

    for bucket in res['aggregations']['target_grain']['buckets']:
        entity_name = bucket['key']

        # 1. Prepare Data
        df = pd.DataFrame([
            {'ds': b['key_as_string'], 'y': b['doc_count']}
            for b in bucket['timeseries']['buckets']
        ])
        df['ds'] = pd.to_datetime(df['ds']).dt.tz_localize(None)

        if len(df) < int(MINIMUM_SAMPLES):
            print(f"Skipping {entity_name}: insufficient data ({len(df)} samples).")
            continue

        try:
            # 2. Build and Fit Model
            model = Prophet(
                interval_width=INTERVAL_WIDTH,
                daily_seasonality=True,
                weekly_seasonality=False
            )
            model.add_seasonality(name='weekly', period=7, fourier_order=10)

            # Fit on all historical data available
            model.fit(df)

            # 3. Create Future Dataframe (History + Next N Days)
            future = model.make_future_dataframe(periods=24 * PREDICT_DAYS, freq='h')
            forecast = model.predict(future)

            # 4. Global Ingestion (Always aligned with BUFFER_PERCENT)
            if ENABLE_ES_INGEST:
                print(f"Pushing results for {entity_name} to {RESULTS_INDEX}...")
                ingest_to_elastic(forecast, df, entity_name)

            # 5. Conditional Debug Plotting
            if GENERATE_PLOTS:
                # Merge last 24h of actuals with forecast to check for triggers
                check = df.iloc[-24:].merge(forecast[['ds', 'yhat_upper', 'yhat_lower']], on='ds')
                is_upper = (check['y'] > (check['yhat_upper'] * BUFFER_PERCENT)).any()
                is_lower = (check['y'] < check['yhat_lower']).any()

                if is_upper or is_lower:
                    print(f"Generating debug plot for {entity_name} (Anomaly Detected).")
                    generate_plot(df, forecast, entity_name)
                else:
                    print(f"No anomaly in last 24h for {entity_name}, skipping plot.")

        except Exception as e:
            print(f"Error analyzing {entity_name}: {e}")

if __name__ == "__main__":
    run_analysis()
