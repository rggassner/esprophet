#!venv/bin/python3
import pandas as pd
from prophet import Prophet
from elasticsearch import Elasticsearch
import urllib3
import matplotlib.pyplot as plt
import os
import json
from dotenv import load_dotenv

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
load_dotenv() 

MIN_DOCS = int(os.getenv("MIN_DOCS", 150000))
ES_HOST = os.getenv("ES_HOST")
ES_USER = os.getenv("ES_USER")
ES_PASS = os.getenv("ES_PASS")
INDEX_PATTERN = os.getenv("INDEX_PATTERN", "logs-waf-prod*")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./alerts_plots")
BUFFER_PERCENT = float(os.getenv("BUFFER_PERCENT", 1.20))
MIN_VOLUME_FLOOR = int(os.getenv("MIN_VOLUME_FLOOR", 50))
INTERVAL_WIDTH = float(os.getenv("INTERVAL_WIDTH", 0.99))
TIMESTAMP = os.getenv("TIMESTAMP_FIELD", "@timestamp")
GRAIN = os.getenv("GRAIN_FIELD", "dest_host")
ANALYSIS_START = os.getenv("ANALYSIS_START", "now-30d")
ANALYSIS_END = os.getenv("ANALYSIS_END", "now-1h")
FIXED_INTERVAL = os.getenv("FIXED_INTERVAL", "1h")
AGG_SIZE = os.getenv("AGG_SIZE", 200)
MINIMUM_SAMPLES = int(os.getenv("MINIMUM_SAMPLES", 168))
PLOT_UPPER = os.getenv("PLOT_UPPER_ALERTS", "True") == "True"
PLOT_LOWER = os.getenv("PLOT_LOWER_ALERTS", "False") == "True"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

es = Elasticsearch(
    [ES_HOST],
    basic_auth=(ES_USER, ES_PASS),
    verify_certs=False,
    ssl_show_warn=False
)

def load_templated_query():
    with open('query_template.json', 'r') as f:
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
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))

    plt.title(f"{entity_name} (Sensitivity: {BUFFER_PERCENT})")
    plt.xlabel("Time (UTC)")
    plt.ylabel("Counts per Hour")
    plt.legend(loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.7)

    filename = entity_name.replace(".", "_") + ".png"
    plt.savefig(os.path.join(OUTPUT_DIR, filename), bbox_inches='tight')
    plt.close()

def run_analysis():
    query = load_templated_query()
    res = es.search(index=INDEX_PATTERN, body=query)
    for bucket in res['aggregations']['target_grain']['buckets']:
        entity_name = bucket['key']

        df = pd.DataFrame([
            {'ds': b['key_as_string'], 'y': b['doc_count']}
            for b in bucket['timeseries']['buckets']
        ])

        df['ds'] = pd.to_datetime(df['ds']).dt.tz_localize(None)

        if len(df) < MINIMUM_SAMPLES:
            continue

        try:
            model = Prophet(interval_width=INTERVAL_WIDTH, daily_seasonality=True, weekly_seasonality=False)
            model.add_seasonality(name='weekly', period=7, fourier_order=10)

            train = df.iloc[:-24]
            model.fit(train)

            forecast = model.predict(df[['ds']])
            check = df.iloc[-24:].merge(forecast[['ds', 'yhat_upper', 'yhat_lower']], on='ds')

            # Use variables from config
            check['is_anomaly'] = (
                (check['y'] > (check['yhat_upper'] * BUFFER_PERCENT)) &
                (check['y'] > MIN_VOLUME_FLOOR)
            )

            check['is_drop'] = (check['y'] < check['yhat_lower']) & (check['yhat_lower'] > 10)

            if check['is_anomaly'].any() or check['is_drop'].any():
                print(f"Significant Alert for {entity_name}!")
                generate_plot(df, forecast, entity_name)

        except Exception as e:
            print(f"Error on {entity_name}: {e}")

if __name__ == "__main__":
    run_analysis()
