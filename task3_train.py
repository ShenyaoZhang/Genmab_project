#!/usr/bin/env python3
"""
Task 3: Train Isolation Forest Model and Analyze Results
This script trains the anomaly detection model and displays results.
"""

import pandas as pd
import numpy as np
import os
from task3_data_and_model import Task3DataProcessor, Task3Model, get_data_file_path


def save_results(results_data, X, pair_names, additional_metrics, output_dir):
    """
    Save results to CSV files with Top-K decoupling, per-drug caps, secondary sorting, and why_flagged.
    
    Args:
        results_data (dict): Results from model.get_results()
        X (numpy.ndarray): Original feature matrix
        pair_names (list): List of drug-event pair names
        additional_metrics (dict): Additional metrics (log_prr, ic, ic025) for each pair
        output_dir (str): Directory to save results
    """
    print("Saving results...")
    
    # Load configuration
    try:
        from config_task3 import CONFIG
        TOP_K_GLOBAL = CONFIG.get("top_k_global", 200)
        PER_DRUG_CAP = CONFIG.get("per_drug_cap", 5)
        SECONDARY_METRIC = CONFIG.get("secondary_metric", "ic025")
        WHY_RULES = CONFIG.get("why_rules", [])
    except ImportError:
        TOP_K_GLOBAL = 200
        PER_DRUG_CAP = 5
        SECONDARY_METRIC = "ic025"
        WHY_RULES = [
            ("log_prr", 1.0, "High PRR"),
            ("ic025", 0.0, "IC025 > 0"),
            ("chi2", 4.0, "High Chi-square"),
        ]
    
    predictions = results_data['predictions']
    anomaly_scores = results_data['anomaly_scores']
    
    # Create results DataFrame
    results_df = pd.DataFrame({
        'drug_event_pair': pair_names,
        'drug': [p.split('||')[0] for p in pair_names],
        'event': [p.split('||')[1] for p in pair_names],
        'is_anomaly': predictions == -1,
        'anomaly_score': -anomaly_scores,  # Negative scores, higher = more anomalous
        'count': X[:, 0],
        'prr': X[:, 1],
        'ror': X[:, 2],
        'chi2': X[:, 3],
        'serious_rate': X[:, 4],
        'death_rate': X[:, 5],
        'hosp_rate': X[:, 6],
        'life_threat_rate': X[:, 7],
        'disable_rate': X[:, 8]
    })
    
    # Add additional metrics (log_prr, ic, ic025)
    results_df['log_prr'] = [additional_metrics.get(pair, {}).get('log_prr', 0.0) for pair in pair_names]
    results_df['ic'] = [additional_metrics.get(pair, {}).get('ic', 0.0) for pair in pair_names]
    results_df['ic025'] = [additional_metrics.get(pair, {}).get('ic025', 0.0) for pair in pair_names]
    
    # --- Top-K decoupling: Sort by anomaly_score first (descending) ---
    results_df = results_df.sort_values('anomaly_score', ascending=False)
    
    # --- Secondary sorting (optional; if column doesn't exist, skip) ---
    if SECONDARY_METRIC in results_df.columns:
        results_df = results_df.sort_values(
            by=['anomaly_score', SECONDARY_METRIC],
            ascending=[False, False]
        )
    
    # --- why_flagged generation (based on simple rule concatenation) ---
    def build_why(row):
        tags = []
        for col, thr, label in WHY_RULES:
            if col in row and pd.notna(row[col]):
                val = row[col]
                # Different columns can have different comparison logic
                # Here unified as "larger is better", IC025/threshold uses >thr
                if val > thr:
                    tags.append(label)
        return "、".join(tags) if tags else "High anomaly score"
    
    results_df['why_flagged'] = results_df.apply(build_why, axis=1)
    
    # --- Per-drug cap ---
    results_df['rank_by_drug'] = results_df.groupby('drug')['anomaly_score'].rank(method='first', ascending=False)
    results_df = results_df[results_df['rank_by_drug'] <= PER_DRUG_CAP]
    
    # --- Global Top-K ---
    results_df = results_df.head(TOP_K_GLOBAL)
    
    # --- Final sorting and rank ---
    results_df = results_df.sort_values(['anomaly_score'], ascending=False).reset_index(drop=True)
    results_df['rank'] = results_df.index + 1
    
    # Save all results
    output_file1 = os.path.join(output_dir, 'task3_ml_isolation_forest_results.csv')
    results_df.to_csv(output_file1, index=False)
    print(f"✓ Complete results saved: {output_file1}")
    print(f"✓ Top-K decoupling: {TOP_K_GLOBAL} global, {PER_DRUG_CAP} per drug")
    
    # Save only anomalies
    anomalies_df = results_df[results_df['is_anomaly'] == True]
    output_file2 = os.path.join(output_dir, 'task3_ml_anomalies_only.csv')
    anomalies_df.to_csv(output_file2, index=False)
    print(f"✓ Anomalies saved: {output_file2} ({len(anomalies_df)} anomalies)\n")
    
    return results_df, anomalies_df


def display_top_anomalies(anomalies_df, n=30):
    """
    Display top N anomalies.
    
    Args:
        anomalies_df (DataFrame): DataFrame containing anomalies
        n (int): Number of top anomalies to display
    """
    print("=" * 80)
    print(f"Top {n} Most Significant Anomalies (Isolation Forest)")
    print("=" * 80)
    print()
    
    for idx, (i, row) in enumerate(anomalies_df.head(n).iterrows(), 1):
        print(f"[{idx}] Anomaly Score: {row['anomaly_score']:.4f}")
        print(f"    Drug: {row['drug']}")
        print(f"    Adverse Event: {row['event']}")
        print(f"    Report Count: {int(row['count'])}")
        print(f"    PRR: {row['prr']:.3f} | ROR: {row['ror']:.3f} | χ²: {row['chi2']:.2f}")
        if 'ic025' in row:
            print(f"    IC025: {row['ic025']:.3f}")
        print(f"    Death Rate: {row['death_rate']*100:.1f}% | Hosp. Rate: {row['hosp_rate']*100:.1f}%")
        if 'why_flagged' in row:
            print(f"    Why Flagged: {row['why_flagged']}")
        print()


def display_drug_summary(anomalies_df, n=20):
    """
    Display anomaly summary by drug.
    
    Args:
        anomalies_df (DataFrame): DataFrame containing anomalies
        n (int): Number of top drugs to display
    """
    print("=" * 80)
    print(f"Anomaly Summary by Drug (Top {n})")
    print("=" * 80)
    print()
    
    drug_anomaly_counts = anomalies_df['drug'].value_counts()
    
    for idx, (drug, count) in enumerate(drug_anomaly_counts.head(n).items(), 1):
        drug_anomalies = anomalies_df[anomalies_df['drug'] == drug]
        max_score = drug_anomalies['anomaly_score'].max()
        top_events = drug_anomalies.nlargest(3, 'anomaly_score')['event'].tolist()
        
        print(f"{idx}. {drug}:")
        print(f"   Anomaly Signals: {count}")
        print(f"   Highest Anomaly Score: {max_score:.4f}")
        print(f"   Top Events: {', '.join(top_events)}")
        print()


def display_epcoritamab_analysis(anomalies_df):
    """
    Display specific analysis for Epcoritamab.
    
    Args:
        anomalies_df (DataFrame): DataFrame containing anomalies
    """
    epc_anomalies = anomalies_df[anomalies_df['drug'] == 'Epcoritamab']
    
    if len(epc_anomalies) > 0:
        print("=" * 80)
        print("Epcoritamab Anomaly Signal Analysis")
        print("=" * 80)
        print()
        print(f"Detected {len(epc_anomalies)} anomaly signals for Epcoritamab")
        print()
        print("Top 10 Epcoritamab Anomalous Events:")
        for idx, (i, row) in enumerate(epc_anomalies.head(10).iterrows(), 1):
            print(f"  {idx}. {row['event']}")
            print(f"     Anomaly Score: {row['anomaly_score']:.4f}")
            print(f"     PRR: {row['prr']:.3f} | Death Rate: {row['death_rate']*100:.1f}%")
        print()


def main():
    """Main function to run the complete training and analysis pipeline."""
    
    print("=" * 80)
    print("Task 3: Isolation Forest Anomaly Detection")
    print("=" * 80)
    print()
    
    # Get data file path
    data_file = get_data_file_path()
    
    # Process data
    processor = Task3DataProcessor(data_file)
    X, pair_names, additional_metrics = processor.process()
    
    # Load configuration
    try:
        from config_task3 import CONFIG
        contamination = CONFIG.get("contamination", 0.15)
        random_state = CONFIG.get("random_state", 42)
        n_estimators = CONFIG.get("n_estimators", 100)
    except ImportError:
        contamination = 0.15
        random_state = 42
        n_estimators = 100
    
    # Train model
    model = Task3Model(contamination=contamination, random_state=random_state, n_estimators=n_estimators)
    model.fit(X)
    
    # Get results
    results = model.get_results()
    
    print("Analyzing results...")
    print(f"✓ Detected anomalies: {results['n_anomalies']} ({results['anomaly_rate']*100:.1f}%)")
    print(f"✓ Normal samples: {results['n_normal']} ({(1-results['anomaly_rate'])*100:.1f}%)\n")
    
    # Save results
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    output_dir = os.path.join(parent_dir, 'data')
    artifacts_dir = os.path.join(parent_dir, 'artifacts')
    
    # Save model and scaler
    model.save_model(artifacts_dir)
    
    # Save results with Top-K decoupling
    results_df, anomalies_df = save_results(results, X, pair_names, additional_metrics, output_dir)
    
    # Quality control metrics
    print("=" * 80)
    print("Quality Control Metrics")
    print("=" * 80)
    print(f"QC> Sample count: {len(results_df)}")
    print(f"QC> Unique drugs: {results_df['drug'].nunique()}")
    print(f"QC> Median events per drug: {results_df.groupby('drug')['event'].nunique().median():.1f}")
    print(f"QC> PRR > 2 proportion: {(results_df['prr'] > 2).mean()*100:.1f}%")
    print(f"QC> Anomaly score P99: {results_df['anomaly_score'].quantile(0.99):.4f}")
    if hasattr(processor, 'n_excluded_outcomes'):
        print(f"QC> Excluded outcome PTs: {processor.n_excluded_outcomes}")
    print()
    
    # Display analysis
    display_top_anomalies(anomalies_df, n=30)
    display_drug_summary(anomalies_df, n=20)
    display_epcoritamab_analysis(anomalies_df)
    
    print("=" * 80)
    print("Task 3 Complete! Isolation Forest ML model trained successfully.")
    print("=" * 80)


if __name__ == "__main__":
    main()

