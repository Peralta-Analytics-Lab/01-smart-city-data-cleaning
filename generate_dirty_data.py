"""
Smart City Dataset Generator - Dirty Version
============================================
This script generates a synthetic dataset simulating IoT sensors 
from a Smart City infrastructure with intentionally embedded data quality issues.

Author: [Jonathan Chavarria Peralta]
Purpose: Portfolio demonstration of data cleaning capabilities
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

def generate_smart_city_data(n_records=10000):
    """
    Generate synthetic Smart City sensor data with embedded quality issues.
    
    Parameters:
    -----------
    n_records : int
        Number of records to generate
        
    Returns:
    --------
    pd.DataFrame
        DataFrame containing dirty sensor data
    """
    
    # Base timestamp generation
    start_date = datetime(2024, 1, 1)
    timestamps = [start_date + timedelta(hours=i) for i in range(n_records)]
    
    # Sensor ID generation
    sensor_zones = ['Downtown', 'Industrial_Park', 'Residential_North', 
                    'Commercial_District', 'University_Area']
    sensor_ids = [f"SNS_{random.choice(sensor_zones)}_{random.randint(1000, 9999)}" 
              for _ in range(n_records)]
    
    # Traffic flow (vehicles per hour)
    traffic_flow = np.random.normal(loc=450, scale=120, size=n_records)
    traffic_flow = np.abs(traffic_flow).astype(int)
    
    # Energy consumption (kWh)
    energy_consumption = np.random.gamma(shape=2, scale=50, size=n_records)
    
    # Air quality index (0-500 scale)
    air_quality = np.random.normal(loc=85, scale=25, size=n_records)
    air_quality = np.clip(air_quality, 0, 500)
    
    # Temperature (Celsius)
    temperature = np.random.normal(loc=22, scale=5, size=n_records)
    
    # Sensor status
    sensor_status = np.random.choice(['Active', 'Maintenance', 'Error'], 
                                     size=n_records, p=[0.85, 0.10, 0.05])
    
    # Create initial clean dataframe
    df = pd.DataFrame({
        'timestamp': timestamps,
        'sensor_id': sensor_ids,
        'zone': [sid.split('_')[1] for sid in sensor_ids],
        'traffic_flow': traffic_flow,
        'energy_consumption': energy_consumption,
        'air_quality_index': air_quality,
        'temperature_celsius': temperature,
        'sensor_status': sensor_status
    })
    
    return df


def introduce_data_quality_issues(df):
    """
    Systematically introduce realistic data quality problems.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Clean dataframe
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with embedded quality issues
    """
    
    df_dirty = df.copy()
    n_records = len(df_dirty)
    
    # === 1. MISSING VALUES (Non-random patterns) ===
    # Sensors in maintenance mode have missing readings (realistic pattern)
    maintenance_mask = df_dirty['sensor_status'] == 'Maintenance'
    df_dirty.loc[maintenance_mask, 'traffic_flow'] = np.nan
    df_dirty.loc[maintenance_mask, 'energy_consumption'] = np.nan
    
    # Random missing values in air quality (sensor malfunction)
    random_missing_aq = np.random.choice(df_dirty.index, size=int(n_records * 0.08))
    df_dirty.loc[random_missing_aq, 'air_quality_index'] = np.nan
    
    # Temperature sensor failures (clustered missing values)
    failure_start = random.randint(1000, 5000)
    df_dirty.loc[failure_start:failure_start+200, 'temperature_celsius'] = np.nan
    
    # === 2. OUTLIERS (Statistical anomalies) ===
    # Extreme traffic values due to sensor calibration errors
    outlier_indices = np.random.choice(df_dirty.index, size=int(n_records * 0.03))
    df_dirty.loc[outlier_indices, 'traffic_flow'] = np.random.uniform(3000, 5000, 
                                                                       size=len(outlier_indices))
    
    # Negative energy consumption (sensor malfunction)
    negative_indices = np.random.choice(df_dirty.index, size=int(n_records * 0.02))
    df_dirty.loc[negative_indices, 'energy_consumption'] *= -1
    
    # Impossible temperature readings
    extreme_temp_indices = np.random.choice(df_dirty.index, size=int(n_records * 0.015))
    df_dirty.loc[extreme_temp_indices, 'temperature_celsius'] = np.random.uniform(-50, 80, 
                                                                                   size=len(extreme_temp_indices))
    
    # === 3. DUPLICATE RECORDS ===
    # Simulate duplicate transmissions from sensors
    duplicate_indices = np.random.choice(df_dirty.index, size=int(n_records * 0.05))
    duplicates = df_dirty.loc[duplicate_indices].copy()
    df_dirty = pd.concat([df_dirty, duplicates], ignore_index=True)
    
    # === 4. INCONSISTENT DATE FORMATS ===
    # Convert some timestamps to string with different formats
    timestamp_col = df_dirty['timestamp'].astype(str).tolist()
    
    for i in random.sample(range(len(timestamp_col)), int(len(timestamp_col) * 0.15)):
        dt = pd.to_datetime(timestamp_col[i])
        # Different format variations
        formats = [
            dt.strftime('%m/%d/%Y %H:%M:%S'),
            dt.strftime('%d-%m-%Y %H:%M'),
            dt.strftime('%Y%m%d_%H%M%S'),
        ]
        timestamp_col[i] = random.choice(formats)
    
    df_dirty['timestamp'] = timestamp_col
    
    # === 5. CATEGORICAL TYPOS AND INCONSISTENCIES ===
    # Introduce typos in zone names
    zone_typos = {
        'Downtown': ['downtown', 'Down town', 'DOWNTOWN', 'Donwtown'],
        'Industrial_Park': ['Industrial Park', 'industrial_park', 'IndustrialPark'],
        'Residential_North': ['Residential North', 'ResidentialNorth', 'residential_north'],
        'Commercial_District': ['Commercial District', 'CommercialDistrict', 'commercial_district'],
        'University_Area': ['University Area', 'UniversityArea', 'university_area']
    }
    
    for i in range(len(df_dirty)):
        if random.random() < 0.12:  # 12% chance of typo
            current_zone = df_dirty.loc[i, 'zone']
            if current_zone in zone_typos:
                df_dirty.loc[i, 'zone'] = random.choice(zone_typos[current_zone])
    
    # Sensor status inconsistencies
    status_variations = {
        'Active': ['active', 'ACTIVE', 'Active ', ' Active'],
        'Maintenance': ['maintenance', 'MAINTENANCE', 'Maint', 'Under Maintenance'],
        'Error': ['error', 'ERROR', 'Err', 'Failure']
    }
    
    for i in range(len(df_dirty)):
        if random.random() < 0.10:
            current_status = df_dirty.loc[i, 'sensor_status']
            if current_status in status_variations:
                df_dirty.loc[i, 'sensor_status'] = random.choice(status_variations[current_status])
    
    return df_dirty


def save_dataset(df, filename='smart_city_dirty_data.csv'):
    """
    Save the dirty dataset to CSV file.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe to save
    filename : str
        Output filename
    """
    df.to_csv(filename, index=False)
    print(f"✓ Dataset saved to '{filename}'")
    print(f"  Total records: {len(df):,}")
    print(f"  Columns: {list(df.columns)}")


def generate_data_quality_report(df):
    """
    Generate a summary report of data quality issues.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dirty dataframe
    """
    print("\n" + "="*60)
    print("DATA QUALITY ISSUES REPORT (BEFORE CLEANING)")
    print("="*60)
    
    # Missing values
    print("\n1. MISSING VALUES:")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    for col in missing[missing > 0].index:
        print(f"   • {col}: {missing[col]:,} ({missing_pct[col]}%)")
    
    # Duplicates
    n_duplicates = df.duplicated().sum()
    print(f"\n2. DUPLICATE RECORDS: {n_duplicates:,}")
    
    # Data types
    print("\n3. DATA TYPE INCONSISTENCIES:")
    print(f"   • timestamp: {df['timestamp'].dtype} (should be datetime)")
    
    # Categorical inconsistencies
    print("\n4. CATEGORICAL VALUE VARIATIONS:")
    print(f"   • zone unique values: {df['zone'].nunique()} (expected: 5)")
    print(f"   • sensor_status unique values: {df['sensor_status'].nunique()} (expected: 3)")
    
    # Statistical outliers (simplified check)
    print("\n5. POTENTIAL OUTLIERS:")
    for col in ['traffic_flow', 'energy_consumption', 'temperature_celsius']:
        if df[col].dtype in ['int64', 'float64']:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            outliers = ((df[col] < (q1 - 3*iqr)) | (df[col] > (q3 + 3*iqr))).sum()
            print(f"   • {col}: ~{outliers:,} potential outliers")
    
    print("\n" + "="*60 + "\n")


def main():
    """Main execution function."""
    
    print("\nGenerating Smart City IoT Dataset...")
    print("-" * 60)
    
    # Generate clean data
    df_clean = generate_smart_city_data(n_records=10000)
    print(f"✓ Generated {len(df_clean):,} clean records")
    
    # Introduce quality issues
    df_dirty = introduce_data_quality_issues(df_clean)
    print(f"✓ Introduced realistic data quality issues")
    
    # Save dataset
    save_dataset(df_dirty)
    
    # Generate report
    generate_data_quality_report(df_dirty)
    
    print("Dataset generation complete! Ready for cleaning demonstration.")


if __name__ == "__main__":
    main()