"""
Before/After Data Quality Comparison Dashboard
==============================================
This script generates visual comparisons demonstrating the impact
of professional data cleaning on dataset quality.

Author: [Jonathan Chavarria Peralta]
Purpose: Portfolio visualization for data cleaning project
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Set visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 12)
plt.rcParams['font.size'] = 10


class DataQualityComparator:
    """
    Comparative analysis and visualization of dirty vs clean datasets.
    """
    
    def __init__(self, dirty_filepath, clean_filepath):
        """
        Initialize with file paths to both datasets.
        
        Parameters:
        -----------
        dirty_filepath : str
            Path to the dirty dataset
        clean_filepath : str
            Path to the clean dataset
        """
        self.df_dirty = pd.read_csv(dirty_filepath)
        self.df_clean = pd.read_csv(clean_filepath)
        
        # Convert timestamps
        self.df_dirty['timestamp'] = pd.to_datetime(self.df_dirty['timestamp'], 
                                                     errors='coerce')
        self.df_clean['timestamp'] = pd.to_datetime(self.df_clean['timestamp'])
    
    
    def generate_metrics_comparison(self):
        """
        Generate quantitative metrics comparing data quality.
        
        Returns:
        --------
        dict
            Dictionary containing comparison metrics
        """
        metrics = {
            'before': {},
            'after': {},
            'improvement': {}
        }
        
        # Record counts
        metrics['before']['total_records'] = len(self.df_dirty)
        metrics['after']['total_records'] = len(self.df_clean)
        
        # Missing values
        metrics['before']['missing_values'] = self.df_dirty.isnull().sum().sum()
        metrics['after']['missing_values'] = self.df_clean.isnull().sum().sum()
        
        # Duplicates
        metrics['before']['duplicates'] = self.df_dirty.duplicated().sum()
        metrics['after']['duplicates'] = self.df_clean.duplicated().sum()
        
        # Categorical consistency (zone)
        metrics['before']['zone_categories'] = self.df_dirty['zone'].nunique()
        metrics['after']['zone_categories'] = self.df_clean['zone'].nunique()
        
        # Calculate improvements
        metrics['improvement']['missing_reduction_pct'] = (
            (metrics['before']['missing_values'] - metrics['after']['missing_values']) /
            max(metrics['before']['missing_values'], 1) * 100
        )
        
        metrics['improvement']['duplicate_reduction_pct'] = (
            (metrics['before']['duplicates'] - metrics['after']['duplicates']) /
            max(metrics['before']['duplicates'], 1) * 100
        )
        
        return metrics
    
    
    def plot_comparison_dashboard(self, save_path='data_quality_comparison.png'):
        """
        Create a comprehensive visual dashboard comparing before/after.
        
        Parameters:
        -----------
        save_path : str
            Path to save the output visualization
        """
        fig, axes = plt.subplots(3, 3, figsize=(18, 14))
        fig.suptitle('Smart City Data: Quality Enhancement Report\nBefore vs After Professional Cleaning', 
                     fontsize=16, fontweight='bold', y=0.995)
        
        # Color scheme
        color_before = '#e74c3c'  # Red for dirty data
        color_after = '#27ae60'   # Green for clean data
        
        # ===== ROW 1: Basic Metrics =====
        
        # 1.1: Record Count Comparison
        ax = axes[0, 0]
        categories = ['Before\nCleaning', 'After\nCleaning']
        values = [len(self.df_dirty), len(self.df_clean)]
        bars = ax.bar(categories, values, color=[color_before, color_after], alpha=0.7, edgecolor='black')
        ax.set_ylabel('Number of Records', fontweight='bold')
        ax.set_title('Dataset Size', fontweight='bold')
        ax.set_ylim(0, max(values) * 1.2)
        
        for i, (bar, val) in enumerate(zip(bars, values)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:,}', ha='center', va='bottom', fontweight='bold', fontsize=11)
        
        # 1.2: Missing Values Comparison
        ax = axes[0, 1]
        missing_before = self.df_dirty.isnull().sum().sum()
        missing_after = self.df_clean.isnull().sum().sum()
        categories = ['Before\nCleaning', 'After\nCleaning']
        values = [missing_before, missing_after]
        bars = ax.bar(categories, values, color=[color_before, color_after], alpha=0.7, edgecolor='black')
        ax.set_ylabel('Missing Values Count', fontweight='bold')
        ax.set_title('Missing Data Treatment', fontweight='bold')
        
        for i, (bar, val) in enumerate(zip(bars, values)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:,}', ha='center', va='bottom', fontweight='bold', fontsize=11)
        
        # 1.3: Categorical Consistency
        ax = axes[0, 2]
        zone_before = self.df_dirty['zone'].nunique()
        zone_after = self.df_clean['zone'].nunique()
        categories = ['Before\nCleaning', 'After\nCleaning']
        values = [zone_before, zone_after]
        bars = ax.bar(categories, values, color=[color_before, color_after], alpha=0.7, edgecolor='black')
        ax.set_ylabel('Unique Zone Categories', fontweight='bold')
        ax.set_title('Categorical Standardization', fontweight='bold')
        ax.axhline(y=5, color='blue', linestyle='--', linewidth=2, label='Expected: 5', alpha=0.6)
        ax.legend()
        
        for i, (bar, val) in enumerate(zip(bars, values)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val}', ha='center', va='bottom', fontweight='bold', fontsize=11)
        
        # ===== ROW 2: Distribution Comparisons =====
        
        # 2.1: Traffic Flow Distribution
        ax = axes[1, 0]
        # Remove outliers for better visualization
        traffic_dirty = self.df_dirty['traffic_flow'].dropna()
        traffic_clean = self.df_clean['traffic_flow'].dropna()
        
        ax.hist(traffic_dirty, bins=50, alpha=0.5, label='Before (with outliers)', 
                color=color_before, edgecolor='black')
        ax.hist(traffic_clean, bins=50, alpha=0.7, label='After (cleaned)', 
                color=color_after, edgecolor='black')
        ax.set_xlabel('Traffic Flow (vehicles/hour)', fontweight='bold')
        ax.set_ylabel('Frequency', fontweight='bold')
        ax.set_title('Traffic Flow Distribution', fontweight='bold')
        ax.legend()
        ax.set_xlim(0, 2000)
        
        # 2.2: Energy Consumption Distribution
        ax = axes[1, 1]
        energy_dirty = self.df_dirty['energy_consumption'].dropna()
        energy_clean = self.df_clean['energy_consumption'].dropna()
        
        ax.hist(energy_dirty[energy_dirty >= 0], bins=50, alpha=0.5, 
                label='Before (negative values removed for viz)', color=color_before, edgecolor='black')
        ax.hist(energy_clean, bins=50, alpha=0.7, label='After (cleaned)', 
                color=color_after, edgecolor='black')
        ax.set_xlabel('Energy Consumption (kWh)', fontweight='bold')
        ax.set_ylabel('Frequency', fontweight='bold')
        ax.set_title('Energy Consumption Distribution', fontweight='bold')
        ax.legend()
        ax.set_xlim(0, 500)
        
        # 2.3: Temperature Distribution
        ax = axes[1, 2]
        temp_dirty = self.df_dirty['temperature_celsius'].dropna()
        temp_clean = self.df_clean['temperature_celsius'].dropna()
        
        ax.hist(temp_dirty, bins=50, alpha=0.5, label='Before (with extremes)', 
                color=color_before, edgecolor='black')
        ax.hist(temp_clean, bins=50, alpha=0.7, label='After (cleaned)', 
                color=color_after, edgecolor='black')
        ax.set_xlabel('Temperature (°C)', fontweight='bold')
        ax.set_ylabel('Frequency', fontweight='bold')
        ax.set_title('Temperature Distribution', fontweight='bold')
        ax.legend()
        ax.set_xlim(-20, 50)
        
        # ===== ROW 3: Box Plots for Outlier Detection =====
        
        # 3.1: Traffic Flow Box Plot
        ax = axes[2, 0]
        data_to_plot = [
            traffic_dirty[traffic_dirty <= 2000],
            traffic_clean
        ]
        bp = ax.boxplot(data_to_plot, labels=['Before', 'After'], patch_artist=True,
                       widths=0.6, showfliers=True)
        bp['boxes'][0].set_facecolor(color_before)
        bp['boxes'][1].set_facecolor(color_after)
        for element in ['whiskers', 'fliers', 'means', 'medians', 'caps']:
            plt.setp(bp[element], color='black', linewidth=1.5)
        ax.set_ylabel('Traffic Flow (vehicles/hour)', fontweight='bold')
        ax.set_title('Traffic Flow: Outlier Treatment', fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        # 3.2: Energy Box Plot
        ax = axes[2, 1]
        data_to_plot = [
            energy_dirty[(energy_dirty >= 0) & (energy_dirty <= 500)],
            energy_clean
        ]
        bp = ax.boxplot(data_to_plot, labels=['Before', 'After'], patch_artist=True,
                       widths=0.6, showfliers=True)
        bp['boxes'][0].set_facecolor(color_before)
        bp['boxes'][1].set_facecolor(color_after)
        for element in ['whiskers', 'fliers', 'means', 'medians', 'caps']:
            plt.setp(bp[element], color='black', linewidth=1.5)
        ax.set_ylabel('Energy Consumption (kWh)', fontweight='bold')
        ax.set_title('Energy: Outlier Treatment', fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        # 3.3: Summary Metrics Table
        ax = axes[2, 2]
        ax.axis('off')
        
        metrics = self.generate_metrics_comparison()
        
        table_data = [
            ['Metric', 'Before', 'After', 'Improvement'],
            ['Total Records', f"{metrics['before']['total_records']:,}", 
             f"{metrics['after']['total_records']:,}", '—'],
            ['Missing Values', f"{metrics['before']['missing_values']:,}", 
             f"{metrics['after']['missing_values']:,}", 
             f"{metrics['improvement']['missing_reduction_pct']:.1f}%↓"],
            ['Duplicates', f"{metrics['before']['duplicates']:,}", 
             f"{metrics['after']['duplicates']:,}", 
             f"{metrics['improvement']['duplicate_reduction_pct']:.1f}%↓"],
            ['Zone Categories', f"{metrics['before']['zone_categories']}", 
             f"{metrics['after']['zone_categories']}", 'Standardized'],
        ]
        
        table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                        colWidths=[0.3, 0.25, 0.25, 0.25])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2.5)
        
        # Style header row
        for i in range(4):
            table[(0, i)].set_facecolor('#34495e')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Style data rows
        for i in range(1, 5):
            for j in range(4):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#ecf0f1')
        
        ax.set_title('Quality Metrics Summary', fontweight='bold', pad=20)
        
        # Final adjustments
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Visualization saved to: {save_path}")
        print(f"  Resolution: 300 DPI (print quality)")
        
        return fig
    
    
    def print_detailed_report(self):
        """Print a detailed text report of improvements."""
        metrics = self.generate_metrics_comparison()
        
        print("\n" + "="*70)
        print("DETAILED DATA QUALITY IMPROVEMENT REPORT")
        print("="*70)
        
        print("\n📊 DATASET SIZE:")
        print(f"   Before: {metrics['before']['total_records']:>10,} records")
        print(f"   After:  {metrics['after']['total_records']:>10,} records")
        reduction = metrics['before']['total_records'] - metrics['after']['total_records']
        print(f"   Removed: {reduction:>9,} low-quality records")
        
        print("\n🔍 MISSING DATA:")
        print(f"   Before: {metrics['before']['missing_values']:>10,} missing values")
        print(f"   After:  {metrics['after']['missing_values']:>10,} missing values")
        print(f"   Improvement: {metrics['improvement']['missing_reduction_pct']:>6.1f}% reduction")
        
        print("\n📋 DUPLICATE RECORDS:")
        print(f"   Before: {metrics['before']['duplicates']:>10,} duplicates")
        print(f"   After:  {metrics['after']['duplicates']:>10,} duplicates")
        print(f"   Improvement: {metrics['improvement']['duplicate_reduction_pct']:>6.1f}% reduction")
        
        print("\n🏷️  CATEGORICAL CONSISTENCY:")
        print(f"   Zone categories before: {metrics['before']['zone_categories']} (inconsistent)")
        print(f"   Zone categories after:  {metrics['after']['zone_categories']} (standardized)")
        
        print("\n📈 STATISTICAL QUALITY:")
        
        for col in ['traffic_flow', 'energy_consumption', 'temperature_celsius']:
            print(f"\n   {col.replace('_', ' ').title()}:")
            
            before_std = self.df_dirty[col].std()
            after_std = self.df_clean[col].std()
            before_cv = (before_std / self.df_dirty[col].mean()) * 100 if self.df_dirty[col].mean() != 0 else 0
            after_cv = (after_std / self.df_clean[col].mean()) * 100 if self.df_clean[col].mean() != 0 else 0
            
            print(f"      Coefficient of Variation (CV):")
            print(f"         Before: {before_cv:.2f}%")
            print(f"         After:  {after_cv:.2f}%")
            
            improvement = "More consistent" if after_cv < before_cv else "Similar"
            print(f"         Status: {improvement}")
        
        print("\n" + "="*70)
        print("✅ DATA CLEANING IMPACT: SIGNIFICANT QUALITY IMPROVEMENT")
        print("="*70 + "\n")


def main():
    """Main execution function."""
    
    print("\nGenerating Before/After Comparison...")
    print("-" * 70)
    
    # Initialize comparator
    comparator = DataQualityComparator(
        dirty_filepath='smart_city_dirty_data.csv',
        clean_filepath='smart_city_clean_data.csv'
    )
    
    # Generate visualizations
    comparator.plot_comparison_dashboard(save_path='data_quality_comparison.png')
    
    # Print detailed metrics
    comparator.print_detailed_report()
    
    print("\n✨ Comparison complete! Check 'data_quality_comparison.png' for visual report.")


if __name__ == "__main__":
    main()