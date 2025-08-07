import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from datetime import datetime
import argparse

class QoEFileCalculator:
    """
    QoE Total Calculator that reads from input files
    Calculates: QoE_Total = η(t)·QoE_6G + (1-η(t))·QoE_5G
    """
    
    def __init__(self, config_file=None):
        """Initialize calculator with optional configuration file"""
        self.config = self._load_config(config_file)
        self.results = []
        
    def _load_config(self, config_file):
        """Load configuration from file or use defaults"""
        default_config = {
            "transition_months": 36,
            "eta_method": "adaptive",
            "network_conditions": {
                "Poor": 0.6,
                "Fair": 0.7,
                "Good": 0.8,
                "Excellent": 0.95
            },
            "output_format": "csv",
            "generate_plots": True
        }
        
        if config_file and Path(config_file).exists():
            with open(config_file, 'r') as f:
                user_config = json.load(f)
                default_config.update(user_config)
                
        return default_config
    
    def read_6g_components(self, 
                           impairment_file: str,
                           ahp_ai_file: str,
                           app_specific_file: str,
                           context_aware_file: str) -> pd.DataFrame:
        """
        Read 6G component data from separate CSV files
        Returns consolidated DataFrame
        """
        print("Reading 6G component files...")
        
        # Read impairment data
        imp_df = pd.read_csv(impairment_file)
        # Use R_Total_adaptive as the impairment score
        imp_df['QoE_Impairment'] = imp_df['R_Total_adaptive'] / 100
        
        # Read AHP AI data
        ahp_df = pd.read_csv(ahp_ai_file)
        ahp_df['QoE_AHP_AI'] = ahp_df['QoE_AHP_AI'] / 100
        
        # Read Application Specific data
        app_df = pd.read_csv(app_specific_file)
        # Already normalized in the dataset
        
        # Read Context Aware data
        ctx_df = pd.read_csv(context_aware_file)
        # Already normalized in the dataset
        
        # Merge all dataframes
        df_6g = pd.DataFrame({
            'Session_ID': imp_df['Session_ID'],
            'Time': imp_df['Time'],
            'QoE_Impairment_Enhanced': imp_df['QoE_Impairment'],
            'QoE_AHP_AI': ahp_df['QoE_AHP_AI'],
            'QoE_Application_Specific': app_df['QoE_Application_Specific'],
            'QoE_Context_Aware': ctx_df['QoE_Context_Aware'],
            'Network_State': ahp_df.get('Network_State', 'Good')
        })
        
        print(f"Loaded {len(df_6g)} sessions for 6G calculation")
        return df_6g
    
    def read_5g_baseline(self, baseline_file: str) -> pd.DataFrame:
        """
        Read 5G baseline data from CSV file
        """
        print("Reading 5G baseline file...")
        
        df_5g = pd.read_csv(baseline_file)
        
        # Ensure QoE_5G_Normalized column exists
        if 'QoE_5G_Normalized' not in df_5g.columns:
            # If only percentage is available, convert to normalized
            if 'QoE_5G_Percent' in df_5g.columns:
                df_5g['QoE_5G_Normalized'] = df_5g['QoE_5G_Percent'] / 100
            else:
                # Calculate from components if needed
                df_5g['QoE_5G_Normalized'] = df_5g['QoE_5G_Components']
        
        print(f"Loaded {len(df_5g)} sessions for 5G baseline")
        return df_5g
    
    def calculate_dynamic_weights(self, network_state: str, time_month: int) -> dict:
        """Calculate dynamic weights α, β, γ, δ based on network state and time"""
        progress = time_month / self.config['transition_months']
        
        if network_state == 'Excellent':
            weights = {
                'alpha': 0.15 - 0.05 * progress,
                'beta': 0.25 + 0.05 * progress,
                'gamma': 0.35 + 0.05 * progress,
                'delta': 0.25 - 0.05 * progress
            }
        elif network_state == 'Good':
            weights = {
                'alpha': 0.25,
                'beta': 0.25,
                'gamma': 0.30,
                'delta': 0.20
            }
        else:  # Fair or Poor
            weights = {
                'alpha': 0.35,
                'beta': 0.20,
                'gamma': 0.25,
                'delta': 0.20
            }
        
        # Normalize to ensure sum = 1
        total = sum(weights.values())
        return {k: v/total for k, v in weights.items()}
    
    def calculate_qoe_6g(self, row: pd.Series) -> tuple:
        """Calculate 6G QoE for a single session"""
        # Get dynamic weights
        weights = self.calculate_dynamic_weights(row['Network_State'], row['Time'])
        
        # Calculate weighted sum
        qoe_6g = (weights['alpha'] * row['QoE_Impairment_Enhanced'] +
                  weights['beta'] * row['QoE_AHP_AI'] +
                  weights['gamma'] * row['QoE_Application_Specific'] +
                  weights['delta'] * row['QoE_Context_Aware'])
        
        return qoe_6g, weights
    
    def calculate_eta(self, time_month: int, method: str = None) -> float:
        """Calculate transition factor η(t)"""
        if method is None:
            method = self.config['eta_method']
            
        t = time_month
        T = self.config['transition_months']
        
        if method == 'linear':
            return t / T
            
        elif method == 'sigmoid':
            return 1 / (1 + np.exp(-0.3 * (t - T/2)))
            
        elif method == 'adaptive':
            # Realistic deployment curve
            if t < 6:
                return 0.005 * t
            elif t < 12:
                return 0.03 + 0.04 * (t - 6)
            elif t < 18:
                return 0.27 + 0.07 * (t - 12)
            elif t < 24:
                return 0.69 + 0.04 * (t - 18)
            elif t < 30:
                return 0.93 + 0.01 * (t - 24)
            else:
                return 0.99 + 0.001 * (t - 30)
                
        else:
            raise ValueError(f"Unknown eta method: {method}")
    
    def calculate_from_files(self,
                           impairment_file: str,
                           ahp_ai_file: str,
                           app_specific_file: str,
                           context_aware_file: str,
                           baseline_5g_file: str,
                           output_file: str = None) -> pd.DataFrame:
        """
        Main calculation function that processes all files
        """
        print("="*60)
        print("QoE Total Calculator - File Processing")
        print("="*60)
        
        # Read all data files
        df_6g = self.read_6g_components(
            impairment_file, ahp_ai_file, 
            app_specific_file, context_aware_file
        )
        df_5g = self.read_5g_baseline(baseline_5g_file)
        
        # Ensure both dataframes have same length and order
        if len(df_6g) != len(df_5g):
            print(f"Warning: Different number of sessions in 6G ({len(df_6g)}) and 5G ({len(df_5g)}) data")
            min_len = min(len(df_6g), len(df_5g))
            df_6g = df_6g.iloc[:min_len]
            df_5g = df_5g.iloc[:min_len]
        
        # Calculate QoE Total for each session
        results = []
        
        for idx, (_, row_6g) in enumerate(df_6g.iterrows()):
            row_5g = df_5g.iloc[idx]
            
            # Calculate 6G QoE
            qoe_6g, weights = self.calculate_qoe_6g(row_6g)
            
            # Get 5G QoE
            qoe_5g = row_5g['QoE_5G_Normalized']
            
            # Calculate eta for all three methods
            time_month = row_6g['Time']
            eta_linear = self.calculate_eta(time_month, 'linear')
            eta_sigmoid = self.calculate_eta(time_month, 'sigmoid')
            eta_adaptive = self.calculate_eta(time_month, 'adaptive')
            
            # Calculate QoE Total for each method
            qoe_total_linear = eta_linear * qoe_6g + (1 - eta_linear) * qoe_5g
            qoe_total_sigmoid = eta_sigmoid * qoe_6g + (1 - eta_sigmoid) * qoe_5g
            qoe_total_adaptive = eta_adaptive * qoe_6g + (1 - eta_adaptive) * qoe_5g
            
            # Determine transition stage
            if eta_adaptive < 0.2:
                stage = "5G_Dominant"
            elif eta_adaptive < 0.5:
                stage = "Early_Transition"
            elif eta_adaptive < 0.8:
                stage = "6G_Emerging"
            elif eta_adaptive < 0.95:
                stage = "6G_Dominant"
            else:
                stage = "Complete_6G"
            
            # Store results
            result = {
                'Session_ID': row_6g['Session_ID'],
                'Time_Month': time_month,
                'Network_State': row_6g['Network_State'],
                
                # Component scores
                'QoE_Impairment': row_6g['QoE_Impairment_Enhanced'],
                'QoE_AHP_AI': row_6g['QoE_AHP_AI'],
                'QoE_Application': row_6g['QoE_Application_Specific'],
                'QoE_Context': row_6g['QoE_Context_Aware'],
                
                # Dynamic weights
                'Weight_Alpha': weights['alpha'],
                'Weight_Beta': weights['beta'],
                'Weight_Gamma': weights['gamma'],
                'Weight_Delta': weights['delta'],
                
                # QoE scores
                'QoE_5G': qoe_5g,
                'QoE_6G': qoe_6g,
                
                # Eta values
                'Eta_Linear': eta_linear,
                'Eta_Sigmoid': eta_sigmoid,
                'Eta_Adaptive': eta_adaptive,
                
                # QoE Total values
                'QoE_Total_Linear': qoe_total_linear,
                'QoE_Total_Sigmoid': qoe_total_sigmoid,
                'QoE_Total_Adaptive': qoe_total_adaptive,
                
                # Percentages
                'QoE_Total_Linear_Percent': qoe_total_linear * 100,
                'QoE_Total_Sigmoid_Percent': qoe_total_sigmoid * 100,
                'QoE_Total_Adaptive_Percent': qoe_total_adaptive * 100,
                
                'Transition_Stage': stage
            }
            
            results.append(result)
        
        # Convert to DataFrame
        self.results_df = pd.DataFrame(results)
        
        # Save results if output file specified
        if output_file:
            self.save_results(output_file)
        
        # Generate plots if configured
        if self.config['generate_plots']:
            self.generate_visualizations()
        
        # Print summary
        self.print_summary()
        
        return self.results_df
    
    def save_results(self, output_file: str):
        """Save results to file"""
        output_path = Path(output_file)
        
        if self.config['output_format'] == 'csv':
            self.results_df.to_csv(output_path, index=False)
            print(f"\nResults saved to: {output_path}")
            
        elif self.config['output_format'] == 'excel':
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                self.results_df.to_excel(writer, sheet_name='QoE_Total', index=False)
                
                # Add summary sheet
                summary_df = self.generate_summary_stats()
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
            print(f"\nResults saved to: {output_path}")
            
        elif self.config['output_format'] == 'json':
            self.results_df.to_json(output_path, orient='records', indent=2)
            print(f"\nResults saved to: {output_path}")
    
    def generate_summary_stats(self) -> pd.DataFrame:
        """Generate summary statistics"""
        summary = {
            'Metric': [
                'Initial QoE Total (%)',
                'Final QoE Total (%)',
                'Total Improvement (%)',
                'Max QoE Achieved (%)',
                'Month 90% Exceeded',
                'Month 100% Exceeded',
                'Average 6G QoE',
                'Average 5G QoE',
                'Component Weight Shift (α)',
                'Component Weight Shift (γ)'
            ],
            'Linear': [
                self.results_df.iloc[0]['QoE_Total_Linear_Percent'],
                self.results_df.iloc[-1]['QoE_Total_Linear_Percent'],
                self.results_df.iloc[-1]['QoE_Total_Linear_Percent'] - self.results_df.iloc[0]['QoE_Total_Linear_Percent'],
                self.results_df['QoE_Total_Linear_Percent'].max(),
                self.results_df[self.results_df['QoE_Total_Linear_Percent'] > 90]['Time_Month'].min(),
                self.results_df[self.results_df['QoE_Total_Linear_Percent'] > 100]['Time_Month'].min() if any(self.results_df['QoE_Total_Linear_Percent'] > 100) else 'N/A',
                self.results_df['QoE_6G'].mean(),
                self.results_df['QoE_5G'].mean(),
                f"{self.results_df.iloc[0]['Weight_Alpha']:.3f} → {self.results_df.iloc[-1]['Weight_Alpha']:.3f}",
                f"{self.results_df.iloc[0]['Weight_Gamma']:.3f} → {self.results_df.iloc[-1]['Weight_Gamma']:.3f}"
            ],
            'Sigmoid': [
                self.results_df.iloc[0]['QoE_Total_Sigmoid_Percent'],
                self.results_df.iloc[-1]['QoE_Total_Sigmoid_Percent'],
                self.results_df.iloc[-1]['QoE_Total_Sigmoid_Percent'] - self.results_df.iloc[0]['QoE_Total_Sigmoid_Percent'],
                self.results_df['QoE_Total_Sigmoid_Percent'].max(),
                self.results_df[self.results_df['QoE_Total_Sigmoid_Percent'] > 90]['Time_Month'].min(),
                self.results_df[self.results_df['QoE_Total_Sigmoid_Percent'] > 100]['Time_Month'].min() if any(self.results_df['QoE_Total_Sigmoid_Percent'] > 100) else 'N/A',
                '-', '-', '-', '-'
            ],
            'Adaptive': [
                self.results_df.iloc[0]['QoE_Total_Adaptive_Percent'],
                self.results_df.iloc[-1]['QoE_Total_Adaptive_Percent'],
                self.results_df.iloc[-1]['QoE_Total_Adaptive_Percent'] - self.results_df.iloc[0]['QoE_Total_Adaptive_Percent'],
                self.results_df['QoE_Total_Adaptive_Percent'].max(),
                self.results_df[self.results_df['QoE_Total_Adaptive_Percent'] > 90]['Time_Month'].min(),
                self.results_df[self.results_df['QoE_Total_Adaptive_Percent'] > 100]['Time_Month'].min() if any(self.results_df['QoE_Total_Adaptive_Percent'] > 100) else 'N/A',
                '-', '-', '-', '-'
            ]
        }
        
        return pd.DataFrame(summary)
    
    def generate_visualizations(self):
        """Generate visualization plots"""
        plt.style.use('seaborn-v0_8-darkgrid')
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('QoE Total Calculation Results', fontsize=16)
        
        # Plot 1: QoE Evolution
        ax1 = axes[0, 0]
        ax1.plot(self.results_df['Time_Month'], self.results_df['QoE_5G'] * 100, 
                'b-', label='QoE 5G', linewidth=2)
        ax1.plot(self.results_df['Time_Month'], self.results_df['QoE_6G'] * 100, 
                'r-', label='QoE 6G', linewidth=2)
        ax1.plot(self.results_df['Time_Month'], self.results_df['QoE_Total_Adaptive_Percent'], 
                'g-', label='QoE Total (Adaptive)', linewidth=3)
        ax1.axhline(y=100, color='k', linestyle='--', alpha=0.5)
        ax1.set_xlabel('Time (months)')
        ax1.set_ylabel('QoE (%)')
        ax1.set_title('QoE Evolution Over Time')
        ax1.legend()
        ax1.set_ylim(60, 110)
        
        # Plot 2: Component Contributions
        ax2 = axes[0, 1]
        components = ['QoE_Impairment', 'QoE_AHP_AI', 'QoE_Application', 'QoE_Context']
        for comp in components:
            ax2.plot(self.results_df['Time_Month'], self.results_df[comp], 
                    label=comp.replace('QoE_', ''), linewidth=2)
        ax2.set_xlabel('Time (months)')
        ax2.set_ylabel('Component Score')
        ax2.set_title('6G Component Evolution')
        ax2.legend()
        
        # Plot 3: Transition Functions
        ax3 = axes[1, 0]
        ax3.plot(self.results_df['Time_Month'], self.results_df['Eta_Linear'], 
                'b--', label='η Linear', linewidth=2)
        ax3.plot(self.results_df['Time_Month'], self.results_df['Eta_Sigmoid'], 
                'r-', label='η Sigmoid', linewidth=2)
        ax3.plot(self.results_df['Time_Month'], self.results_df['Eta_Adaptive'], 
                'g-', label='η Adaptive', linewidth=3)
        ax3.set_xlabel('Time (months)')
        ax3.set_ylabel('Transition Factor (η)')
        ax3.set_title('Transition Function Comparison')
        ax3.legend()
        ax3.set_ylim(0, 1.05)
        
        # Plot 4: Weight Evolution
        ax4 = axes[1, 1]
        ax4.plot(self.results_df['Time_Month'], self.results_df['Weight_Alpha'], 
                label='α (Impairment)', linewidth=2)
        ax4.plot(self.results_df['Time_Month'], self.results_df['Weight_Beta'], 
                label='β (AHP AI)', linewidth=2)
        ax4.plot(self.results_df['Time_Month'], self.results_df['Weight_Gamma'], 
                label='γ (Application)', linewidth=2)
        ax4.plot(self.results_df['Time_Month'], self.results_df['Weight_Delta'], 
                label='δ (Context)', linewidth=2)
        ax4.set_xlabel('Time (months)')
        ax4.set_ylabel('Weight Value')
        ax4.set_title('Dynamic Weight Evolution')
        ax4.legend()
        
        plt.tight_layout()
        
        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_file = f"qoe_total_analysis_{timestamp}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"\nVisualization saved to: {plot_file}")
        
        plt.show()
    
    def print_summary(self):
        """Print summary of results"""
        print("\n" + "="*60)
        print("QoE TOTAL CALCULATION SUMMARY")
        print("="*60)
        
        # Basic statistics
        print(f"\nTotal Sessions Processed: {len(self.results_df)}")
        print(f"Time Period: {self.results_df['Time_Month'].min()} - {self.results_df['Time_Month'].max()} months")
        
        # QoE Evolution (Adaptive)
        initial_qoe = self.results_df.iloc[0]['QoE_Total_Adaptive_Percent']
        final_qoe = self.results_df.iloc[-1]['QoE_Total_Adaptive_Percent']
        improvement = final_qoe - initial_qoe
        
        print(f"\nQoE Evolution (Adaptive Method):")
        print(f"  Initial QoE: {initial_qoe:.2f}%")
        print(f"  Final QoE: {final_qoe:.2f}%")
        print(f"  Total Improvement: {improvement:.2f}%")
        
        # Milestones
        print(f"\nKey Milestones:")
        milestones = [80, 90, 95, 100]
        for milestone in milestones:
            sessions = self.results_df[self.results_df['QoE_Total_Adaptive_Percent'] >= milestone]
            if len(sessions) > 0:
                month = sessions.iloc[0]['Time_Month']
                print(f"  {milestone}% QoE achieved: Month {month}")
            else:
                print(f"  {milestone}% QoE achieved: Not reached")
        
        # Component contributions at end
        print(f"\nFinal Component Weights:")
        final_row = self.results_df.iloc[-1]
        print(f"  α (Impairment): {final_row['Weight_Alpha']:.3f} ({final_row['Weight_Alpha']*100:.1f}%)")
        print(f"  β (AHP AI): {final_row['Weight_Beta']:.3f} ({final_row['Weight_Beta']*100:.1f}%)")
        print(f"  γ (Application): {final_row['Weight_Gamma']:.3f} ({final_row['Weight_Gamma']*100:.1f}%)")
        print(f"  δ (Context): {final_row['Weight_Delta']:.3f} ({final_row['Weight_Delta']*100:.1f}%)")
        
        # Transition stages
        print(f"\nTransition Stage Distribution:")
        stage_counts = self.results_df['Transition_Stage'].value_counts()
        for stage, count in stage_counts.items():
            print(f"  {stage}: {count} sessions ({count/len(self.results_df)*100:.1f}%)")
        
        print("\n" + "="*60)


# Example usage and main function
def main():
    """Main function to demonstrate file-based QoE calculation"""
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Calculate QoE Total from component files')
    parser.add_argument('--impairment', required=True, help='Path to impairment enhanced CSV file')
    parser.add_argument('--ahp', required=True, help='Path to AHP AI CSV file')
    parser.add_argument('--app', required=True, help='Path to application specific CSV file')
    parser.add_argument('--context', required=True, help='Path to context aware CSV file')
    parser.add_argument('--baseline', required=True, help='Path to 5G baseline CSV file')
    parser.add_argument('--output', default='qoe_total_results.csv', help='Output file path')
    parser.add_argument('--config', help='Configuration JSON file')
    parser.add_argument('--no-plots', action='store_true', help='Disable plot generation')
    
    args = parser.parse_args()
    
    # Create calculator instance
    calculator = QoEFileCalculator(config_file=args.config)
    
    # Disable plots if requested
    if args.no_plots:
        calculator.config['generate_plots'] = False
    
    # Process files and calculate QoE Total
    results = calculator.calculate_from_files(
        impairment_file=args.impairment,
        ahp_ai_file=args.ahp,
        app_specific_file=args.app,
        context_aware_file=args.context,
        baseline_5g_file=args.baseline,
        output_file=args.output
    )
    
    print("\nCalculation complete!")
    
    # Return results for further processing if needed
    return results


# Sample data generator for testing
def generate_sample_files():
    """Generate sample input files for testing"""
    
    # Create sample directory
    Path("sample_data").mkdir(exist_ok=True)
    
    # Generate sample impairment data
    imp_data = {
        'Session_ID': [f'S{i:03d}' for i in range(1, 61)],
        'Time': list(range(1, 37)) + list(range(1, 25)),
        'R_Total_adaptive': np.linspace(77, 121, 60)
    }
    pd.DataFrame(imp_data).to_csv('sample_data/impairment_enhanced.csv', index=False)
    
    # Generate sample AHP AI data
    ahp_data = {
        'Session_ID': [f'S{i:03d}' for i in range(1, 61)],
        'Time': list(range(1, 37)) + list(range(1, 25)),
        'QoE_AHP_AI': np.linspace(52, 100, 60),
        'Network_State': ['Fair'] * 10 + ['Good'] * 20 + ['Excellent'] * 30
    }
    pd.DataFrame(ahp_data).to_csv('sample_data/ahp_ai.csv', index=False)
    
    # Generate sample application specific data
    app_data = {
        'Session_ID': [f'S{i:03d}' for i in range(1, 61)],
        'QoE_Application_Specific': np.linspace(0.625, 0.999, 60)
    }
    pd.DataFrame(app_data).to_csv('sample_data/app_specific.csv', index=False)
    
    # Generate sample context aware data
    ctx_data = {
        'Session_ID': [f'S{i:03d}' for i in range(1, 61)],
        'QoE_Context_Aware': np.linspace(0.595, 0.999, 60)
    }
    pd.DataFrame(ctx_data).to_csv('sample_data/context_aware.csv', index=False)
    
    # Generate sample 5G baseline data
    baseline_data = {
        'Session_ID': [f'S{i:03d}' for i in range(1, 61)],
        'QoE_5G_Normalized': np.linspace(0.76, 0.98, 60),
        'QoE_5G_Percent': np.linspace(76, 98, 60)
    }
    pd.DataFrame(baseline_data).to_csv('sample_data/baseline_5g.csv', index=False)
    
    # Generate sample config file
    config = {
        "transition_months": 36,
        "eta_method": "adaptive",
        "output_format": "excel",
        "generate_plots": True
    }
    with open('sample_data/config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print("Sample files generated in 'sample_data' directory")


if __name__ == "__main__":
    # Uncomment to generate sample files for testing
    # generate_sample_files()
    
    # Run main calculation
    main()
