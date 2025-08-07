import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import seaborn as sns

@dataclass
class NetworkMetrics:
    """Network performance metrics"""
    delay_ms: float
    packet_loss: float
    jitter_ms: float
    bitrate_mbps: float
    
@dataclass
class VideoMetrics:
    """Video streaming specific metrics"""
    resolution: str
    codec: str
    vmaf_score: float
    buffering_ratio: float
    startup_delay: float
    
@dataclass
class ContextMetrics:
    """Context-aware metrics"""
    device_type: str
    location_type: str
    time_of_day: str
    network_type: str
    user_activity: str

class QoECalculator:
    """
    Comprehensive QoE Calculator for 5G to 6G transition
    Implements: QoE_Total = η(t)·QoE_6G + (1-η(t))·QoE_5G
    """
    
    def __init__(self):
        self.transition_month = 36  # Total transition period
        self.network_conditions = {
            'Poor': 0.6,
            'Fair': 0.7,
            'Good': 0.8,
            'Excellent': 0.95
        }
        
    def calculate_qoe_5g(self, network_metrics: NetworkMetrics, 
                        video_metrics: VideoMetrics) -> float:
        """
        Calculate traditional 5G QoE based on network and video metrics
        Returns: QoE score normalized to 0-1
        """
        # Network performance component
        delay_score = 1 - min(network_metrics.delay_ms / 100, 1)
        loss_score = 1 - min(network_metrics.packet_loss * 1000, 1)
        jitter_score = 1 - min(network_metrics.jitter_ms / 10, 1)
        
        # Video quality component
        resolution_scores = {
            '720p': 0.7, '1080p': 0.8, '1440p': 0.85,
            '2160p': 0.9, '4K': 0.95, '8K': 1.0
        }
        res_score = resolution_scores.get(video_metrics.resolution, 0.7)
        
        # Calculate MOS based on VMAF
        mos_video = 1 + (video_metrics.vmaf_score / 100) * 4
        
        # Buffering impact
        buffering_impact = 1 - video_metrics.buffering_ratio
        
        # Combine components
        network_qoe = 0.3 * delay_score + 0.3 * loss_score + 0.2 * jitter_score + 0.2 * (network_metrics.bitrate_mbps / 100)
        video_qoe = 0.4 * (mos_video / 5) + 0.3 * res_score + 0.3 * buffering_impact
        
        # Final 5G QoE
        qoe_5g = 0.5 * network_qoe + 0.5 * video_qoe
        
        return min(qoe_5g, 0.98)  # 5G typically caps at 98%
    
    def calculate_qoe_impairment_enhanced(self, network_metrics: NetworkMetrics, 
                                         time_month: int) -> float:
        """
        Calculate enhanced impairment score for 6G
        Includes sub-millisecond precision and AI prediction
        """
        # Enhanced delay calculation with 6G improvements
        delay_factor = np.exp(-network_metrics.delay_ms / 10)
        
        # Advanced codec efficiency in 6G
        loss_factor = 1 - (network_metrics.packet_loss * 100)
        
        # Sub-millisecond jitter handling
        jitter_factor = 1 - (network_metrics.jitter_ms / 5)
        
        # Time-based improvement factor
        improvement_factor = 1 + (time_month / self.transition_month) * 0.5
        
        # Calculate R-value (E-model based)
        r_value = 100 - (15 * (1 - delay_factor)) - (20 * (1 - loss_factor)) - (10 * (1 - jitter_factor))
        
        # Normalize and apply improvement
        qoe_impairment = (r_value / 100) * improvement_factor
        
        return min(qoe_impairment, 1.2)  # Can exceed 1.0 due to 6G enhancements
    
    def calculate_qoe_ahp_ai(self, video_metrics: VideoMetrics, 
                            context_metrics: ContextMetrics,
                            time_month: int) -> float:
        """
        AI-enhanced AHP (Analytic Hierarchy Process) QoE calculation
        Uses reinforcement learning weights
        """
        # Define criteria
        criteria = {
            'video_quality': video_metrics.vmaf_score / 100,
            'buffering': 1 - video_metrics.buffering_ratio,
            'startup_delay': 1 - min(video_metrics.startup_delay / 5, 1),
            'bitrate_adaptation': 0.8 + 0.2 * (time_month / self.transition_month),
            'frame_stability': 0.85 + 0.15 * (time_month / self.transition_month)
        }
        
        # Dynamic weights based on learning progress
        if time_month < 12:
            weights = [0.2, 0.2, 0.2, 0.2, 0.2]
        elif time_month < 24:
            weights = [0.35, 0.25, 0.15, 0.15, 0.1]
        else:
            weights = [0.45, 0.2, 0.1, 0.1, 0.15]
            
        # Calculate weighted sum
        qoe_ahp = sum(w * v for w, v in zip(weights, criteria.values()))
        
        return qoe_ahp
    
    def calculate_qoe_application_specific(self, video_metrics: VideoMetrics,
                                          context_metrics: ContextMetrics) -> float:
        """
        Application-specific QoE for video streaming
        """
        # Base video quality
        video_quality = video_metrics.vmaf_score / 100
        
        # Resolution impact
        resolution_factor = {
            '720p': 0.7, '1080p': 0.8, '1440p': 0.85,
            '2160p': 0.9, '4K': 0.95, '8K': 1.0
        }.get(video_metrics.resolution, 0.7)
        
        # Codec efficiency
        codec_factor = {
            'H264': 0.8, 'H265': 0.9, 'AV1': 1.0, 'VVC': 1.1
        }.get(video_metrics.codec, 0.8)
        
        # Context multiplier
        context_multiplier = self.network_conditions.get(
            context_metrics.network_type, 0.7
        )
        
        # Calculate QoE
        qoe_app = (0.4 * video_quality + 
                  0.3 * resolution_factor + 
                  0.2 * codec_factor + 
                  0.1 * context_multiplier)
        
        return qoe_app
    
    def calculate_qoe_context_aware(self, context_metrics: ContextMetrics,
                                   network_metrics: NetworkMetrics) -> float:
        """
        Context-aware QoE calculation
        """
        # Location factor
        location_scores = {
            'Indoor': 0.95, 'Outdoor': 0.75, 'Transit': 0.65, 'Office': 0.9
        }
        location_factor = location_scores.get(context_metrics.location_type, 0.7)
        
        # Time factor
        time_scores = {
            'Morning': 0.85, 'Afternoon': 0.8, 'Evening': 0.75, 'Night': 0.9
        }
        time_factor = time_scores.get(context_metrics.time_of_day, 0.8)
        
        # Device factor
        device_scores = {
            'Mobile': 0.75, 'Tablet': 0.85, 'Desktop': 0.95, 'SmartTV': 1.0
        }
        device_factor = device_scores.get(context_metrics.device_type, 0.8)
        
        # Network factor based on actual performance
        network_factor = min((network_metrics.bitrate_mbps / 50), 1.0)
        
        # Weighted combination
        weights = [0.15, 0.1, 0.2, 0.25, 0.15, 0.1, 0.05]
        factors = [location_factor, time_factor, device_factor, network_factor, 
                  0.85, 0.9, 0.8]  # Additional factors for user, environment, social
        
        qoe_context = sum(w * f for w, f in zip(weights, factors))
        
        return qoe_context
    
    def calculate_dynamic_weights(self, network_condition: str, 
                                 time_month: int) -> Dict[str, float]:
        """
        Calculate dynamic weights based on network condition and time
        Returns: Dictionary with weights α, β, γ, δ
        """
        progress = time_month / self.transition_month
        
        if network_condition == 'Excellent':
            weights = {
                'alpha': 0.15 - 0.05 * progress,
                'beta': 0.25 + 0.05 * progress,
                'gamma': 0.35 + 0.05 * progress,
                'delta': 0.25 - 0.05 * progress
            }
        elif network_condition == 'Good':
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
            
        # Ensure weights sum to 1
        total = sum(weights.values())
        return {k: v/total for k, v in weights.items()}
    
    def calculate_qoe_6g(self, network_metrics: NetworkMetrics,
                        video_metrics: VideoMetrics,
                        context_metrics: ContextMetrics,
                        time_month: int) -> Tuple[float, Dict[str, float]]:
        """
        Calculate complete 6G QoE with all components
        Returns: (QoE_6G, component_scores)
        """
        # Calculate individual components
        qoe_impairment = self.calculate_qoe_impairment_enhanced(network_metrics, time_month)
        qoe_ahp = self.calculate_qoe_ahp_ai(video_metrics, context_metrics, time_month)
        qoe_app = self.calculate_qoe_application_specific(video_metrics, context_metrics)
        qoe_context = self.calculate_qoe_context_aware(context_metrics, network_metrics)
        
        # Get dynamic weights
        network_condition = self._determine_network_condition(network_metrics)
        weights = self.calculate_dynamic_weights(network_condition, time_month)
        
        # Calculate weighted sum
        qoe_6g = (weights['alpha'] * qoe_impairment +
                 weights['beta'] * qoe_ahp +
                 weights['gamma'] * qoe_app +
                 weights['delta'] * qoe_context)
        
        components = {
            'impairment': qoe_impairment,
            'ahp_ai': qoe_ahp,
            'application': qoe_app,
            'context': qoe_context,
            'weights': weights
        }
        
        return qoe_6g, components
    
    def calculate_eta(self, time_month: int, method: str = 'adaptive') -> float:
        """
        Calculate transition factor η(t)
        Methods: 'linear', 'sigmoid', 'adaptive'
        """
        t = time_month
        
        if method == 'linear':
            return t / self.transition_month
        
        elif method == 'sigmoid':
            return 1 / (1 + np.exp(-0.3 * (t - self.transition_month/2)))
        
        elif method == 'adaptive':
            # Custom adaptive function based on deployment reality
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
            raise ValueError(f"Unknown method: {method}")
    
    def calculate_qoe_total(self, network_metrics: NetworkMetrics,
                           video_metrics: VideoMetrics,
                           context_metrics: ContextMetrics,
                           time_month: int,
                           eta_method: str = 'adaptive') -> Dict:
        """
        Calculate total QoE combining 5G and 6G
        Returns comprehensive results dictionary
        """
        # Calculate 5G QoE
        qoe_5g = self.calculate_qoe_5g(network_metrics, video_metrics)
        
        # Calculate 6G QoE and components
        qoe_6g, components = self.calculate_qoe_6g(network_metrics, video_metrics, 
                                                   context_metrics, time_month)
        
        # Calculate transition factor
        eta = self.calculate_eta(time_month, eta_method)
        
        # Calculate total QoE
        qoe_total = eta * qoe_6g + (1 - eta) * qoe_5g
        
        return {
            'time_month': time_month,
            'qoe_5g': qoe_5g,
            'qoe_6g': qoe_6g,
            'qoe_total': qoe_total,
            'eta': eta,
            'components': components,
            'qoe_total_percent': qoe_total * 100
        }
    
    def _determine_network_condition(self, network_metrics: NetworkMetrics) -> str:
        """Determine network condition based on metrics"""
        score = 0
        
        if network_metrics.delay_ms < 5:
            score += 3
        elif network_metrics.delay_ms < 20:
            score += 2
        elif network_metrics.delay_ms < 50:
            score += 1
            
        if network_metrics.packet_loss < 0.0001:
            score += 3
        elif network_metrics.packet_loss < 0.001:
            score += 2
        elif network_metrics.packet_loss < 0.01:
            score += 1
            
        if network_metrics.bitrate_mbps > 50:
            score += 3
        elif network_metrics.bitrate_mbps > 20:
            score += 2
        elif network_metrics.bitrate_mbps > 10:
            score += 1
            
        if score >= 8:
            return 'Excellent'
        elif score >= 5:
            return 'Good'
        elif score >= 3:
            return 'Fair'
        else:
            return 'Poor'


def simulate_network_evolution(month: int) -> NetworkMetrics:
    """Simulate network metrics evolution over time"""
    # Simulate improving network conditions
    delay = 25 * np.exp(-month / 12) + 0.1
    packet_loss = 0.001 * np.exp(-month / 10) + 0.00001
    jitter = 4 * np.exp(-month / 15) + 0.01
    bitrate = 10 + 80 * (month / 36) + np.random.normal(0, 5)
    
    return NetworkMetrics(
        delay_ms=max(delay, 0.1),
        packet_loss=max(packet_loss, 0.00001),
        jitter_ms=max(jitter, 0.01),
        bitrate_mbps=min(max(bitrate, 5), 100)
    )


def main():
    """Main function to demonstrate QoE calculation"""
    calculator = QoECalculator()
    
    # Simulate QoE evolution over 36 months
    results = []
    
    for month in range(1, 37):
        # Simulate evolving metrics
        network_metrics = simulate_network_evolution(month)
        
        # Video metrics improve over time
        video_metrics = VideoMetrics(
            resolution='1080p' if month < 12 else ('4K' if month < 24 else '8K'),
            codec='H264' if month < 12 else ('H265' if month < 24 else 'AV1'),
            vmaf_score=75 + 20 * (month / 36),
            buffering_ratio=0.05 * np.exp(-month / 10),
            startup_delay=3 * np.exp(-month / 15)
        )
        
        # Context metrics
        context_metrics = ContextMetrics(
            device_type='SmartTV',
            location_type='Indoor',
            time_of_day='Evening',
            network_type='Good' if month < 12 else 'Excellent',
            user_activity='Active'
        )
        
        # Calculate QoE
        result = calculator.calculate_qoe_total(
            network_metrics, video_metrics, context_metrics, month
        )
        results.append(result)
    
    # Convert to DataFrame for analysis
    df = pd.DataFrame(results)
    
    # Plot results
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(df['time_month'], df['qoe_5g'], 'b-', label='QoE 5G', linewidth=2)
    plt.plot(df['time_month'], df['qoe_6g'], 'r-', label='QoE 6G', linewidth=2)
    plt.plot(df['time_month'], df['qoe_total'], 'g-', label='QoE Total', linewidth=3)
    plt.axhline(y=1.0, color='k', linestyle='--', alpha=0.5)
    plt.xlabel('Time (months)')
    plt.ylabel('QoE Score')
    plt.title('QoE Evolution: 5G to 6G Transition')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 2)
    plt.plot(df['time_month'], df['eta'], 'purple', linewidth=2)
    plt.xlabel('Time (months)')
    plt.ylabel('Transition Factor (η)')
    plt.title('6G Adoption Rate (Adaptive)')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 3)
    # Extract component scores
    components_df = pd.DataFrame([r['components'] for r in results])
    plt.plot(df['time_month'], components_df['impairment'], label='Impairment')
    plt.plot(df['time_month'], components_df['ahp_ai'], label='AHP AI')
    plt.plot(df['time_month'], components_df['application'], label='Application')
    plt.plot(df['time_month'], components_df['context'], label='Context')
    plt.xlabel('Time (months)')
    plt.ylabel('Component Score')
    plt.title('6G QoE Components Evolution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 4)
    # Plot dynamic weights
    weights_alpha = [r['components']['weights']['alpha'] for r in results]
    weights_beta = [r['components']['weights']['beta'] for r in results]
    weights_gamma = [r['components']['weights']['gamma'] for r in results]
    weights_delta = [r['components']['weights']['delta'] for r in results]
    
    plt.plot(df['time_month'], weights_alpha, label='α (Impairment)')
    plt.plot(df['time_month'], weights_beta, label='β (AHP AI)')
    plt.plot(df['time_month'], weights_gamma, label='γ (Application)')
    plt.plot(df['time_month'], weights_delta, label='δ (Context)')
    plt.xlabel('Time (months)')
    plt.ylabel('Weight Value')
    plt.title('Dynamic Weight Evolution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print("QoE Evolution Summary")
    print("=" * 50)
    print(f"Initial QoE Total: {results[0]['qoe_total_percent']:.2f}%")
    print(f"Final QoE Total: {results[-1]['qoe_total_percent']:.2f}%")
    print(f"Total Improvement: {results[-1]['qoe_total_percent'] - results[0]['qoe_total_percent']:.2f}%")
    print(f"\nMonth when QoE exceeds 90%: {next((r['time_month'] for r in results if r['qoe_total'] > 0.9), 'N/A')}")
    print(f"Month when QoE exceeds 100%: {next((r['time_month'] for r in results if r['qoe_total'] > 1.0), 'N/A')}")
    
    # Component contribution at final month
    final_components = results[-1]['components']
    final_weights = final_components['weights']
    print(f"\nFinal Component Contributions:")
    print(f"  Impairment: {final_weights['alpha']*100:.1f}%")
    print(f"  AHP AI: {final_weights['beta']*100:.1f}%")
    print(f"  Application: {final_weights['gamma']*100:.1f}%")
    print(f"  Context: {final_weights['delta']*100:.1f}%")


if __name__ == "__main__":
    main()
