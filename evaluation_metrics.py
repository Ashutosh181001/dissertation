"""
Real-time Anomaly Detection Evaluation System

This module tracks the performance of anomaly detection models in real-time,
calculating precision, recall, F1 scores, and other metrics.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns

class AnomalyEvaluator:
    """
    Tracks and evaluates anomaly detection performance metrics.
    """

    def __init__(self, window_minutes=60):
        self.window_minutes = window_minutes
        self.metrics_log = "evaluation_metrics.csv"
        self.summary_log = "evaluation_summary.json"

        # Performance tracking
        self.detections = defaultdict(lambda: {
            'true_positives': 0,
            'false_positives': 0,
            'false_negatives': 0,
            'true_negatives': 0,
            'detection_times': [],
            'confidence_scores': [],
            'latencies': [],  # Time to detect anomaly
            'severity_distribution': defaultdict(int)
        })

        # Ground truth tracking (for labeled data)
        self.ground_truth_anomalies = set()

        # Time-based metrics
        self.hourly_metrics = defaultdict(lambda: defaultdict(int))

        # Sliding window for real-time metrics
        self.sliding_window = defaultdict(list)
        self.window_size = 1000  # Last 1000 detections

        # Load existing metrics if available
        self._load_existing_metrics()

    def _load_existing_metrics(self):
        """Load existing metrics from summary file if it exists"""
        if os.path.exists(self.summary_log):
            try:
                with open(self.summary_log, 'r') as f:
                    data = json.load(f)
                    # Restore detection counts from saved metrics
                    if 'metrics' in data:
                        for model_name, metrics in data['metrics'].items():
                            # Restore the actual counts if available
                            if 'true_positives' in metrics:
                                self.detections[model_name]['true_positives'] = metrics['true_positives']
                            if 'false_positives' in metrics:
                                self.detections[model_name]['false_positives'] = metrics['false_positives']
                            if 'false_negatives' in metrics:
                                self.detections[model_name]['false_negatives'] = metrics['false_negatives']
                            if 'true_negatives' in metrics:
                                self.detections[model_name]['true_negatives'] = metrics['true_negatives']

                print(f"Loaded existing metrics for {len(self.detections)} models")
            except Exception as e:
                print(f"Could not load existing metrics: {e}")

    def add_ground_truth(self, timestamp: str, is_anomaly: bool):
        """Mark a timestamp as containing a true anomaly"""
        if is_anomaly:
            self.ground_truth_anomalies.add(timestamp)

    def record_detection(self, timestamp: str, model_name: str,
                        detected: bool, confidence: float = None,
                        actual_anomaly: bool = None, latency_ms: float = None):
        """
        Record a detection event with enhanced tracking.

        Parameters:
        -----------
        timestamp: str
            Timestamp of the detection
        model_name: str
            Name of the model making the detection
        detected: bool
            Whether an anomaly was detected
        confidence: float
            Confidence score (0-1) if available
        actual_anomaly: bool
            Ground truth if known (for supervised evaluation)
        latency_ms: float
            Detection latency in milliseconds
        """

        # If we have ground truth, use it
        if actual_anomaly is None:
            # Check if this timestamp is in our ground truth set
            actual_anomaly = timestamp in self.ground_truth_anomalies
            # If still None, assume normal (no anomaly) for TN/FP calculation
            if actual_anomaly is None:
                actual_anomaly = False

        # Update confusion matrix
        if detected and actual_anomaly:
            self.detections[model_name]['true_positives'] += 1
            detection_type = 'TP'
        elif detected and not actual_anomaly:
            self.detections[model_name]['false_positives'] += 1
            detection_type = 'FP'
        elif not detected and actual_anomaly:
            self.detections[model_name]['false_negatives'] += 1
            detection_type = 'FN'
        else:
            self.detections[model_name]['true_negatives'] += 1
            detection_type = 'TN'

        # Track detection time and confidence
        self.detections[model_name]['detection_times'].append(timestamp)
        if confidence is not None:
            self.detections[model_name]['confidence_scores'].append(confidence)

        if latency_ms is not None:
            self.detections[model_name]['latencies'].append(latency_ms)

        # Update sliding window
        self.sliding_window[model_name].append({
            'timestamp': timestamp,
            'type': detection_type,
            'confidence': confidence
        })

        # Keep window size limited
        if len(self.sliding_window[model_name]) > self.window_size:
            self.sliding_window[model_name].pop(0)

        # Update hourly metrics
        hour_key = pd.to_datetime(timestamp).strftime('%Y-%m-%d %H:00')
        self.hourly_metrics[hour_key][f"{model_name}_detections"] += int(detected)
        self.hourly_metrics[hour_key][f"{model_name}_{detection_type}"] += 1

        # Log to file
        self._log_detection(timestamp, model_name, detected, confidence, actual_anomaly, detection_type)

    def get_metrics(self, model_name: str = None) -> Dict[str, float]:
        """
        Calculate performance metrics for a specific model or all models.

        Returns:
        --------
        Dictionary containing precision, recall, F1 score, etc.
        """
        if model_name:
            models = [model_name]
        else:
            models = list(self.detections.keys())

        all_metrics = {}

        for model in models:
            stats = self.detections[model]
            tp = stats['true_positives']
            fp = stats['false_positives']
            fn = stats['false_negatives']
            tn = stats['true_negatives']

            # Calculate metrics
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0

            # False positive rate
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

            # Matthews Correlation Coefficient
            mcc_num = (tp * tn) - (fp * fn)
            mcc_den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
            mcc = mcc_num / mcc_den if mcc_den > 0 else 0

            # Additional metrics
            avg_confidence = np.mean(stats['confidence_scores']) if stats['confidence_scores'] else None
            avg_latency = np.mean(stats['latencies']) if stats['latencies'] else None

            # Sliding window metrics (recent performance)
            recent_precision = precision
            if model in self.sliding_window and len(self.sliding_window[model]) > 0:
                recent = self.sliding_window[model][-100:]  # Last 100 detections
                recent_tp = sum(1 for r in recent if r['type'] == 'TP')
                recent_fp = sum(1 for r in recent if r['type'] == 'FP')
                recent_precision = recent_tp / (recent_tp + recent_fp) if (recent_tp + recent_fp) > 0 else 0

            all_metrics[model] = {
                'precision': round(precision, 4),
                'recall': round(recall, 4),
                'f1_score': round(f1, 4),
                'accuracy': round(accuracy, 4),
                'false_positive_rate': round(fpr, 4),
                'mcc': round(mcc, 4),
                'total_detections': tp + fp,
                'total_anomalies': tp + fn,
                'true_positives': tp,
                'false_positives': fp,
                'false_negatives': fn,
                'true_negatives': tn,
                'avg_confidence': round(avg_confidence, 4) if avg_confidence else None,
                'avg_latency_ms': round(avg_latency, 2) if avg_latency else None,
                'recent_precision': round(recent_precision, 4)
            }

        return all_metrics

    def get_real_time_metrics(self, model_name: str = None, window_size: int = 100) -> Dict:
        """Get real-time performance metrics based on recent detections"""
        metrics = {}
        models = [model_name] if model_name else list(self.sliding_window.keys())

        for model in models:
            if model in self.sliding_window:
                recent = self.sliding_window[model][-window_size:]
                if recent:
                    tp = sum(1 for r in recent if r['type'] == 'TP')
                    fp = sum(1 for r in recent if r['type'] == 'FP')
                    fn = sum(1 for r in recent if r['type'] == 'FN')
                    tn = sum(1 for r in recent if r['type'] == 'TN')

                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

                    metrics[model] = {
                        'window_size': len(recent),
                        'precision': round(precision, 4),
                        'recall': round(recall, 4),
                        'f1_score': round(f1, 4),
                        'detections': tp + fp,
                        'anomalies': tp + fn
                    }

        return metrics

    def get_time_based_metrics(self, hours: int = 24) -> pd.DataFrame:
        """Get metrics over the last N hours"""
        cutoff_time = datetime.now() - timedelta(hours=hours)

        metrics_data = []
        for hour, metrics in self.hourly_metrics.items():
            hour_dt = pd.to_datetime(hour)
            if hour_dt >= cutoff_time:
                metrics_data.append({
                    'hour': hour,
                    **metrics
                })

        if metrics_data:
            df = pd.DataFrame(metrics_data).sort_values('hour')
            # Calculate hourly precision for each model
            for model in self.detections.keys():
                tp_col = f"{model}_TP"
                fp_col = f"{model}_FP"
                if tp_col in df.columns and fp_col in df.columns:
                    df[f"{model}_precision"] = df.apply(
                        lambda row: row[tp_col] / (row[tp_col] + row[fp_col])
                        if (row.get(tp_col, 0) + row.get(fp_col, 0)) > 0 else 0,
                        axis=1
                    )
            return df

        return pd.DataFrame()

    def plot_performance(self, save_path: str = "evaluation_plots.png"):
        """Generate enhanced performance visualization plots"""
        metrics = self.get_metrics()

        if not metrics:
            print("No metrics to plot yet")
            return

        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Anomaly Detection Performance Metrics', fontsize=16, fontweight='bold')

        # Color palette
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E']

        # 1. Model Comparison Bar Chart
        ax1 = axes[0, 0]
        models = list(metrics.keys())
        f1_scores = [metrics[m]['f1_score'] for m in models]
        precisions = [metrics[m]['precision'] for m in models]
        recalls = [metrics[m]['recall'] for m in models]

        x = np.arange(len(models))
        width = 0.25

        bars1 = ax1.bar(x - width, f1_scores, width, label='F1 Score', alpha=0.8, color=colors[0])
        bars2 = ax1.bar(x, precisions, width, label='Precision', alpha=0.8, color=colors[1])
        bars3 = ax1.bar(x + width, recalls, width, label='Recall', alpha=0.8, color=colors[2])

        ax1.set_xlabel('Models', fontweight='bold')
        ax1.set_ylabel('Score', fontweight='bold')
        ax1.set_title('Model Performance Comparison', fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(models, rotation=45, ha='right')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([0, 1.05])

        # Add value labels on bars
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax1.text(bar.get_x() + bar.get_width()/2., height,
                            f'{height:.2f}', ha='center', va='bottom', fontsize=8)

        # 2. Confusion Matrix Heatmap (for best model)
        ax2 = axes[0, 1]
        if metrics:
            best_model = max(metrics.items(), key=lambda x: x[1]['f1_score'])[0]
            stats = self.detections[best_model]
            cm = np.array([
                [stats['true_negatives'], stats['false_positives']],
                [stats['false_negatives'], stats['true_positives']]
            ])

            # Create percentage annotations
            cm_sum = cm.sum()
            cm_percentages = cm / cm_sum * 100 if cm_sum > 0 else cm

            annot_text = np.array([[f'{cm[0,0]}\n({cm_percentages[0,0]:.1f}%)',
                                    f'{cm[0,1]}\n({cm_percentages[0,1]:.1f}%)'],
                                   [f'{cm[1,0]}\n({cm_percentages[1,0]:.1f}%)',
                                    f'{cm[1,1]}\n({cm_percentages[1,1]:.1f}%)']])

            sns.heatmap(cm, annot=annot_text, fmt='', cmap='Blues', ax=ax2,
                       xticklabels=['Normal', 'Anomaly'],
                       yticklabels=['Normal', 'Anomaly'],
                       cbar_kws={'label': 'Count'})
            ax2.set_title(f'Confusion Matrix - {best_model}', fontweight='bold')
            ax2.set_xlabel('Predicted', fontweight='bold')
            ax2.set_ylabel('Actual', fontweight='bold')

        # 3. Time-based Detection Rate
        ax3 = axes[0, 2]
        time_df = self.get_time_based_metrics(24)
        if not time_df.empty:
            time_df['hour'] = pd.to_datetime(time_df['hour'])

            for i, col in enumerate([c for c in time_df.columns if c.endswith('_detections')]):
                model_name = col.replace('_detections', '')
                ax3.plot(time_df['hour'], time_df[col],
                        label=model_name, marker='o', linewidth=2,
                        color=colors[i % len(colors)])

            ax3.set_xlabel('Time', fontweight='bold')
            ax3.set_ylabel('Number of Detections', fontweight='bold')
            ax3.set_title('Detection Rate Over Time (24h)', fontweight='bold')
            ax3.legend(loc='best')
            ax3.grid(True, alpha=0.3)
            plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # 4. ROC-style plot (FPR vs Recall)
        ax4 = axes[1, 0]
        for i, model in enumerate(models):
            fpr = metrics[model]['false_positive_rate']
            recall = metrics[model]['recall']
            f1 = metrics[model]['f1_score']
            ax4.scatter(fpr, recall, s=150, label=f"{model} (F1: {f1:.3f})",
                       color=colors[i % len(colors)], edgecolors='black', linewidth=1)

        ax4.set_xlabel('False Positive Rate', fontweight='bold')
        ax4.set_ylabel('True Positive Rate (Recall)', fontweight='bold')
        ax4.set_title('Model Operating Points', fontweight='bold')
        ax4.set_xlim(-0.05, 1.05)
        ax4.set_ylim(-0.05, 1.05)
        ax4.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Random')
        ax4.legend(loc='best')
        ax4.grid(True, alpha=0.3)

        # 5. Precision over Time (sliding window)
        ax5 = axes[1, 1]
        if not time_df.empty:
            for i, model in enumerate(models):
                precision_col = f"{model}_precision"
                if precision_col in time_df.columns:
                    ax5.plot(time_df['hour'], time_df[precision_col],
                            label=model, marker='s', linewidth=2,
                            color=colors[i % len(colors)])

            ax5.set_xlabel('Time', fontweight='bold')
            ax5.set_ylabel('Precision', fontweight='bold')
            ax5.set_title('Precision Trend (24h)', fontweight='bold')
            ax5.legend(loc='best')
            ax5.grid(True, alpha=0.3)
            ax5.set_ylim([0, 1.05])
            plt.setp(ax5.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # 6. Detection Distribution
        ax6 = axes[1, 2]
        detection_data = []
        for model in models:
            stats = self.detections[model]
            detection_data.append([
                stats['true_positives'],
                stats['false_positives'],
                stats['false_negatives'],
                stats['true_negatives']
            ])

        detection_df = pd.DataFrame(detection_data,
                                   columns=['TP', 'FP', 'FN', 'TN'],
                                   index=models)

        detection_df.plot(kind='bar', stacked=True, ax=ax6,
                         color=['#2E7D32', '#C62828', '#F57C00', '#1976D2'],
                         alpha=0.8)
        ax6.set_xlabel('Models', fontweight='bold')
        ax6.set_ylabel('Count', fontweight='bold')
        ax6.set_title('Detection Distribution by Type', fontweight='bold')
        ax6.legend(title='Type', loc='best')
        ax6.grid(True, alpha=0.3)
        plt.setp(ax6.xaxis.get_majorticklabels(), rotation=45, ha='right')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Performance plots saved to {save_path}")

    def generate_report(self) -> str:
        """Generate a comprehensive evaluation report"""
        metrics = self.get_metrics()
        real_time_metrics = self.get_real_time_metrics()

        report = ["="*60]
        report.append("ANOMALY DETECTION EVALUATION REPORT")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("="*60)
        report.append("")

        for model, model_metrics in metrics.items():
            report.append(f"\n{model.upper()}")
            report.append("-" * len(model))

            # Performance metrics
            report.append(f"Precision: {model_metrics['precision']:.2%}")
            report.append(f"Recall: {model_metrics['recall']:.2%}")
            report.append(f"F1 Score: {model_metrics['f1_score']:.2%}")
            report.append(f"Accuracy: {model_metrics['accuracy']:.2%}")
            report.append(f"False Positive Rate: {model_metrics['false_positive_rate']:.2%}")
            report.append(f"Matthews Correlation: {model_metrics['mcc']:.3f}")

            # Real-time performance
            if model in real_time_metrics:
                rt = real_time_metrics[model]
                report.append(f"\nRecent Performance (last {rt['window_size']} detections):")
                report.append(f"  Recent Precision: {rt['precision']:.2%}")
                report.append(f"  Recent Recall: {rt['recall']:.2%}")
                report.append(f"  Recent F1: {rt['f1_score']:.2%}")

            # Detection statistics
            report.append(f"\nTotal Detections: {model_metrics['total_detections']}")
            report.append(f"Total True Anomalies: {model_metrics['total_anomalies']}")

            if model_metrics['avg_confidence'] is not None:
                report.append(f"Average Confidence: {model_metrics['avg_confidence']:.3f}")

            if model_metrics.get('avg_latency_ms') is not None:
                report.append(f"Average Latency: {model_metrics['avg_latency_ms']:.2f} ms")

            # Confusion matrix
            report.append(f"\nConfusion Matrix:")
            report.append(f"  TP: {model_metrics['true_positives']} | FP: {model_metrics['false_positives']}")
            report.append(f"  FN: {model_metrics['false_negatives']} | TN: {model_metrics['true_negatives']}")

        # Best performing model
        if metrics:
            best_model = max(metrics.items(), key=lambda x: x[1]['f1_score'])
            report.append(f"\n{'='*60}")
            report.append(f"BEST PERFORMING MODEL: {best_model[0]} (F1: {best_model[1]['f1_score']:.2%})")

        return '\n'.join(report)

    def _log_detection(self, timestamp, model_name, detected, confidence, actual, detection_type):
        """Log detection event to CSV"""
        log_entry = {
            'timestamp': timestamp,
            'model': model_name,
            'detected': detected,
            'confidence': confidence,
            'actual_anomaly': actual,
            'detection_type': detection_type,
            'logged_at': datetime.now().isoformat()
        }

        pd.DataFrame([log_entry]).to_csv(
            self.metrics_log,
            mode='a',
            header=not os.path.exists(self.metrics_log),
            index=False
        )

    def save_summary(self):
        """Save summary metrics to JSON"""
        metrics = self.get_metrics()
        real_time_metrics = self.get_real_time_metrics()

        # Calculate total evaluations properly
        total_evaluations = 0
        for model_name, model_metrics in metrics.items():
            total_evaluations += (
                model_metrics.get('true_positives', 0) +
                model_metrics.get('false_positives', 0) +
                model_metrics.get('false_negatives', 0) +
                model_metrics.get('true_negatives', 0)
            )

        summary = {
            'generated_at': datetime.now().isoformat(),
            'metrics': metrics,
            'real_time_metrics': real_time_metrics,
            'total_evaluations': total_evaluations,
            'evaluation_window_minutes': self.window_minutes
        }

        with open(self.summary_log, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"Summary saved to {self.summary_log}")


# Example usage
if __name__ == "__main__":
    # Initialize evaluator
    evaluator = AnomalyEvaluator()

    # Simulate some detections for testing
    import random

    models = ['z_score', 'isoforest', 'filtered_isoforest']

    for i in range(100):
        timestamp = (datetime.now() - timedelta(hours=random.randint(0, 24))).isoformat()
        for model in models:
            # Simulate detection with some randomness
            detected = random.random() > 0.7
            actual = random.random() > 0.8
            confidence = random.random()

            evaluator.record_detection(timestamp, model, detected, confidence, actual)

    # Get metrics
    print(evaluator.generate_report())

    # Generate plots
    evaluator.plot_performance(save_path="evaluation_plots.png")

    # Save summary
    evaluator.save_summary()