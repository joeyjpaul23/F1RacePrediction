"""
Model Evaluation for F1 Race Predictions
Provides metrics, performance analysis, and validation tools.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score, KFold
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from typing import Dict, List, Tuple, Optional
import json
import os
from datetime import datetime

logger = logging.getLogger(__name__)

class ModelEvaluator:
    """Evaluates and analyzes F1 prediction model performance."""
    
    def __init__(self, results_path: str = 'models/evaluation_results/'):
        self.results_path = results_path
        self.evaluation_history = []
        os.makedirs(results_path, exist_ok=True)
    
    def evaluate_predictions(self, 
                           y_true: np.ndarray, 
                           y_pred: np.ndarray,
                           model_name: str = "ensemble") -> Dict:
        """Evaluate prediction accuracy using multiple metrics."""
        try:
            # Basic regression metrics
            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
            
            # Position-specific metrics
            position_accuracy = self._calculate_position_accuracy(y_true, y_pred)
            top_5_accuracy = self._calculate_top_n_accuracy(y_true, y_pred, n=5)
            top_10_accuracy = self._calculate_top_n_accuracy(y_true, y_pred, n=10)
            
            # Driver-specific analysis
            driver_performance = self._analyze_driver_performance(y_true, y_pred)
            
            # Track-specific analysis
            track_performance = self._analyze_track_performance(y_true, y_pred)
            
            results = {
                'model_name': model_name,
                'timestamp': datetime.now().isoformat(),
                'basic_metrics': {
                    'mse': mse,
                    'rmse': rmse,
                    'mae': mae,
                    'r2_score': r2
                },
                'position_metrics': {
                    'overall_accuracy': position_accuracy,
                    'top_5_accuracy': top_5_accuracy,
                    'top_10_accuracy': top_10_accuracy
                },
                'driver_performance': driver_performance,
                'track_performance': track_performance
            }
            
            # Save results
            self._save_evaluation_results(results)
            
            logger.info(f"Evaluation completed for {model_name}")
            return results
            
        except Exception as e:
            logger.error(f"Error evaluating predictions: {e}")
            raise
    
    def _calculate_position_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate accuracy of predicted positions."""
        # Count exact position matches
        exact_matches = np.sum(y_true == y_pred)
        return exact_matches / len(y_true)
    
    def _calculate_top_n_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray, n: int) -> float:
        """Calculate accuracy for top N positions."""
        # Count predictions within N positions of actual
        within_n = np.sum(np.abs(y_true - y_pred) <= n)
        return within_n / len(y_true)
    
    def _analyze_driver_performance(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Analyze prediction performance by driver position."""
        driver_stats = {}
        
        for position in range(1, 21):
            # Find predictions for this driver position
            mask = y_true == position
            if np.sum(mask) > 0:
                true_positions = y_true[mask]
                pred_positions = y_pred[mask]
                
                driver_stats[position] = {
                    'count': len(true_positions),
                    'avg_predicted_position': np.mean(pred_positions),
                    'std_predicted_position': np.std(pred_positions),
                    'exact_matches': np.sum(true_positions == pred_positions),
                    'within_1_position': np.sum(np.abs(true_positions - pred_positions) <= 1),
                    'within_3_positions': np.sum(np.abs(true_positions - pred_positions) <= 3)
                }
        
        return driver_stats
    
    def _analyze_track_performance(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Analyze prediction performance by track characteristics."""
        # This would require track information - simplified for now
        return {
            'note': 'Track-specific analysis requires additional track metadata'
        }
    
    def cross_validate_model(self, model, X: np.ndarray, y: np.ndarray, 
                           cv_folds: int = 5) -> Dict:
        """Perform cross-validation on the model."""
        try:
            kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
            
            # Cross-validation scores
            cv_scores = cross_val_score(model, X, y, cv=kfold, scoring='r2')
            cv_mae = cross_val_score(model, X, y, cv=kfold, scoring='neg_mean_absolute_error')
            cv_mse = cross_val_score(model, X, y, cv=kfold, scoring='neg_mean_squared_error')
            
            results = {
                'cv_folds': cv_folds,
                'r2_scores': cv_scores.tolist(),
                'mae_scores': (-cv_mae).tolist(),  # Convert back to positive
                'mse_scores': (-cv_mse).tolist(),  # Convert back to positive
                'r2_mean': np.mean(cv_scores),
                'r2_std': np.std(cv_scores),
                'mae_mean': np.mean(-cv_mae),
                'mae_std': np.std(-cv_mae),
                'mse_mean': np.mean(-cv_mse),
                'mse_std': np.std(-cv_mse)
            }
            
            logger.info(f"Cross-validation completed: R² = {results['r2_mean']:.4f} ± {results['r2_std']:.4f}")
            return results
            
        except Exception as e:
            logger.error(f"Error in cross-validation: {e}")
            raise
    
    def compare_models(self, model_results: Dict[str, Dict]) -> Dict:
        """Compare performance of multiple models."""
        try:
            comparison = {
                'models': list(model_results.keys()),
                'metrics': {}
            }
            
            # Extract metrics for comparison
            for metric in ['r2_score', 'mse', 'mae']:
                comparison['metrics'][metric] = {
                    model: results.get('basic_metrics', {}).get(metric, 0)
                    for model, results in model_results.items()
                }
            
            # Find best model for each metric
            comparison['best_models'] = {}
            for metric, values in comparison['metrics'].items():
                if metric == 'r2_score':
                    best_model = max(values, key=values.get)
                else:
                    best_model = min(values, key=values.get)
                comparison['best_models'][metric] = best_model
            
            # Calculate rankings
            comparison['rankings'] = self._calculate_model_rankings(model_results)
            
            return comparison
            
        except Exception as e:
            logger.error(f"Error comparing models: {e}")
            raise
    
    def _calculate_model_rankings(self, model_results: Dict[str, Dict]) -> Dict:
        """Calculate rankings for each model across different metrics."""
        metrics = ['r2_score', 'mse', 'mae']
        rankings = {}
        
        for metric in metrics:
            values = []
            for model, results in model_results.items():
                value = results.get('basic_metrics', {}).get(metric, 0)
                values.append((model, value))
            
            # Sort by metric value (descending for R², ascending for others)
            if metric == 'r2_score':
                values.sort(key=lambda x: x[1], reverse=True)
            else:
                values.sort(key=lambda x: x[1])
            
            rankings[metric] = [model for model, _ in values]
        
        return rankings
    
    def generate_performance_report(self, evaluation_results: Dict) -> str:
        """Generate a human-readable performance report."""
        try:
            report = []
            report.append("=" * 60)
            report.append("F1 RACE PREDICTION MODEL PERFORMANCE REPORT")
            report.append("=" * 60)
            report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            report.append("")
            
            # Basic metrics
            basic = evaluation_results.get('basic_metrics', {})
            report.append("BASIC METRICS:")
            report.append(f"  R² Score: {basic.get('r2_score', 0):.4f}")
            report.append(f"  Mean Squared Error: {basic.get('mse', 0):.4f}")
            report.append(f"  Root Mean Squared Error: {basic.get('rmse', 0):.4f}")
            report.append(f"  Mean Absolute Error: {basic.get('mae', 0):.4f}")
            report.append("")
            
            # Position metrics
            position = evaluation_results.get('position_metrics', {})
            report.append("POSITION ACCURACY:")
            report.append(f"  Overall Accuracy: {position.get('overall_accuracy', 0):.2%}")
            report.append(f"  Top 5 Accuracy: {position.get('top_5_accuracy', 0):.2%}")
            report.append(f"  Top 10 Accuracy: {position.get('top_10_accuracy', 0):.2%}")
            report.append("")
            
            # Driver performance summary
            driver_perf = evaluation_results.get('driver_performance', {})
            if driver_perf:
                report.append("DRIVER PERFORMANCE SUMMARY:")
                total_exact = sum(stats.get('exact_matches', 0) for stats in driver_perf.values())
                total_predictions = sum(stats.get('count', 0) for stats in driver_perf.values())
                if total_predictions > 0:
                    report.append(f"  Overall Exact Match Rate: {total_exact/total_predictions:.2%}")
                report.append("")
            
            report.append("=" * 60)
            
            return "\n".join(report)
            
        except Exception as e:
            logger.error(f"Error generating performance report: {e}")
            return f"Error generating report: {e}"
    
    def plot_performance_metrics(self, evaluation_results: Dict, save_path: Optional[str] = None):
        """Create visualization plots for performance metrics."""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('F1 Race Prediction Model Performance Analysis', fontsize=16)
            
            # 1. Position accuracy distribution
            driver_perf = evaluation_results.get('driver_performance', {})
            if driver_perf:
                positions = list(driver_perf.keys())
                accuracies = [driver_perf[pos].get('exact_matches', 0) / max(driver_perf[pos].get('count', 1), 1) 
                            for pos in positions]
                
                axes[0, 0].bar(positions, accuracies)
                axes[0, 0].set_title('Position Prediction Accuracy by Driver Position')
                axes[0, 0].set_xlabel('Driver Position')
                axes[0, 0].set_ylabel('Accuracy')
                axes[0, 0].grid(True, alpha=0.3)
            
            # 2. Error distribution
            basic_metrics = evaluation_results.get('basic_metrics', {})
            if 'mae' in basic_metrics:
                axes[0, 1].text(0.5, 0.5, f"MAE: {basic_metrics['mae']:.3f}\nRMSE: {basic_metrics.get('rmse', 0):.3f}", 
                               ha='center', va='center', transform=axes[0, 1].transAxes, fontsize=12)
                axes[0, 1].set_title('Error Metrics')
                axes[0, 1].axis('off')
            
            # 3. R² Score
            if 'r2_score' in basic_metrics:
                axes[1, 0].pie([basic_metrics['r2_score'], 1 - basic_metrics['r2_score']], 
                              labels=['Explained Variance', 'Unexplained Variance'],
                              autopct='%1.1f%%', startangle=90)
                axes[1, 0].set_title('R² Score Breakdown')
            
            # 4. Position accuracy summary
            position_metrics = evaluation_results.get('position_metrics', {})
            if position_metrics:
                metrics = ['Overall', 'Top 5', 'Top 10']
                values = [
                    position_metrics.get('overall_accuracy', 0),
                    position_metrics.get('top_5_accuracy', 0),
                    position_metrics.get('top_10_accuracy', 0)
                ]
                
                axes[1, 1].bar(metrics, values)
                axes[1, 1].set_title('Position Accuracy Summary')
                axes[1, 1].set_ylabel('Accuracy')
                axes[1, 1].set_ylim(0, 1)
                for i, v in enumerate(values):
                    axes[1, 1].text(i, v + 0.01, f'{v:.2%}', ha='center')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Performance plots saved to {save_path}")
            else:
                plt.show()
            
        except Exception as e:
            logger.error(f"Error creating performance plots: {e}")
    
    def get_model_stats(self) -> Dict:
        """Get overall model statistics."""
        try:
            stats = {
                'total_evaluations': len(self.evaluation_history),
                'last_evaluation': None,
                'best_performance': None,
                'average_performance': None
            }
            
            if self.evaluation_history:
                stats['last_evaluation'] = self.evaluation_history[-1].get('timestamp')
                
                # Calculate best and average performance
                r2_scores = [eval.get('basic_metrics', {}).get('r2_score', 0) 
                           for eval in self.evaluation_history]
                
                if r2_scores:
                    stats['best_performance'] = max(r2_scores)
                    stats['average_performance'] = np.mean(r2_scores)
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting model stats: {e}")
            return {'error': str(e)}
    
    def get_performance_metrics(self) -> Dict:
        """Get current performance metrics."""
        if not self.evaluation_history:
            return {'status': 'No evaluation data available'}
        
        return self.evaluation_history[-1]
    
    def get_historical_accuracy(self) -> List[Dict]:
        """Get historical accuracy trends."""
        return self.evaluation_history
    
    def _save_evaluation_results(self, results: Dict):
        """Save evaluation results to file."""
        try:
            self.evaluation_history.append(results)
            
            # Save to file
            results_file = os.path.join(self.results_path, f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            # Save history
            history_file = os.path.join(self.results_path, "evaluation_history.json")
            with open(history_file, 'w') as f:
                json.dump(self.evaluation_history, f, indent=2)
            
            logger.info(f"Evaluation results saved to {results_file}")
            
        except Exception as e:
            logger.error(f"Error saving evaluation results: {e}") 