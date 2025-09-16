"""
Interface evaluator for comprehensive beauty assessment.
"""

import torch
import torch.nn as nn
import numpy as np
import os
from typing import Dict, List, Optional, Tuple
import json

# Handle optional imports
try:
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    from scipy.stats import pearsonr, spearmanr
except ImportError:
    from ..utils.fallbacks import mean_squared_error, mean_absolute_error, pearsonr, spearmanr

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError:
    from ..utils.fallbacks import plt, sns

try:
    from tqdm import tqdm
except ImportError:
    from ..utils.fallbacks import tqdm

from ..models import BeautyPredictor


class InterfaceEvaluator:
    """
    Comprehensive evaluator for interface beauty models.
    
    Provides evaluation metrics, visualization tools, and analysis
    capabilities for beauty prediction models.
    """
    
    def __init__(self, model: BeautyPredictor, device: str = 'cuda'):
        """
        Initialize evaluator.
        
        Args:
            model: Trained beauty prediction model
            device: Device to run evaluation on
        """
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()
        
        # Evaluation results storage
        self.results = {}
        
    def evaluate_dataset(self, dataloader, save_dir: Optional[str] = None) -> Dict:
        """
        Evaluate model on a dataset.
        
        Args:
            dataloader: DataLoader for evaluation dataset
            save_dir: Optional directory to save results
            
        Returns:
            Dictionary with evaluation metrics
        """
        predictions = []
        ground_truth = []
        aspect_predictions = {}
        aspect_ground_truth = {}
        principle_predictions = {}
        principle_ground_truth = {}
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                images = batch['image'].to(self.device)
                beauty_scores = batch['beauty_score'].cpu().numpy()
                
                # Get model predictions
                outputs = self.model(
                    images,
                    return_aspects=True,
                    return_principles=True,
                )
                
                pred_beauty = torch.sigmoid(outputs[0]).cpu().numpy()
                predictions.extend(pred_beauty.flatten())
                ground_truth.extend(beauty_scores.flatten())
                
                # Collect aspect predictions
                if len(outputs) > 1 and outputs[1]:
                    for aspect, values in outputs[1].items():
                        if aspect not in aspect_predictions:
                            aspect_predictions[aspect] = []
                            aspect_ground_truth[aspect] = []
                        
                        pred_aspect = torch.sigmoid(values).cpu().numpy()
                        aspect_predictions[aspect].extend(pred_aspect.flatten())
                        
                        if 'beauty_aspects' in batch and aspect in batch['beauty_aspects']:
                            gt_aspect = batch['beauty_aspects'][aspect].cpu().numpy()
                            aspect_ground_truth[aspect].extend(gt_aspect.flatten())
                
                # Collect principle predictions
                if len(outputs) > 2 and outputs[2]:
                    for principle, values in outputs[2].items():
                        if principle not in principle_predictions:
                            principle_predictions[principle] = []
                            principle_ground_truth[principle] = []
                        
                        pred_principle = torch.sigmoid(values).cpu().numpy()
                        principle_predictions[principle].extend(pred_principle.flatten())
                        
                        if 'design_principles' in batch and principle in batch['design_principles']:
                            gt_principle = batch['design_principles'][principle].cpu().numpy()
                            principle_ground_truth[principle].extend(gt_principle.flatten())
        
        # Calculate metrics
        results = self._calculate_metrics(
            predictions, ground_truth,
            aspect_predictions, aspect_ground_truth,
            principle_predictions, principle_ground_truth
        )
        
        # Store results
        self.results = results
        
        # Save results if directory provided
        if save_dir:
            self._save_results(results, save_dir)
        
        return results
    
    def _calculate_metrics(self, pred, gt, aspect_pred, aspect_gt, principle_pred, principle_gt):
        """Calculate comprehensive evaluation metrics."""
        results = {}
        
        # Overall beauty score metrics
        pred_arr = np.array(pred)
        gt_arr = np.array(gt)
        
        results['overall'] = {
            'mse': mean_squared_error(gt_arr, pred_arr),
            'mae': mean_absolute_error(gt_arr, pred_arr),
            'rmse': np.sqrt(mean_squared_error(gt_arr, pred_arr)),
            'pearson': pearsonr(gt_arr, pred_arr)[0],
            'spearman': spearmanr(gt_arr, pred_arr)[0],
            'predictions': pred_arr,
            'ground_truth': gt_arr,
        }
        
        # Beauty aspect metrics
        results['aspects'] = {}
        for aspect in aspect_pred:
            if aspect in aspect_gt and len(aspect_gt[aspect]) > 0:
                pred_asp = np.array(aspect_pred[aspect])
                gt_asp = np.array(aspect_gt[aspect])
                
                results['aspects'][aspect] = {
                    'mse': mean_squared_error(gt_asp, pred_asp),
                    'mae': mean_absolute_error(gt_asp, pred_asp),
                    'pearson': pearsonr(gt_asp, pred_asp)[0],
                    'spearman': spearmanr(gt_asp, pred_asp)[0],
                }
        
        # Design principle metrics
        results['principles'] = {}
        for principle in principle_pred:
            if principle in principle_gt and len(principle_gt[principle]) > 0:
                pred_pri = np.array(principle_pred[principle])
                gt_pri = np.array(principle_gt[principle])
                
                results['principles'][principle] = {
                    'mse': mean_squared_error(gt_pri, pred_pri),
                    'mae': mean_absolute_error(gt_pri, pred_pri),
                    'pearson': pearsonr(gt_pri, pred_pri)[0],
                    'spearman': spearmanr(gt_pri, pred_pri)[0],
                }
        
        return results
    
    def _save_results(self, results: Dict, save_dir: str):
        """Save evaluation results to directory."""
        os.makedirs(save_dir, exist_ok=True)
        
        # Save numerical results
        import json
        with open(os.path.join(save_dir, 'metrics.json'), 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            json_results = {}
            for key, value in results.items():
                if key == 'overall':
                    json_results[key] = {k: float(v) if isinstance(v, (np.float64, np.float32)) 
                                       else v.tolist() if isinstance(v, np.ndarray) else v 
                                       for k, v in value.items()}
                else:
                    json_results[key] = {
                        sub_key: {k: float(v) if isinstance(v, (np.float64, np.float32)) else v 
                                 for k, v in sub_value.items()}
                        for sub_key, sub_value in value.items()
                    }
            json.dump(json_results, f, indent=2)
        
        # Create visualizations
        self._create_visualizations(results, save_dir)
    
    def _create_visualizations(self, results: Dict, save_dir: str):
        """Create evaluation visualizations."""
        # Prediction vs ground truth scatter plot
        if 'overall' in results:
            plt.figure(figsize=(10, 8))
            
            pred = results['overall']['predictions']
            gt = results['overall']['ground_truth']
            
            plt.scatter(gt, pred, alpha=0.6, s=30)
            plt.plot([0, 1], [0, 1], 'r--', linewidth=2)
            plt.xlabel('Ground Truth Beauty Score')
            plt.ylabel('Predicted Beauty Score')
            plt.title(f'Prediction vs Ground Truth\nPearson: {results["overall"]["pearson"]:.3f}, RMSE: {results["overall"]["rmse"]:.3f}')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'prediction_scatter.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            # Distribution plots
            plt.figure(figsize=(12, 5))
            
            plt.subplot(1, 2, 1)
            plt.hist(gt, bins=30, alpha=0.7, label='Ground Truth', color='blue')
            plt.hist(pred, bins=30, alpha=0.7, label='Predictions', color='red')
            plt.xlabel('Beauty Score')
            plt.ylabel('Frequency')
            plt.title('Score Distributions')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.subplot(1, 2, 2)
            residuals = pred - gt
            plt.hist(residuals, bins=30, alpha=0.7, color='green')
            plt.xlabel('Prediction Error')
            plt.ylabel('Frequency')
            plt.title('Residuals Distribution')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'distributions.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
        # Aspect/Principle performance comparison
        if 'aspects' in results and results['aspects']:
            aspects_metrics = []
            aspect_names = []
            
            for aspect, metrics in results['aspects'].items():
                aspect_names.append(aspect)
                aspects_metrics.append([metrics['mae'], metrics['pearson']])
            
            if aspects_metrics:
                plt.figure(figsize=(12, 6))
                
                plt.subplot(1, 2, 1)
                mae_scores = [m[0] for m in aspects_metrics]
                plt.bar(aspect_names, mae_scores, color='skyblue')
                plt.xlabel('Beauty Aspects')
                plt.ylabel('Mean Absolute Error')
                plt.title('Aspect Prediction MAE')
                plt.xticks(rotation=45)
                plt.grid(True, alpha=0.3)
                
                plt.subplot(1, 2, 2)
                corr_scores = [m[1] for m in aspects_metrics]
                plt.bar(aspect_names, corr_scores, color='lightcoral')
                plt.xlabel('Beauty Aspects')
                plt.ylabel('Pearson Correlation')
                plt.title('Aspect Prediction Correlation')
                plt.xticks(rotation=45)
                plt.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, 'aspects_performance.png'), dpi=300, bbox_inches='tight')
                plt.close()
        
    def analyze_predictions(self, dataloader, n_samples: int = 10, save_dir: Optional[str] = None):
        """
        Analyze individual predictions with visualizations.
        
        Args:
            dataloader: DataLoader for analysis
            n_samples: Number of samples to analyze
            save_dir: Directory to save analysis results
        """
        analyses = []
        
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if i >= n_samples:
                    break
                
                images = batch['image'].to(self.device)
                
                # Get comprehensive analysis
                for j in range(len(images)):
                    image = images[j:j+1]
                    analysis = self.model.analyze_interface(image)
                    
                    # Add ground truth if available
                    if 'beauty_score' in batch:
                        analysis['ground_truth_beauty'] = batch['beauty_score'][j].item()
                    
                    # Add image for visualization
                    analysis['image'] = images[j].cpu()
                    analysis['image_path'] = batch.get('image_path', [f'sample_{i}_{j}'])[j]
                    
                    analyses.append(analysis)
        
        if save_dir:
            self._save_analysis(analyses, save_dir)
        
        return analyses
    
    def _save_analysis(self, analyses: List[Dict], save_dir: str):
        """Save detailed analysis results."""
        os.makedirs(save_dir, exist_ok=True)
        
        for i, analysis in enumerate(analyses):
            sample_dir = os.path.join(save_dir, f'sample_{i:03d}')
            os.makedirs(sample_dir, exist_ok=True)
            
            # Save analysis text
            explanation = self.model.get_beauty_explanation(analysis)
            with open(os.path.join(sample_dir, 'analysis.txt'), 'w') as f:
                f.write(f"Image: {analysis['image_path']}\n")
                f.write(f"Predicted Beauty Score: {analysis['overall_beauty_score']:.3f}\n")
                if 'ground_truth_beauty' in analysis:
                    f.write(f"Ground Truth Score: {analysis['ground_truth_beauty']:.3f}\n")
                    f.write(f"Prediction Error: {abs(analysis['overall_beauty_score'] - analysis['ground_truth_beauty']):.3f}\n")
                f.write(f"\n{explanation}")
            
            # Save saliency maps if available
            if 'saliency_maps' in analysis:
                saliency = analysis['saliency_maps'][0, 0]  # First image, first channel
                plt.figure(figsize=(8, 6))
                plt.imshow(saliency, cmap='hot', interpolation='bilinear')
                plt.colorbar()
                plt.title('Saliency Map')
                plt.axis('off')
                plt.savefig(os.path.join(sample_dir, 'saliency.png'), dpi=300, bbox_inches='tight')
                plt.close()
    
    def compare_models(self, other_evaluators: List['InterfaceEvaluator'], 
                      dataloader, model_names: List[str]) -> Dict:
        """
        Compare multiple models on the same dataset.
        
        Args:
            other_evaluators: List of other evaluators to compare
            dataloader: Dataset to evaluate on
            model_names: Names for the models
            
        Returns:
            Comparison results
        """
        all_evaluators = [self] + other_evaluators
        all_results = []
        
        for evaluator in all_evaluators:
            results = evaluator.evaluate_dataset(dataloader)
            all_results.append(results)
        
        # Create comparison table
        comparison = {
            'models': model_names,
            'metrics': {}
        }
        
        metrics_to_compare = ['mse', 'mae', 'rmse', 'pearson', 'spearman']
        for metric in metrics_to_compare:
            comparison['metrics'][metric] = [
                results['overall'][metric] for results in all_results
            ]
        
        return comparison
    
    def get_summary_report(self) -> str:
        """Generate a summary report of evaluation results."""
        if not self.results:
            return "No evaluation results available. Run evaluate_dataset() first."
        
        report = "=== Interface Beauty Evaluation Report ===\n\n"
        
        # Overall performance
        overall = self.results['overall']
        report += f"Overall Performance:\n"
        report += f"  RMSE: {overall['rmse']:.4f}\n"
        report += f"  MAE: {overall['mae']:.4f}\n"
        report += f"  Pearson Correlation: {overall['pearson']:.4f}\n"
        report += f"  Spearman Correlation: {overall['spearman']:.4f}\n\n"
        
        # Beauty aspects performance
        if 'aspects' in self.results and self.results['aspects']:
            report += "Beauty Aspects Performance:\n"
            for aspect, metrics in self.results['aspects'].items():
                report += f"  {aspect.capitalize()}:\n"
                report += f"    MAE: {metrics['mae']:.4f}\n"
                report += f"    Pearson: {metrics['pearson']:.4f}\n"
            report += "\n"
        
        # Design principles performance
        if 'principles' in self.results and self.results['principles']:
            report += "Design Principles Performance:\n"
            for principle, metrics in self.results['principles'].items():
                report += f"  {principle.capitalize()}:\n"
                report += f"    MAE: {metrics['mae']:.4f}\n"
                report += f"    Pearson: {metrics['pearson']:.4f}\n"
        
        return report


class BeautyBenchmark:
    """Benchmarking suite for beauty evaluation models."""
    
    def __init__(self):
        self.benchmark_datasets = []
        self.benchmark_results = {}
    
    def add_dataset(self, name: str, dataloader):
        """Add a dataset to the benchmark suite."""
        self.benchmark_datasets.append((name, dataloader))
    
    def run_benchmark(self, model: BeautyPredictor, save_dir: str):
        """Run benchmark on all datasets."""
        evaluator = InterfaceEvaluator(model)
        
        for dataset_name, dataloader in self.benchmark_datasets:
            print(f"Evaluating on {dataset_name}...")
            dataset_save_dir = os.path.join(save_dir, dataset_name)
            results = evaluator.evaluate_dataset(dataloader, dataset_save_dir)
            self.benchmark_results[dataset_name] = results
        
        # Create summary report
        self._create_benchmark_report(save_dir)
    
    def _create_benchmark_report(self, save_dir: str):
        """Create comprehensive benchmark report."""
        report_path = os.path.join(save_dir, 'benchmark_report.txt')
        
        with open(report_path, 'w') as f:
            f.write("=== Beauty Evaluation Benchmark Report ===\n\n")
            
            for dataset_name, results in self.benchmark_results.items():
                f.write(f"Dataset: {dataset_name}\n")
                f.write(f"  RMSE: {results['overall']['rmse']:.4f}\n")
                f.write(f"  Pearson: {results['overall']['pearson']:.4f}\n")
                f.write("\n")
        
        print(f"Benchmark report saved to {report_path}")