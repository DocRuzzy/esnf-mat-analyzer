#!/usr/bin/env python3
"""
Example demonstration of multiscale uniformity analysis concepts.

This script creates synthetic examples to illustrate different types of 
uniformity problems and how they appear at different scales.
"""

import numpy as np
import matplotlib.pyplot as plt
from esnf_mat_analyzer.analysis.multiscale_uniformity import MultiScaleUniformityAnalyzer

def create_example_mats():
    """Create synthetic mat examples with different uniformity characteristics."""
    
    examples = {}
    size = 200
    
    # 1. Ideal uniform mat
    uniform_mat = 2.0 + 0.05 * np.random.randn(size, size)
    uniform_mat = np.clip(uniform_mat, 1.5, 2.5)
    examples['Ideal Uniform'] = uniform_mat
    
    # 2. Large-scale gradient (Scale 0 problem)
    y, x = np.ogrid[:size, :size]
    gradient_mat = 1.5 + 1.0 * (x / size) + 0.05 * np.random.randn(size, size)
    examples['Large-scale Gradient'] = gradient_mat
    
    # 3. Fine texture variations (Scale 1-2 problem)
    texture_mat = 2.0 + 0.3 * np.sin(x * 0.3) * np.sin(y * 0.3) + 0.05 * np.random.randn(size, size)
    examples['Fine Texture Variation'] = texture_mat
    
    # 4. Medium-scale patches (Scale 3 problem)
    patches_mat = 2.0 * np.ones((size, size))
    # Add circular patches of different thickness
    center1 = (size//3, size//3)
    center2 = (2*size//3, 2*size//3)
    center3 = (size//3, 2*size//3)
    
    for cy, cx in [center1, center2, center3]:
        y_dist = (y - cy)**2
        x_dist = (x - cx)**2
        distance = np.sqrt(y_dist + x_dist)
        patch = np.exp(-distance / 20)
        patches_mat += 0.8 * patch
    
    patches_mat += 0.05 * np.random.randn(size, size)
    examples['Medium-scale Patches'] = patches_mat
    
    # 5. Multi-scale problems (poor at all scales)
    multi_problem = gradient_mat + 0.5 * texture_mat + 0.3 * patches_mat
    multi_problem += 0.1 * np.random.randn(size, size)
    examples['Multi-scale Problems'] = multi_problem
    
    return examples

def create_roi_mask(size):
    """Create a circular ROI mask."""
    y, x = np.ogrid[:size, :size]
    center = size // 2
    radius = size // 2 - 20
    distance = np.sqrt((x - center)**2 + (y - center)**2)
    return distance <= radius

def analyze_examples():
    """Analyze all example mats and display results."""
    
    print("Multiscale Uniformity Analysis - Example Demonstration")
    print("=" * 60)
    print()
    
    # Create examples
    examples = create_example_mats()
    roi_mask = create_roi_mask(200)
    
    # Initialize analyzer
    analyzer = MultiScaleUniformityAnalyzer(wavelet='db4', levels=4)
    
    # Analyze each example
    results = {}
    for name, mat in examples.items():
        print(f"Analyzing: {name}")
        result = analyzer.analyze_multiscale_uniformity(mat, roi_mask)
        results[name] = result
        
        # Print scale-by-scale breakdown
        print(f"  Results:")
        for scale_key, value in result.items():
            scale_num = scale_key.replace('scale_', '').replace('_uniformity', '')
            scale_type = "Large-scale (trends)" if scale_num == '0' else f"Detail scale {scale_num}"
            print(f"    Scale {scale_num}: {value:.3f} - {scale_type}")
        
        overall = np.mean(list(result.values()))
        interpretation = get_interpretation(overall)
        print(f"  Overall Score: {overall:.3f} - {interpretation}")
        print()
    
    # Create visualization
    create_comparison_visualization(examples, results, roi_mask)

def get_interpretation(score):
    """Provide interpretation of uniformity score."""
    if score >= 0.9:
        return "Excellent uniformity"
    elif score >= 0.8:
        return "Good uniformity"
    elif score >= 0.7:
        return "Moderate uniformity"
    elif score >= 0.6:
        return "Fair uniformity"
    else:
        return "Poor uniformity"

def create_comparison_visualization(examples, results, roi_mask):
    """Create comprehensive visualization of examples and results."""
    
    n_examples = len(examples)
    fig, axes = plt.subplots(3, n_examples, figsize=(4*n_examples, 12))
    
    if n_examples == 1:
        axes = axes.reshape(-1, 1)
    
    example_names = list(examples.keys())
    
    # Row 1: Original thickness maps
    for i, (name, mat) in enumerate(examples.items()):
        ax = axes[0, i]
        im = ax.imshow(mat, cmap='viridis', aspect='equal')
        ax.set_title(f'{name}\\nThickness Map', fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im, ax=ax, shrink=0.8, label='Thickness')
    
    # Row 2: ROI-masked thickness maps
    for i, (name, mat) in enumerate(examples.items()):
        ax = axes[1, i]
        masked_mat = np.ma.masked_array(mat, mask=~roi_mask)
        im = ax.imshow(masked_mat, cmap='viridis', aspect='equal')
        ax.set_title(f'ROI-Masked\\nThickness', fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im, ax=ax, shrink=0.8, label='Thickness')
    
    # Row 3: Scale analysis results
    for i, name in enumerate(example_names):
        ax = axes[2, i]
        result = results[name]
        
        scale_names = [f'S{k.replace("scale_", "").replace("_uniformity", "")}' for k in result.keys()]
        scale_values = list(result.values())
        
        bars = ax.bar(scale_names, scale_values, 
                     color=['lightcoral' if v < 0.7 else 'lightblue' if v < 0.9 else 'lightgreen' 
                           for v in scale_values],
                     edgecolor='black', linewidth=0.5)
        
        ax.set_title(f'Scale Uniformity\\nScores', fontsize=10)
        ax.set_ylabel('Uniformity Score')
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, scale_values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                   f'{value:.2f}', ha='center', va='bottom', fontsize=8)
        
        # Add overall score
        overall = np.mean(scale_values)
        ax.text(0.5, 0.95, f'Overall: {overall:.3f}', transform=ax.transAxes, 
               ha='center', va='top', fontweight='bold',
               bbox=dict(boxstyle="round,pad=0.3", 
                        facecolor='lightgreen' if overall >= 0.8 else 'orange' if overall >= 0.6 else 'pink',
                        alpha=0.7))
    
    plt.suptitle('Multiscale Uniformity Analysis Examples\\n' +
                'Green bars: Excellent (>0.9), Blue bars: Good (0.7-0.9), Red bars: Poor (<0.7)', 
                fontsize=14)
    plt.tight_layout()
    plt.savefig('multiscale_examples_comparison.png', dpi=150, bbox_inches='tight')
    print("Visualization saved to: multiscale_examples_comparison.png")
    plt.show()

def explain_scale_problems():
    """Explain what each type of scale problem represents physically."""
    
    print("\\n" + "="*60)
    print("PHYSICAL INTERPRETATION OF SCALE PROBLEMS")
    print("="*60)
    print()
    
    explanations = {
        "Scale 0 (Large-scale) Problems": [
            "• Overall thickness gradients across the mat",
            "• Non-uniform substrate temperature",
            "• Varying collector distance",
            "• Edge effects from electric field distortion",
            "• Uneven solution flow rate over time"
        ],
        
        "Scale 1-2 (Fine-scale) Problems": [
            "• Individual fiber diameter variations",
            "• Inconsistent fiber density/packing",
            "• Surface roughness variations",
            "• Local charge density fluctuations",
            "• Nozzle tip irregularities"
        ],
        
        "Scale 3-4 (Medium-scale) Problems": [
            "• Local thickness patches or 'lumps'",
            "• Fiber bundle formation",
            "• Processing defects (drops, clogs)",
            "• Local air flow disturbances",
            "• Substrate surface irregularities"
        ]
    }
    
    for category, issues in explanations.items():
        print(f"{category}:")
        for issue in issues:
            print(f"  {issue}")
        print()
    
    print("PROCESS OPTIMIZATION GUIDANCE:")
    print("-" * 30)
    print("• Poor Scale 0: Check temperature uniformity, collector alignment")
    print("• Poor Scale 1-2: Check solution properties, nozzle condition")  
    print("• Poor Scale 3-4: Check air flow, substrate preparation")
    print("• Poor all scales: Fundamental process issues, review all parameters")

if __name__ == "__main__":
    analyze_examples()
    explain_scale_problems()
    
    print("\\n" + "="*60)
    print("For complete documentation, see: MULTISCALE_UNIFORMITY_GUIDE.md")
    print("="*60)
