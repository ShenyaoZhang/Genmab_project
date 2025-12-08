"""
NLP Analysis Visualizations for PowerPoint Slides
Generates publication-ready figures for CRS narrative analysis
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import json
import os

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14


def create_nlp_pipeline_diagram(save_path='figures/nlp_pipeline.png'):
    """Create a visual diagram of the NLP pipeline."""
    
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # Colors
    colors = {
        'input': '#64B5F6',      # Blue
        'process': '#81C784',    # Green
        'features': '#FFB74D',   # Orange
        'output': '#EF5350',     # Red
        'optional': '#BA68C8',   # Purple
    }
    
    # Step 1: Input
    box1 = FancyBboxPatch((0.5, 5.5), 3, 1.5, boxstyle="round,pad=0.05,rounding_size=0.2",
                          facecolor=colors['input'], edgecolor='black', linewidth=2)
    ax.add_patch(box1)
    ax.text(2, 6.25, 'INPUT\nNarrative Text\n(FAERS reports)', ha='center', va='center', 
            fontsize=11, fontweight='bold', color='white')
    
    # Arrow 1
    ax.annotate('', xy=(4, 6.25), xytext=(3.5, 6.25),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))
    
    # Step 2: Rule-based extraction
    box2 = FancyBboxPatch((4.2, 5), 4, 2.5, boxstyle="round,pad=0.05,rounding_size=0.2",
                          facecolor=colors['process'], edgecolor='black', linewidth=2)
    ax.add_patch(box2)
    ax.text(6.2, 6.8, 'RULE-BASED EXTRACTION', ha='center', va='center', 
            fontsize=11, fontweight='bold', color='white')
    ax.text(6.2, 6.1, '• Fever, hypotension, hypoxia\n• ICU, intubation, vasopressors\n• Tocilizumab, steroids\n• CRS grade (1-4)\n• Time to onset', 
            ha='center', va='center', fontsize=9, color='white')
    
    # Arrow 2
    ax.annotate('', xy=(8.7, 6.25), xytext=(8.2, 6.25),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))
    
    # Step 3: Features
    box3 = FancyBboxPatch((9, 5.5), 3.5, 1.5, boxstyle="round,pad=0.05,rounding_size=0.2",
                          facecolor=colors['features'], edgecolor='black', linewidth=2)
    ax.add_patch(box3)
    ax.text(10.75, 6.25, 'EXTRACTED FEATURES\nSeverity Score\n(0-5 scale)', ha='center', va='center', 
            fontsize=11, fontweight='bold', color='white')
    
    # Optional BERT path
    box_bert = FancyBboxPatch((4.2, 2), 4, 2, boxstyle="round,pad=0.05,rounding_size=0.2",
                              facecolor=colors['optional'], edgecolor='black', linewidth=2, linestyle='--')
    ax.add_patch(box_bert)
    ax.text(6.2, 3.5, 'BERT EMBEDDINGS (Optional)', ha='center', va='center', 
            fontsize=11, fontweight='bold', color='white')
    ax.text(6.2, 2.7, 'BioBERT / ClinicalBERT\n768-dim [CLS] token', 
            ha='center', va='center', fontsize=9, color='white')
    
    # Arrows for BERT path
    ax.annotate('', xy=(4.2, 3), xytext=(2, 5.5),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.5, linestyle='--'))
    ax.annotate('', xy=(9, 3), xytext=(8.2, 3),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.5, linestyle='--'))
    
    # BERT features box
    box_bert_feat = FancyBboxPatch((9, 2.25), 3.5, 1.5, boxstyle="round,pad=0.05,rounding_size=0.2",
                                   facecolor=colors['optional'], edgecolor='black', linewidth=2, linestyle='--')
    ax.add_patch(box_bert_feat)
    ax.text(10.75, 3, 'BERT FEATURES\n768 dimensions', ha='center', va='center', 
            fontsize=10, fontweight='bold', color='white')
    
    # Arrow to model
    ax.annotate('', xy=(10.75, 5.3), xytext=(10.75, 4),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))
    ax.annotate('', xy=(10.75, 3.75), xytext=(10.75, 3.75),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.5, linestyle='--'))
    
    # Merge point
    ax.plot(10.75, 4.5, 'ko', markersize=10)
    
    # Arrow down to model
    ax.annotate('', xy=(10.75, 1.5), xytext=(10.75, 4.3),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))
    
    # Step 4: Model
    box4 = FancyBboxPatch((9, 0.3), 3.5, 1.2, boxstyle="round,pad=0.05,rounding_size=0.2",
                          facecolor=colors['output'], edgecolor='black', linewidth=2)
    ax.add_patch(box4)
    ax.text(10.75, 0.9, 'SEVERITY CLASSIFIER\nGradient Boosting\nPredict: Severe CRS (Y/N)', 
            ha='center', va='center', fontsize=10, fontweight='bold', color='white')
    
    # Legend
    legend_items = [
        ('Input Data', colors['input']),
        ('Feature Extraction', colors['process']),
        ('Extracted Features', colors['features']),
        ('Prediction Model', colors['output']),
        ('Optional (BERT)', colors['optional']),
    ]
    
    for i, (label, color) in enumerate(legend_items):
        ax.add_patch(plt.Rectangle((0.5, 3.8 - i*0.5), 0.4, 0.35, facecolor=color, edgecolor='black'))
        ax.text(1.05, 3.95 - i*0.5, label, va='center', fontsize=9)
    
    ax.set_title('NLP Pipeline for CRS Narrative Analysis', fontsize=16, fontweight='bold', pad=20)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")


def create_feature_extraction_results(save_path='figures/nlp_feature_results.png'):
    """Create visualization of what features were extracted from narratives."""
    
    # Load narrative features if available
    features_path = 'narrative_features.json'
    if os.path.exists(features_path):
        with open(features_path, 'r') as f:
            features_data = json.load(f)
    else:
        # Use placeholder data
        features_data = [{'has_narrative': True, 'mentions_fever': False, 
                         'mentions_hypotension': False, 'mentions_hypoxia': False,
                         'mentions_icu': False, 'mentions_intubation': False,
                         'mentions_vasopressor': False, 'mentions_tocilizumab': False,
                         'mentions_steroids': False}] * 100
    
    # Calculate extraction rates
    feature_names = [
        ('mentions_fever', 'Fever'),
        ('mentions_hypotension', 'Hypotension'),
        ('mentions_hypoxia', 'Hypoxia'),
        ('mentions_icu', 'ICU Admission'),
        ('mentions_intubation', 'Intubation'),
        ('mentions_vasopressor', 'Vasopressor'),
        ('mentions_tocilizumab', 'Tocilizumab'),
        ('mentions_steroids', 'Steroids'),
    ]
    
    extraction_rates = []
    labels = []
    for key, label in feature_names:
        rate = sum(1 for r in features_data if r.get(key, False)) / len(features_data) * 100
        extraction_rates.append(rate)
        labels.append(label)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left panel: Feature extraction rates
    colors = ['#EF5350' if r < 5 else '#FFB74D' if r < 20 else '#81C784' for r in extraction_rates]
    bars = ax1.barh(range(len(labels)), extraction_rates, color=colors, edgecolor='black', height=0.6)
    
    ax1.set_yticks(range(len(labels)))
    ax1.set_yticklabels(labels, fontsize=11)
    ax1.set_xlabel('% of Narratives Mentioning Feature', fontsize=12)
    ax1.set_title('Feature Extraction Results\nfrom FAERS Narratives', fontsize=14, fontweight='bold')
    ax1.set_xlim(0, max(extraction_rates) + 10 if max(extraction_rates) > 0 else 100)
    
    # Add percentage labels
    for bar, rate in zip(bars, extraction_rates):
        ax1.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, 
                f'{rate:.1f}%', va='center', fontsize=10)
    
    # Add note if all rates are low
    if max(extraction_rates) < 5:
        ax1.text(0.5, 0.5, 'Note: Very low extraction rates\ndue to short/empty narratives', 
                transform=ax1.transAxes, ha='center', va='center', fontsize=12,
                bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='orange', alpha=0.9))
    
    # Right panel: Narrative length distribution
    narrative_lengths = [r.get('narrative_length', 0) for r in features_data]
    
    ax2.hist(narrative_lengths, bins=20, color='#64B5F6', edgecolor='black', alpha=0.8)
    ax2.axvline(x=np.mean(narrative_lengths), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(narrative_lengths):.0f} chars')
    ax2.axvline(x=100, color='green', linestyle='--', linewidth=2, label='Min useful length (~100 chars)')
    
    ax2.set_xlabel('Narrative Length (characters)', fontsize=12)
    ax2.set_ylabel('Number of Reports', fontsize=12)
    ax2.set_title('Distribution of Narrative Lengths', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=10)
    
    # Add interpretation
    short_narratives = sum(1 for l in narrative_lengths if l < 50)
    pct_short = short_narratives / len(narrative_lengths) * 100
    ax2.text(0.95, 0.95, f'{pct_short:.0f}% of narratives\nare < 50 characters\n(too short for NLP)', 
             transform=ax2.transAxes, ha='right', va='top', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='orange', alpha=0.9))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")


def create_model_performance_plot(save_path='figures/nlp_model_performance.png'):
    """Create visualization of NLP model performance."""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left panel: AUC comparison
    models = ['Random\nGuessing', 'Current NLP\nModel', 'Good Model\n(Target)']
    aucs = [0.50, 0.50, 0.80]  # Current model AUC is ~0.50
    colors = ['#90A4AE', '#EF5350', '#81C784']
    
    bars = ax1.bar(models, aucs, color=colors, edgecolor='black', width=0.6)
    ax1.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    ax1.set_ylabel('ROC-AUC Score', fontsize=12)
    ax1.set_ylim(0, 1)
    ax1.set_title('Model Discrimination Ability', fontsize=14, fontweight='bold')
    
    for bar, auc in zip(bars, aucs):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{auc:.2f}', ha='center', fontsize=12, fontweight='bold')
    
    # Add annotation
    ax1.annotate('Current model = random\n(no predictive value)', 
                xy=(1, 0.50), xytext=(1.5, 0.3),
                fontsize=10, ha='center',
                arrowprops=dict(arrowstyle='->', color='red'),
                bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='orange'))
    
    # Right panel: Feature importance (all zeros currently)
    features = ['Narrative\nLength', 'Severity\nScore', 'Fever', 'Hypotension', 'Hypoxia']
    importances = [0.0, 0.0, 0.0, 0.0, 0.0]  # All zeros
    
    bars2 = ax2.barh(range(len(features)), importances, color='#64B5F6', edgecolor='black', height=0.5)
    ax2.set_yticks(range(len(features)))
    ax2.set_yticklabels(features, fontsize=11)
    ax2.set_xlabel('Feature Importance', fontsize=12)
    ax2.set_xlim(0, 0.5)
    ax2.set_title('Top Feature Importances', fontsize=14, fontweight='bold')
    
    # Add note about zero importance
    ax2.text(0.5, 0.5, 'All importances = 0\n\nNo discriminative\nfeatures found\nin current narratives', 
             transform=ax2.transAxes, ha='center', va='center', fontsize=11,
             bbox=dict(boxstyle='round', facecolor='#FFCDD2', edgecolor='red', alpha=0.9))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")


def create_limitations_and_future(save_path='figures/nlp_limitations_future.png'):
    """Create visualization of current limitations and future directions."""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left panel: Current limitations
    ax1.axis('off')
    ax1.set_title('Current Limitations', fontsize=16, fontweight='bold', color='#EF5350')
    
    limitations = [
        ('Short Narratives', 'Average length ~25 chars\n(just dates/codes)'),
        ('Empty Content', 'Most narratives lack\nclinical descriptions'),
        ('No Signal', 'AUC = 0.50\n(random performance)'),
        ('Missing Data', 'No CRS grades or\nonset times extracted'),
    ]
    
    for i, (title, desc) in enumerate(limitations):
        y_pos = 0.85 - i * 0.22
        # Icon box
        ax1.add_patch(FancyBboxPatch((0.05, y_pos - 0.08), 0.12, 0.15, 
                                      boxstyle="round,pad=0.02", facecolor='#FFCDD2', 
                                      edgecolor='#EF5350', linewidth=2, transform=ax1.transAxes))
        ax1.text(0.11, y_pos, '!', transform=ax1.transAxes, fontsize=20, 
                ha='center', va='center', color='#EF5350', fontweight='bold')
        # Text
        ax1.text(0.22, y_pos + 0.02, title, transform=ax1.transAxes, fontsize=12, 
                fontweight='bold', va='center')
        ax1.text(0.22, y_pos - 0.06, desc, transform=ax1.transAxes, fontsize=10, 
                va='center', color='gray')
    
    # Right panel: Future directions
    ax2.axis('off')
    ax2.set_title('Future Directions', fontsize=16, fontweight='bold', color='#81C784')
    
    future_items = [
        ('Rich Clinical Notes', 'Full EHR narratives with\ndetailed symptom descriptions'),
        ('Literature Mining', 'Published case reports\nwith structured outcomes'),
        ('Multi-modal Fusion', 'Combine NLP features with\nstructured data (dose, labs)'),
        ('Advanced NLP', 'Fine-tuned BioBERT for\nCRS-specific classification'),
    ]
    
    for i, (title, desc) in enumerate(future_items):
        y_pos = 0.85 - i * 0.22
        # Icon box
        ax2.add_patch(FancyBboxPatch((0.05, y_pos - 0.08), 0.12, 0.15, 
                                      boxstyle="round,pad=0.02", facecolor='#C8E6C9', 
                                      edgecolor='#81C784', linewidth=2, transform=ax2.transAxes))
        ax2.text(0.11, y_pos, '→', transform=ax2.transAxes, fontsize=18, 
                ha='center', va='center', color='#81C784', fontweight='bold')
        # Text
        ax2.text(0.22, y_pos + 0.02, title, transform=ax2.transAxes, fontsize=12, 
                fontweight='bold', va='center')
        ax2.text(0.22, y_pos - 0.06, desc, transform=ax2.transAxes, fontsize=10, 
                va='center', color='gray')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")


def create_nlp_summary_figure(save_path='figures/nlp_summary.png'):
    """Create a 2-panel summary figure for NLP presentation."""
    
    fig = plt.figure(figsize=(16, 8))
    
    # Left panel: Pipeline diagram (simplified)
    ax1 = fig.add_subplot(121)
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.axis('off')
    ax1.set_title('A. NLP Pipeline & Methods', fontsize=14, fontweight='bold', loc='left')
    
    # Pipeline boxes
    steps = [
        (5, 9, 'Narrative Text Input', '#64B5F6'),
        (5, 7, 'Rule-Based Feature Extraction\n(fever, hypotension, ICU, etc.)', '#81C784'),
        (5, 5, 'Severity Score Calculation\n(count of ICU-level indicators)', '#FFB74D'),
        (5, 3, 'Gradient Boosting Classifier\n(predict severe CRS)', '#EF5350'),
        (5, 1, 'Output: Severity Prediction', '#BA68C8'),
    ]
    
    for i, (x, y, text, color) in enumerate(steps):
        box = FancyBboxPatch((x - 3.5, y - 0.7), 7, 1.4, 
                             boxstyle="round,pad=0.05,rounding_size=0.2",
                             facecolor=color, edgecolor='black', linewidth=2, alpha=0.9)
        ax1.add_patch(box)
        ax1.text(x, y, text, ha='center', va='center', fontsize=10, fontweight='bold', color='white')
        
        # Arrow to next
        if i < len(steps) - 1:
            ax1.annotate('', xy=(5, y - 1.1), xytext=(5, y - 0.7),
                        arrowprops=dict(arrowstyle='->', color='black', lw=2))
    
    # Optional BERT note
    ax1.text(9, 6, 'Optional:\nBioBERT\nembeddings', ha='center', va='center', fontsize=9,
             bbox=dict(boxstyle='round', facecolor='#E1BEE7', edgecolor='#BA68C8', alpha=0.8))
    ax1.annotate('', xy=(8, 6), xytext=(6.5, 7),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1, linestyle='--'))
    
    # Right panel: Results summary
    ax2 = fig.add_subplot(122)
    ax2.axis('off')
    ax2.set_title('B. Results & Limitations', fontsize=14, fontweight='bold', loc='left')
    
    # Results table
    results_text = """
CURRENT RESULTS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Total narratives analyzed: 100
• Average narrative length: ~25 chars
• Model AUC: 0.50 (random performance)
• Features extracted: None meaningful

WHY LIMITED RESULTS?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• FAERS narratives are mostly empty
  or contain only dates/codes
• No detailed clinical descriptions
• Example: "CASE EVENT DATE: 20210323"

FUTURE POTENTIAL:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Rich EHR clinical notes
• Published case reports
• Detailed adverse event narratives
• Multi-modal: NLP + structured data
"""
    
    ax2.text(0.1, 0.95, results_text, transform=ax2.transAxes, fontsize=11,
             va='top', ha='left', family='monospace',
             bbox=dict(boxstyle='round', facecolor='#F5F5F5', edgecolor='gray', alpha=0.9))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")


def main():
    """Generate all NLP visualizations."""
    print("Generating NLP analysis visualizations...")
    print("=" * 50)
    
    # Create figures directory
    os.makedirs('figures', exist_ok=True)
    
    # Generate individual figures
    create_nlp_pipeline_diagram()
    create_feature_extraction_results()
    create_model_performance_plot()
    create_limitations_and_future()
    
    # Generate combined summary figure
    create_nlp_summary_figure()
    
    print("=" * 50)
    print("All NLP visualizations saved to 'figures/' directory")
    print("\nFor PowerPoint:")
    print("  - Slide 1 (Methods): Use nlp_pipeline.png")
    print("  - Slide 2 (Results): Use nlp_summary.png OR nlp_limitations_future.png")


if __name__ == "__main__":
    main()


