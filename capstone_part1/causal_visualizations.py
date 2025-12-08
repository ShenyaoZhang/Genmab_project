"""
Causal Analysis Visualizations for PowerPoint Slides
Generates publication-ready figures for CRS risk analysis
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import json
import os

# Set style for clean presentation graphics
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16


def create_dag_visualization(save_path='figures/dag_framework.png'):
    """Create a clean DAG diagram for the causal framework."""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # Define node positions
    nodes = {
        'Confounders': (5, 7, '#FFB74D'),      # Orange - top
        'Dose': (2, 4.5, '#64B5F6'),           # Blue - left
        'CRS Severity': (8, 4.5, '#EF5350'),   # Red - right
        'T-cell\nActivation': (3.5, 2.5, '#81C784'),  # Green - pathway
        'Cytokine\nRelease': (6.5, 2.5, '#81C784'),   # Green - pathway
        'Effect Modifiers\n(Steroids, Tocilizumab)': (5, 0.8, '#BA68C8'),  # Purple - bottom
    }
    
    # Draw nodes
    for name, (x, y, color) in nodes.items():
        box = mpatches.FancyBboxPatch(
            (x - 1.3, y - 0.5), 2.6, 1,
            boxstyle="round,pad=0.1,rounding_size=0.2",
            facecolor=color, edgecolor='black', linewidth=2, alpha=0.9
        )
        ax.add_patch(box)
        ax.text(x, y, name, ha='center', va='center', fontsize=11, fontweight='bold', color='white')
    
    # Draw arrows
    arrow_style = dict(arrowstyle='->', color='#333333', lw=2, mutation_scale=20)
    
    # Confounders to Dose and CRS
    ax.annotate('', xy=(2.5, 5.5), xytext=(4, 6.5), arrowprops=arrow_style)
    ax.annotate('', xy=(7.5, 5.5), xytext=(6, 6.5), arrowprops=arrow_style)
    
    # Causal pathway: Dose -> T-cell -> Cytokine -> CRS
    ax.annotate('', xy=(3, 3.5), xytext=(2.5, 4), arrowprops=arrow_style)
    ax.annotate('', xy=(5.5, 2.5), xytext=(4.5, 2.5), arrowprops=arrow_style)
    ax.annotate('', xy=(7.5, 4), xytext=(7, 3), arrowprops=arrow_style)
    
    # Direct dose to CRS
    ax.annotate('', xy=(6.7, 4.5), xytext=(3.3, 4.5), 
                arrowprops=dict(arrowstyle='->', color='#333333', lw=2.5, 
                               mutation_scale=20, connectionstyle='arc3,rad=0.2'))
    
    # Effect modifiers blocking pathway
    ax.annotate('', xy=(5, 2), xytext=(5, 1.5), 
                arrowprops=dict(arrowstyle='-|>', color='#BA68C8', lw=2, mutation_scale=15))
    
    # Legend
    legend_items = [
        ('Exposure', '#64B5F6'),
        ('Outcome', '#EF5350'),
        ('Confounder', '#FFB74D'),
        ('Causal Pathway', '#81C784'),
        ('Effect Modifier', '#BA68C8'),
    ]
    
    for i, (label, color) in enumerate(legend_items):
        ax.add_patch(plt.Rectangle((0.3, 7.3 - i*0.5), 0.3, 0.3, facecolor=color, edgecolor='black'))
        ax.text(0.75, 7.45 - i*0.5, label, va='center', fontsize=10)
    
    ax.set_title('Causal Framework (DAG) for CRS Risk Analysis', fontsize=18, fontweight='bold', pad=20)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")


def create_forest_plot(save_path='figures/forest_plot_associations.png'):
    """Create a forest plot showing odds ratios for key variables."""
    
    # Load results
    with open('causal_analysis_results.json', 'r') as f:
        results = json.load(f)
    
    # Define variables to show with interpretations
    variables = [
        ('has_steroids', 'Steroid Premedication', 'Effect Modifier'),
        ('has_tocilizumab', 'Tocilizumab Use', 'Effect Modifier'),
        ('has_rituximab', 'Rituximab Co-med', 'Correlational'),
        ('sex_male', 'Male Sex', 'Uncertain'),
    ]
    
    # Extract ORs from results
    assoc_data = {r['variable']: r for r in results.get('associations', [])}
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    y_positions = []
    labels = []
    ors = []
    colors = []
    
    color_map = {
        'Effect Modifier': '#BA68C8',
        'Correlational': '#FFB74D', 
        'Uncertain': '#90A4AE',
        'Confounder': '#FFB74D',
    }
    
    for i, (var, label, var_type) in enumerate(variables):
        if var in assoc_data and assoc_data[var].get('odds_ratio'):
            or_val = assoc_data[var]['odds_ratio']
            if or_val and not np.isnan(or_val) and or_val > 0:
                y_positions.append(i)
                labels.append(label)
                ors.append(or_val)
                colors.append(color_map.get(var_type, '#90A4AE'))
    
    # Plot
    log_ors = [np.log10(or_val) if or_val > 0 else 0 for or_val in ors]
    
    bars = ax.barh(range(len(labels)), log_ors, color=colors, edgecolor='black', height=0.6)
    
    # Add OR values as text
    for i, (bar, or_val) in enumerate(zip(bars, ors)):
        width = bar.get_width()
        ax.text(width + 0.1, bar.get_y() + bar.get_height()/2, 
                f'OR = {or_val:.1f}', va='center', fontsize=11, fontweight='bold')
    
    # Reference line at OR = 1 (log10(1) = 0)
    ax.axvline(x=0, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.text(0.05, len(labels) - 0.3, 'OR = 1\n(No effect)', fontsize=9, color='gray')
    
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=12)
    ax.set_xlabel('Log₁₀(Odds Ratio)', fontsize=13)
    ax.set_title('Association with Severe CRS\n(Unadjusted Odds Ratios)', fontsize=16, fontweight='bold')
    
    # Add legend
    legend_patches = [
        mpatches.Patch(color='#BA68C8', label='Effect Modifier (Causal)'),
        mpatches.Patch(color='#FFB74D', label='Correlational'),
        mpatches.Patch(color='#90A4AE', label='Uncertain'),
    ]
    ax.legend(handles=legend_patches, loc='lower right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")


def create_propensity_score_plot(save_path='figures/propensity_score_results.png'):
    """Create visualization for propensity score analysis results."""
    
    with open('causal_analysis_results.json', 'r') as f:
        results = json.load(f)
    
    ps = results.get('propensity_score', {})
    
    ate = ps.get('ate', 0) * 100  # Convert to percentage points
    ci_lower = ps.get('ate_ci_lower', -0.1) * 100
    ci_upper = ps.get('ate_ci_upper', 0.1) * 100
    n_treated = ps.get('n_treated', 0)
    n_control = ps.get('n_control', 0)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left panel: Sample sizes
    categories = ['With Steroids', 'Without Steroids']
    counts = [n_treated, n_control]
    colors = ['#64B5F6', '#EF5350']
    
    bars = ax1.bar(categories, counts, color=colors, edgecolor='black', width=0.6)
    ax1.set_ylabel('Number of Patients', fontsize=13)
    ax1.set_title('Treatment Groups\n(After Propensity Matching)', fontsize=14, fontweight='bold')
    
    for bar, count in zip(bars, counts):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
                str(count), ha='center', fontsize=14, fontweight='bold')
    
    # Right panel: ATE with CI
    ax2.errorbar([1], [ate], yerr=[[ate - ci_lower], [ci_upper - ate]], 
                 fmt='o', markersize=15, color='#64B5F6', capsize=10, capthick=3, elinewidth=3)
    
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
    ax2.set_xlim(0.5, 1.5)
    ax2.set_xticks([1])
    ax2.set_xticklabels(['Steroid Effect'], fontsize=12)
    ax2.set_ylabel('Change in Severe CRS Risk\n(Percentage Points)', fontsize=12)
    ax2.set_title('Average Treatment Effect (ATE)\nof Steroid Premedication', fontsize=14, fontweight='bold')
    
    # Add annotation
    significance = "NOT statistically significant" if ci_lower <= 0 <= ci_upper else "Statistically significant"
    ax2.text(1, ci_upper + 2, f'ATE = {ate:.1f}%\n95% CI: [{ci_lower:.1f}%, {ci_upper:.1f}%]\n({significance})',
             ha='center', fontsize=11, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")


def create_sensitivity_evalue_plot(save_path='figures/sensitivity_evalue.png'):
    """Create E-value sensitivity analysis visualization."""
    
    with open('causal_analysis_results.json', 'r') as f:
        results = json.load(f)
    
    sens = results.get('sensitivity', {})
    observed_or = sens.get('observed_or', 1)
    e_value = sens.get('e_value', 1)
    e_value_ci = sens.get('e_value_ci', 1)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create bar chart comparing observed OR and E-value
    categories = ['Observed OR\n(High vs Low Dose)', 'E-value\n(Robustness Threshold)']
    values = [observed_or, e_value]
    colors = ['#64B5F6', '#EF5350']
    
    bars = ax.bar(categories, values, color=colors, edgecolor='black', width=0.5)
    
    # Add value labels
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{val:.2f}', ha='center', fontsize=14, fontweight='bold')
    
    ax.set_ylabel('Risk Ratio', fontsize=13)
    ax.set_title('Sensitivity Analysis: How Robust is the Dose-CRS Association?', 
                 fontsize=15, fontweight='bold')
    
    # Add interpretation box
    interpretation = (
        f"E-value = {e_value:.2f}\n\n"
        f"An unmeasured confounder would need to\n"
        f"be associated with BOTH dose choice AND\n"
        f"severe CRS by a risk ratio of at least\n"
        f"{e_value:.1f} to fully explain away the\n"
        f"observed dose-CRS relationship."
    )
    
    ax.text(0.98, 0.95, interpretation, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='orange', alpha=0.9))
    
    # Reference line at 1
    ax.axhline(y=1, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax.text(1.3, 1.2, 'No association', fontsize=9, color='gray')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")


def create_summary_slide_figure(save_path='figures/causal_summary.png'):
    """Create a summary figure combining key results for one slide."""
    
    with open('causal_analysis_results.json', 'r') as f:
        results = json.load(f)
    
    fig = plt.figure(figsize=(14, 10))
    
    # Create grid
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # Panel A: Variable roles
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.axis('off')
    ax1.set_title('A. Variable Classification', fontsize=14, fontweight='bold', loc='left')
    
    table_data = [
        ['Variable', 'Role', 'Interpretation'],
        ['Epcoritamab Dose', 'Exposure', 'Primary factor of interest'],
        ['Severe CRS', 'Outcome', 'Death / life-threatening / not recovered'],
        ['Age, Disease Stage', 'Confounders', 'Must adjust for'],
        ['Steroids, Tocilizumab', 'Effect Modifiers', 'Block causal pathway'],
        ['# Co-medications', 'Correlational', 'Marker, not causal'],
    ]
    
    colors_table = [['#E0E0E0']*3] + [['white']*3]*5
    table = ax1.table(cellText=table_data, loc='center', cellLoc='left',
                      colWidths=[0.35, 0.25, 0.4], cellColours=colors_table)
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.8)
    
    # Panel B: Key Odds Ratios
    ax2 = fig.add_subplot(gs[0, 1])
    assoc_data = {r['variable']: r for r in results.get('associations', [])}
    
    vars_to_plot = [('has_steroids', 'Steroids'), ('has_tocilizumab', 'Tocilizumab'), 
                    ('has_rituximab', 'Rituximab'), ('sex_male', 'Male Sex')]
    
    ors = []
    labels = []
    for var, label in vars_to_plot:
        if var in assoc_data and assoc_data[var].get('odds_ratio'):
            or_val = assoc_data[var]['odds_ratio']
            if or_val and not np.isnan(or_val) and or_val > 0:
                ors.append(np.log10(or_val))
                labels.append(f'{label}\n(OR={or_val:.1f})')
    
    colors = ['#BA68C8', '#BA68C8', '#FFB74D', '#90A4AE'][:len(ors)]
    ax2.barh(range(len(labels)), ors, color=colors, edgecolor='black', height=0.6)
    ax2.axvline(x=0, color='black', linestyle='--', linewidth=1)
    ax2.set_yticks(range(len(labels)))
    ax2.set_yticklabels(labels, fontsize=10)
    ax2.set_xlabel('Log₁₀(Odds Ratio)')
    ax2.set_title('B. Associations with Severe CRS', fontsize=14, fontweight='bold', loc='left')
    
    # Panel C: Propensity Score Result
    ax3 = fig.add_subplot(gs[1, 0])
    ps = results.get('propensity_score', {})
    ate = ps.get('ate', 0) * 100
    ci_lower = ps.get('ate_ci_lower', -0.1) * 100
    ci_upper = ps.get('ate_ci_upper', 0.1) * 100
    
    ax3.errorbar([0.5], [ate], yerr=[[ate - ci_lower], [ci_upper - ate]], 
                 fmt='o', markersize=12, color='#64B5F6', capsize=8, capthick=2, elinewidth=2)
    ax3.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax3.set_xlim(0, 1)
    ax3.set_xticks([0.5])
    ax3.set_xticklabels(['Steroid Effect'])
    ax3.set_ylabel('Δ Severe CRS Risk (%)')
    ax3.set_title('C. Causal Effect of Steroids (IPW)', fontsize=14, fontweight='bold', loc='left')
    
    sig_text = "Not significant" if ci_lower <= 0 <= ci_upper else "Significant"
    ax3.text(0.5, ci_upper + 3, f'ATE = {ate:.1f}%\n95% CI: [{ci_lower:.1f}%, {ci_upper:.1f}%]\n({sig_text})',
             ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Panel D: E-value
    ax4 = fig.add_subplot(gs[1, 1])
    sens = results.get('sensitivity', {})
    observed_or = sens.get('observed_or', 1)
    e_value = sens.get('e_value', 1)
    
    bars = ax4.bar(['Observed OR', 'E-value'], [observed_or, e_value], 
                   color=['#64B5F6', '#EF5350'], edgecolor='black', width=0.5)
    for bar, val in zip(bars, [observed_or, e_value]):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                f'{val:.1f}', ha='center', fontsize=11, fontweight='bold')
    ax4.axhline(y=1, color='gray', linestyle='--', linewidth=1)
    ax4.set_ylabel('Risk Ratio')
    ax4.set_title('D. Sensitivity Analysis (Dose Effect)', fontsize=14, fontweight='bold', loc='left')
    
    plt.suptitle('Causal Analysis Results Summary', fontsize=18, fontweight='bold', y=1.02)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")


def main():
    """Generate all visualizations."""
    print("Generating causal analysis visualizations...")
    print("=" * 50)
    
    # Create figures directory
    os.makedirs('figures', exist_ok=True)
    
    # Generate individual figures
    create_dag_visualization()
    create_forest_plot()
    create_propensity_score_plot()
    create_sensitivity_evalue_plot()
    
    # Generate combined summary figure
    create_summary_slide_figure()
    
    print("=" * 50)
    print("All visualizations saved to 'figures/' directory")
    print("\nFor PowerPoint:")
    print("  - Slide 1 (Framework): Use dag_framework.png")
    print("  - Slide 2 (Results): Use causal_summary.png OR individual plots")


if __name__ == "__main__":
    main()


