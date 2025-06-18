import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle, FancyBboxPatch
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as mcolors
import os

# Create a modern-looking figure with a white background
plt.style.use('default')
fig, ax = plt.subplots(figsize=(14, 10), dpi=300)
fig.patch.set_facecolor('#FFFFFF')
ax.set_facecolor('#FFFFFF')
ax.set_xlim(0, 14)
ax.set_ylim(0, 10)
ax.axis('off')

# Modern color palette with gradients
def create_gradient(base_color, darken_factor=0.2):
    """Create gradient colors for modern look"""
    r, g, b = mcolors.to_rgb(base_color)
    r2 = max(0.0, r - darken_factor)
    g2 = max(0.0, g - darken_factor)
    b2 = max(0.0, b - darken_factor)
    return LinearSegmentedColormap.from_list("", [(r2, g2, b2), base_color])

colors = {
    'input': create_gradient('#3498db'),         # Blue
    'preprocess': create_gradient('#9b59b6'),    # Purple
    'regression': create_gradient('#2ecc71'),    # Green
    'impact': create_gradient('#e74c3c'),        # Red
    'fee': create_gradient('#f39c12'),           # Orange
    'integration': create_gradient('#8e44ad'),   # Dark Purple
    'output': create_gradient('#16a085'),        # Teal
    'bg': '#FFFFFF',                             # White background
    'text': '#FFFFFF',                           # White text for colored boxes
    'dark_text': '#333333',                      # Dark text for light backgrounds
    'arrow': '#555555',                          # Dark gray arrows
    'border': '#333333',                         # Dark gray borders
    'highlight': '#f1c40f'                       # Yellow highlight
}

# Function to draw a modern box with gradient
def draw_modern_box(x, y, width, height, text, color_map, alpha=0.9, fontsize=12, zorder=10):
    # Draw gradient background
    for i in np.linspace(0, height, 100):
        color = color_map(i/height)
        rect = Rectangle((x, y+i), width, height/100, facecolor=color, edgecolor=None, alpha=alpha)
        ax.add_patch(rect)
    
    # Draw border with rounded corners using FancyBboxPatch
    pad = 0.01
    border = FancyBboxPatch(
        (x-pad, y-pad), 
        width+2*pad, height+2*pad,
        boxstyle=f"round,pad={pad},rounding_size=0.1",
        ec=colors['border'], fc='none',
        linewidth=1.5, alpha=0.8, zorder=zorder
    )
    ax.add_patch(border)
    
    # Add text with a professional font
    ax.text(x + width/2, y + height/2, text, ha='center', va='center', 
            fontsize=fontsize, color=colors['text'], weight='bold', 
            family='sans-serif', zorder=zorder+1)
    
    return border

# Function to draw a modern arrow
def draw_modern_arrow(start, end, color=colors['arrow'], style='-|>', linewidth=2.0, alpha=0.9, curve=0.1):
    arrow = FancyArrowPatch(
        start, end, 
        arrowstyle=style, 
        connectionstyle=f"arc3,rad={curve}", 
        color=color, 
        linewidth=linewidth,
        alpha=alpha,
        zorder=5
    )
    ax.add_patch(arrow)
    return arrow

# Function to add a code example/detail box
def add_detail_box(x, y, width, height, text, color='#f5f5f5', fontsize=9, alpha=0.9):
    rect = Rectangle((x, y), width, height, facecolor=color, edgecolor=colors['border'], 
                      alpha=alpha, linewidth=1, linestyle='-', zorder=9)
    ax.add_patch(rect)
    ax.text(x + 0.1, y + height - 0.15, text, ha='left', va='top', 
            fontsize=fontsize, color=colors['dark_text'], family='monospace', zorder=10,
            wrap=True)
    return rect

# Draw the main workflow components with modern styling
# Input
draw_modern_box(4, 8.8, 6, 0.8, "L2 Orderbook Data & Market Conditions", colors['input'], fontsize=14)

# Preprocessing
draw_modern_box(4, 7.8, 6, 0.8, "Preprocessing & Feature Extraction", colors['preprocess'], fontsize=14)

# Models (on the same level)
# Regression Models
draw_modern_box(1, 6.2, 3, 1.0, "Regression Models", colors['regression'], fontsize=13, zorder=20)
# AlmgrenChriss Model
draw_modern_box(5.5, 6.2, 3, 1.0, "Almgren-Chriss Model", colors['impact'], fontsize=13, zorder=20)
# Fee Model
draw_modern_box(10, 6.2, 3, 1.0, "Fee Model", colors['fee'], fontsize=13, zorder=20)

# Integration
draw_modern_box(4, 4, 6, 0.8, "Cost Integration & Analysis", colors['integration'], fontsize=14)

# Output
draw_modern_box(4, 2.8, 6, 0.8, "Execution Cost & Market Impact Estimate", colors['output'], fontsize=14)

# Draw modern arrows connecting components
# Main flow arrows
draw_modern_arrow((7, 8.8), (7, 8.6))
draw_modern_arrow((7, 7.8), (7, 7.6))

# Arrows from preprocessing to models
draw_modern_arrow((6, 7.6), (2.5, 6.7), curve=-0.2)
draw_modern_arrow((7, 7.6), (7, 7.2))
draw_modern_arrow((8, 7.6), (11.5, 6.7), curve=0.2)

# Arrows from models to integration
draw_modern_arrow((2.5, 6.2), (5, 4.8), curve=0.2)
draw_modern_arrow((7, 6.2), (7, 4.8))
draw_modern_arrow((11.5, 6.2), (9, 4.8), curve=-0.2)

# Arrow from integration to output
draw_modern_arrow((7, 4), (7, 3.6))

# Add details about each component
# Regression Models
regr_detail = """
LinearRegression: Estimates slippage
- Uses matrix operations with Eigen
- y = β₀ + β₁·Q + β₂·V + β₃·D + β₄·σ

QuantileRegression: For tail risk
- Estimates specific percentiles
- Supports risk assessment

LogisticRegression: Maker/taker prediction
- P(maker) = sigmoid(α₀ + α₁·Q + α₂·V)
"""
add_detail_box(0.2, 4.8, 4.6, 1.2, regr_detail)

# Almgren-Chriss Model
impact_detail = """
AlmgrenChriss: Market impact calculation
- Temporary impact: σ·γ·√(τ/V)·(q/Q)ᵟ
- Permanent impact: κ·q/V
- Optimal execution trajectory
- Risk-adjusted cost minimization
"""
add_detail_box(4.7, 4.8, 4.6, 1.2, impact_detail)

# Fee Model
fee_detail = """
FeeModel: Fee structure calculations
- Exchange-specific fee tiers (OKX)
- Maker/taker rate differentiation
- Weighted fee calculation:
  fee = makerRate·P + takerRate·(1-P)
"""
add_detail_box(9.2, 4.8, 4.6, 1.2, fee_detail)

# Feature Extraction
features_detail = """
Features extracted from orderbook:
- Order book depth and imbalance
- Bid-ask spread measurements
- Volatility metrics
- Volume normalization
"""
add_detail_box(1.2, 7.2, 4, 0.4, features_detail, color='#f0f0f0')

# Market Environment
market_detail = """
Market Environment Parameters:
- Current price and volatility
- Recent trading volume
- Liquidity conditions
"""
add_detail_box(8.8, 7.2, 4, 0.4, market_detail, color='#f0f0f0')

# Cost Integration
integration_detail = """
Total Cost = Slippage + Market Impact + Trading Fees
- Weighted by execution probability
- Accounts for order type and size
- Market-specific adjustments
"""
add_detail_box(8.4, 3.8, 5, 0.6, integration_detail, color='#f0f0f0')

# Add title with professional typography
plt.figtext(0.5, 0.96, "High-Performance Trade Simulator Component Architecture", 
            fontsize=22, ha='center', color=colors['dark_text'], weight='bold', 
            family='sans-serif')

# Add a subtle watermark
plt.figtext(0.5, 0.04, "Krrish Choudhary", 
            fontsize=12, ha='center', color='#aaaaaa', alpha=0.7,
            style='italic', family='serif')

# Save the figure with tight layout
output_dir = "documentation/images"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "detailed_architecture_white.png")
plt.savefig(output_path, bbox_inches='tight', dpi=300, 
            facecolor=fig.get_facecolor(), edgecolor='none')
print(f"Detailed architecture diagram with white background saved to {output_path}")
plt.close()