import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# Set font to Arial
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']

def meV_to_cm1(x):
    return x * 8.0655439

def cm1_to_meV(x):
    return x / 8.0655439

def parse_filename(filename):
    """
    Parses filename: CsPdS-{layerN}_{Pol}{time}_merged_meV.csv
    Returns a dictionary with layer, pol, and time.
    """
    pattern = r"CsPdS-(?P<layer>[^_]+)_(?P<pol>RL|RR)(?P<time>.*)_merged_meV\.csv"
    match = re.match(pattern, filename)
    if match:
        info = match.groupdict()
        # Clean up time string (e.g., '18_min' -> '18 min')
        time_str = info['time'].replace('_', ' ').strip()
        if not time_str:
            time_str = "0 min"
        info['time_label'] = time_str
        return info
    return None

def plot_categorized_data(data_dir='data'):
    if not os.path.exists(data_dir):
        print(f"Directory '{data_dir}' not found.")
        return

    files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    
    # Categorize files by (layer, pol)
    categories = {}
    for f in files:
        info = parse_filename(f)
        if info:
            key = (info['layer'], info['pol'])
            if key not in categories:
                categories[key] = []
            categories[key].append((f, info['time_label']))
            
    if not categories:
        print("No matching files found in the data directory.")
        return

    # Process each category
    for (layer, pol), file_list in categories.items():
        # Sort by time label if possible (extracting number if present)
        file_list.sort(key=lambda x: int(re.search(r'\d+', x[1]).group()) if re.search(r'\d+', x[1]) else 0)
        
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        # Use a sequential discrete colormap from magma
        n_files = len(file_list)
        # Resampling magma to get n_files discrete colors
        cmap = plt.get_cmap('magma')
        colors = cmap(np.linspace(0.2, 0.85, n_files)) if n_files > 1 else [cmap(0.5)]
        
        for i, (fname, time_label) in enumerate(file_list):
            path = os.path.join(data_dir, fname)
            try:
                # Format: junk in Col 1, X in Row 1, Y in Row 2
                df = pd.read_csv(path, header=None)
                
                # Extract X and Y (skipping the first column)
                x_vals = df.iloc[0, 1:].values.astype(float)
                y_vals = df.iloc[1, 1:].values.astype(float)
                
                ax1.plot(x_vals, y_vals+min(max(y_vals),2000)*i, label=time_label, color=colors[i], alpha=0.8)
            except Exception as e:
                print(f"Error processing {fname}: {e}")
                continue
                
        # Main axis formatting (meV)
        ax1.set_xlabel('Raman Shift (meV)', fontsize=12)
        ax1.set_ylabel('Intensity (a.u.)', fontsize=12)
        ax1.set_title(f'CsPdS - {layer} - {pol}', fontsize=14, fontweight='bold')
        ax1.legend(title='Time Elapsed', frameon=True)
        # ax1.grid(True, linestyle='--', alpha=0.6)
        
        # Secondary axis (cm-1) on top
        ax2 = ax1.secondary_xaxis('top', functions=(meV_to_cm1, cm1_to_meV))
        ax2.set_xlabel('Raman Shift (cm$^{-1}$)', fontsize=12)
        # ax1.set_xlim(30, 70)
        
        plt.tight_layout()
        # Save as PDF
        pdf_name = f"CsPdS_{layer}_{pol}.pdf"
        plt.savefig(pdf_name, format='pdf')
        print(f"Saved: {pdf_name}")
        plt.close()

if __name__ == "__main__":
    plot_categorized_data()
