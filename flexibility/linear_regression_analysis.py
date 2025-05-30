import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import re

def normalize_id(pid):
    return re.sub(r'[._]', '_', pid.strip().lower())

def read_predictions(file_path):
    predictions = {}
    current_protein = None
    current_values = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_protein is not None:
                    predictions[normalize_id(current_protein)] = np.array(current_values)
                current_protein = line[1:]
                current_values = []
            elif line:
                values = [float(x) for x in line.replace(', ', ',').split(',') if x]
                current_values.extend(values)
    if current_protein is not None:
        predictions[normalize_id(current_protein)] = np.array(current_values)
    return predictions

def main():
    # Read predictions
    ml_predictions = read_predictions('test_output-3D-all-predictions.txt')
    enm_predictions = read_predictions('test_output-3D-all-predictions_enm.txt')
    
    # Get common proteins
    common_proteins = set(ml_predictions.keys()) & set(enm_predictions.keys())
    
    # Prepare data for regression
    all_ml = []
    all_enm = []
    for protein in sorted(common_proteins):
        ml_values = ml_predictions[protein]
        enm_values = enm_predictions[protein]
        if len(ml_values) != len(enm_values):
            print(f"Skipping {protein}: ML={len(ml_values)}, ENM={len(enm_values)}")
            continue
        all_ml.extend(ml_values)
        all_enm.extend(enm_values)
    
    all_ml = np.array(all_ml)
    all_enm = np.array(all_enm)
    
    # Find transformation to align with y=x
    # We want to find f(enm) ≈ ml, so we fit enm to predict ml
    X = all_enm.reshape(-1, 1)  # ENM as input
    y = all_ml  # ML as target
    
    # Perform linear regression
    model = LinearRegression()
    model.fit(X, y)
    
    # Get transformed ENM values
    transformed_enm = model.predict(X)
    
    # Calculate correlations with transformed values
    pearson_r, pearson_p = stats.pearsonr(all_ml, transformed_enm)
    spearman_r, spearman_p = stats.spearmanr(all_ml, transformed_enm)
    
    # Calculate statistics
    r2 = r2_score(all_ml, transformed_enm)
    mse = mean_squared_error(all_ml, transformed_enm)
    rmse = np.sqrt(mse)
    
    # Print results
    print("\nTransformation Results:")
    print("-" * 50)
    print(f"Transformation: ML ≈ {model.coef_[0]:.4f} × ENM + {model.intercept_:.4f}")
    print(f"R² score: {r2:.4f}")
    print(f"Root Mean Square Error (RMSE): {rmse:.4f}")
    print(f"Pearson correlation (r): {pearson_r:.4f} (p={pearson_p:.2e})")
    print(f"Spearman correlation (r): {spearman_r:.4f} (p={spearman_p:.2e})")
    
    # Create visualization
    plt.figure(figsize=(10, 8))
    
    # Plot original data points
    sns.scatterplot(x=all_ml, y=all_enm, alpha=0.3, s=20, label='Original data', color='gray')
    
    # Plot transformed data points
    sns.scatterplot(x=all_ml, y=transformed_enm, alpha=0.5, s=20, label='Transformed ENM', color='blue')
    
    # Plot y=x line
    min_val = min(all_ml.min(), transformed_enm.min())
    max_val = max(all_ml.max(), transformed_enm.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, label='y=x line')
    
    # Add correlation statistics in a text box
    stats_text = f'Pearson r = {pearson_r:.4f}\nSpearman r = {spearman_r:.4f}'
    plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'),
             verticalalignment='top', fontsize=10)
    
    plt.xlabel('ML Flexibility Prediction')
    plt.ylabel('ENM Flexibility Prediction')
    plt.title('ML vs Transformed ENM Flexibility Predictions')
    plt.legend()
    plt.tight_layout()
    plt.savefig('linear_regression_analysis.png', dpi=300)
    plt.close()
    
    print("\nPlot saved as 'linear_regression_analysis.png'")

if __name__ == "__main__":
    main() 