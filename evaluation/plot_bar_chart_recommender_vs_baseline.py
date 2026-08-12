import pandas as pd
import matplotlib.pyplot as plt

# 1. Load the raw trial data saved by your evaluation script
df = pd.read_csv('evaluation_results.csv')

# 2. Calculate the Mean and Standard Deviation automatically
means = [df['Baseline_Recall_%'].mean(), df['Model_Recall_%'].mean()]
stds = [df['Baseline_Recall_%'].std(), df['Model_Recall_%'].std()]
labels = ['Baseline (Popularity)', 'Hybrid Model']

# 3. Create the Bar Chart
plt.figure(figsize=(8, 6))
bars = plt.bar(labels, means, yerr=stds, capsize=10,
               color=['#d9534f', '#5cb85c'], alpha=0.9, edgecolor='black')

# 4. Add labels and styling
plt.ylabel('Mean Recall@20 (%)', fontsize=12)
plt.title('Recommendation Performance: Baseline vs. Hybrid Model', fontsize=14, fontweight='bold')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# 5. Save the image so you can put it in your writeup!
plt.savefig('recall_bar_chart.png', dpi=300, bbox_inches='tight')
print("✅ Chart successfully saved as 'recall_bar_chart.png'")

plt.show()