import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


data = pd.read_csv('lab8official.csv')
corr_matrix = data.corr()

#5 absolute correlations
top_features = corr_matrix.abs().unstack().sort_values(kind="quicksort", ascending=False)
top_features = top_features[top_features < 1].drop_duplicates().head(5).index

unique_features = pd.Index(top_features.get_level_values(0)).union(top_features.get_level_values(1)).unique()
filtered_corr_matrix = corr_matrix.loc[unique_features, unique_features]


plt.figure(figsize=(8, 6))
sns.heatmap(filtered_corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)


plt.title('Task2 Correlation Heatmap')
plt.show()
