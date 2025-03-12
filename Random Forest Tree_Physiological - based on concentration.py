# -*- coding: utf-8 -*-
"""
Created on Mon Mar  3 09:18:49 2025

@author: Sueda
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Mar  3 08:48:15 2025

@author: Sueda
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import warnings
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy.stats import f_oneway
from scipy import stats
import itertools
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.tree import export_graphviz
from IPython.display import Image
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import plot_tree

warnings.filterwarnings('ignore')

# Define the data
conditions = ['5 mM L-Met']*15 + ['10 mM L-Met']*15 + ['15 mM L-Met']*15 + ['20 mM L-Met']*15 + ['25 mM L-Met']*15 + ['HCl Kontrol']*15 + ['H2O Kontrol']*15 + \
             ['5 mM L-Met']*15 + ['10 mM L-Met']*15 + ['15 mM L-Met']*15 + ['20 mM L-Met']*15 + ['25 mM L-Met']*15 + ['HCl Kontrol']*15 + ['H2O Kontrol']*15

times = ['Recovery']*105 + ['Treatment']*105

shoot_lengths = [
    30, 37, 36, 30, 35, 32, 35, 33, 33, 32, 35, 35, 31, 38, 39,  # 5 mM L-Met REC
    32, 25, 33, 22, 25, 27, 30, 31, 21, 33, 22, 34, 31, 36, 29,     # 10 mM L-Met REC
    22, 16, 24, 16, 14, 16, 23, 15, 16, 16, 15, 16, 22, 16, 18,   # 15 mM L-Met REC
    15, 13, 11, 16, 16, 11, 14, 14, 13, 15, 12, 14, 11, 16, 16,     # 20 mM L-Met REC
    12, 13, 11, 10, 13, 10, 10, 10, 10, 14, 13, 14, 10, 10, 12,     # 25 mM L-Met REC
    7, 8, 10, 9, 9, 9, 11, 9, 10, 11, 10, 10, 10, 9, 10,            # HCl Kontrol REC
    26, 27, 24, 22, 29, 23, 26, 22, 24, 25, 23, 26, 32, 23, 24,   # H2O Kontrol REC
    19, 20, 20, 19, 21, 22, 27, 19, 18, 17, 20, 17, 17, 18, 22,   # 5 mM L-Met No REC
    13, 16, 15, 17, 17, 16, 14, 13, 16, 16, 16, 14, 18,  14, 15,   # 10 mM L-Met No REC
    15, 17, 10, 17, 13, 15, 17, 15, 11, 13, 11, 13, 10, 16, 12,     # 15 mM L-Met No REC
    11, 12, 12, 13, 10, 9, 12, 9, 10, 6, 10, 7, 9, 10, 10,          # 20 mM L-Met No REC
    5, 4, 5, 7, 11, 4, 3, 3, 11, 5, 11, 6, 4, 9, 5,                 # 25 mM L-Met No REC
    7, 6, 6, 5, 7, 7, 7, 4, 6, 6, 6, 6, 4, 4, 5,                    # HCl Kontrol No REC
    27, 29, 21, 24, 21, 22, 20, 22, 22, 25, 20, 26, 28, 25, 24    # H2O Kontrol No REC
]

root_lengths = [
    14, 18, 16, 17, 13, 11, 15, 16, 14, 15, 14, 16, 14, 14, 13,  # 5 mM L-Met REC
    18, 14, 13, 14, 12, 11, 13, 14, 15, 12, 16, 11, 11, 15, 14,    # 10 mM L-Met REC
    14, 16, 13, 15, 12, 15, 13, 14, 15, 16, 16, 14, 15, 13, 16,  # 15 mM L-Met REC
    14, 14, 16, 17, 15, 12, 12, 15, 11, 15, 14, 18, 19, 15, 11,  # 20 mM L-Met REC
    15, 15, 17, 16, 12, 12, 12, 14, 16, 15, 14, 17, 14, 10, 13,     # 25 mM L-Met REC
    10, 10, 13, 12, 11, 12, 11, 9, 10, 8, 8, 7,  14, 10, 8,          # HCl Kontrol REC
    16, 20, 25, 24, 21, 19, 23, 18, 16, 22, 20, 24, 24, 18, 19,  # H2O Kontrol REC
    11, 9, 8, 9, 9, 10, 14, 12, 13, 11, 11, 10, 14, 9, 13,         # 5 mM L-Met No REC
    9, 7, 15, 14, 9, 15, 9, 9, 14, 11, 9, 12, 13, 10, 12,           # 10 mM L-Met No REC
    12, 13, 11, 14, 15, 11, 12, 12, 19, 11, 11, 12, 10, 14, 13,  # 15 mM L-Met No REC
    15, 14, 10, 13, 13, 12, 11, 11, 10, 15, 12, 13, 10, 10, 11,    # 20 mM L-Met No REC
    16, 11, 11, 11, 11, 9, 10, 12, 9, 15, 14, 15, 14, 11, 9,        # 25 mM L-Met No REC
    10, 9, 14, 12, 11, 11, 10, 12, 9, 10, 13, 12, 10, 9, 11,       # HCl Kontrol No REC
    21, 14, 19, 17, 14, 20, 17, 22, 29, 20, 16, 18, 22, 20, 19   # H2O Kontrol No REC
]

fresh_weight = [
    0.041, 0.037, 0.035, 0.038, 0.04, 0.039, 0.043, 0.038, 0.042, 0.043, 0.035, 0.043, 0.039, 0.044, 0.041,  # 5 mM REC
    0.032, 0.025, 0.032, 0.028, 0.025, 0.028, 0.029, 0.026, 0.031, 0.027, 0.028, 0.034, 0.033, 0.031, 0.03,  # 10 mM REC
    0.027, 0.02, 0.026, 0.018, 0.02, 0.023, 0.03, 0.018, 0.017, 0.019, 0.019, 0.033, 0.021, 0.024, 0.028,     # 15 mM REC
    0.018, 0.02, 0.019, 0.018, 0.017, 0.02, 0.018, 0.017, 0.018, 0.016, 0.017, 0.02, 0.022, 0.023, 0.018,   # 20 mM REC
    0.014, 0.014, 0.014, 0.012, 0.011, 0.011, 0.011, 0.013, 0.011, 0.011, 0.013, 0.015, 0.012, 0.011, 0.012,   # 25 mM REC
    0.018, 0.013, 0.019, 0.014, 0.013, 0.018, 0.019, 0.01, 0.013, 0.016, 0.017, 0.019, 0.012, 0.015, 0.012,   # HCl Kontrol REC
    0.054, 0.062, 0.058, 0.055, 0.05, 0.05, 0.058, 0.06, 0.054, 0.055, 0.06, 0.054, 0.05, 0.055, 0.058,      # H2O Kontrol REC
    0.03, 0.021, 0.024, 0.03, 0.038, 0.03, 0.035, 0.035, 0.029, 0.025, 0.028, 0.032, 0.021, 0.032, 0.03,      # 5 mM No REC
    0.018, 0.018, 0.021, 0.016, 0.017, 0.028, 0.023,  0.03, 0.031, 0.025, 0.019, 0.029, 0.025, 0.022, 0.03,    # 10 mM No REC
    0.025, 0.025, 0.021, 0.028, 0.017,  0.026, 0.018, 0.025, 0.017, 0.026, 0.024, 0.026, 0.022, 0.024, 0.02,    # 15 mM No REC
    0.018, 0.032, 0.028, 0.027, 0.024, 0.018, 0.017, 0.019, 0.014,  0.02, 0.015, 0.014, 0.018, 0.017, 0.014,   # 20 mM No REC
    0.015, 0.016, 0.012, 0.022, 0.013, 0.018, 0.011, 0.014, 0.019, 0.014, 0.015, 0.016, 0.018, 0.015, 0.01,    # 25 mM No REC
    0.007, 0.002, 0.008, 0.007, 0.002, 0.011, 0.011, 0.004,  0.002, 0.004, 0.006, 0.006, 0.004, 0.003, 0.005,     # HCl Kontrol No REC
    0.053, 0.059, 0.051, 0.057, 0.059, 0.05, 0.048, 0.051, 0.055, 0.05, 0.052, 0.049, 0.055, 0.051, 0.053        # H2O Kontrol No REC
]

dry_weight = [
    0.008, 0.008, 0.007, 0.007, 0.009, 0.006, 0.009, 0.008, 0.009, 0.007, 0.008, 0.008, 0.007, 0.006, 0.007,
    0.009, 0.006, 0.007, 0.006, 0.005, 0.005, 0.006, 0.006, 0.008, 0.005, 0.005,0.008, 0.007, 0.007, 0.009,
    0.007, 0.006, 0.007, 0.006, 0.007, 0.004, 0.006, 0.006, 0.005, 0.005, 0.007, 0.005, 0.005, 0.007, 0.004,
    0.006, 0.006, 0.007, 0.007, 0.005, 0.005, 0.005, 0.004,0.005, 0.005, 0.005, 0.004, 0.004, 0.005, 0.004,
    0.004, 0.004, 0.004, 0.003, 0.003, 0.003, 0.004, 0.005, 0.005, 0.005, 0.004, 0.005, 0.005, 0.004, 0.005,
    0.003, 0.002, 0.002, 0.003, 0.003, 0.002, 0.003, 0.002, 0.002, 0.004, 0.002, 0.003, 0.002, 0.003, 0.002,
    0.009, 0.012, 0.009, 0.009, 0.010, 0.009, 0.009, 0.012, 0.009, 0.011, 0.012, 0.010, 0.010, 0.009, 0.010,
    0.006, 0.008, 0.006, 0.008, 0.007, 0.008, 0.008, 0.008, 0.008, 0.007, 0.006, 0.006, 0.007, 0.007, 0.008,
    0.005, 0.005, 0.005, 0.006, 0.005, 0.006, 0.004, 0.006, 0.005, 0.005, 0.007,0.007, 0.005, 0.006, 0.009,
    0.006, 0.007, 0.005, 0.006, 0.004, 0.006, 0.006, 0.004, 0.005, 0.004, 0.005, 0.007, 0.005, 0.006, 0.005,
    0.004, 0.007, 0.006, 0.004, 0.007, 0.003, 0.004, 0.005, 0.004, 0.005, 0.007, 0.006, 0.004, 0.005, 0.004,
    0.004, 0.004, 0.005, 0.003, 0.004, 0.003 ,0.003, 0.005, 0.004, 0.005, 0.003, 0.006, 0.005, 0.004, 0.005,
    0.003, 0.003, 0.002, 0.001, 0.003, 0.003, 0.004, 0.001, 0.001, 0.001, 0.002, 0.003, 0.002, 0.002, 0.001,
    0.007, 0.008, 0.006, 0.009, 0.006, 0.008, 0.007, 0.006, 0.007, 0.008, 0.01, 0.005, 0.006, 0.007, 0.006
]

water_content = [
    85.36585366, 78.37837838, 81.57894737, 85,          82.05128205, 79.06976744, 79.31034483, 76.31578947, 80.95238095, 79.06976744, 79.41176471, 77.14285714, 84.61538462, 82.05128205, 82.92682927, 
    71.875,	     76,          78.125,      78.57142857, 78.94736842, 80,          82.14285714, 79.31034483, 76.92307692, 81.48148148, 82.14285714, 75.75757576, 77.41935484, 76.66666667, 77.5,
    74.07407407, 70,          73.07692308, 66.66666667, 82.60869565, 80,          83.33333333, 73.33333333, 73.68421053, 73.68421053, 75.75757576, 66.66666667, 84.84848485, 70.58823529, 70.83333333, 
    66.66666667, 63.15789474, 70.58823529, 66.66666667, 75,          87.09677419, 83.33333333, 70.58823529, 87.5,        70.58823529, 75,          81.81818182, 82.60869565, 72.22222222, 73.33333333,
    71.43,       71.43,       71.43,       72.73,       72.72727273, 63.63636364, 61.53846154, 70,          54.54545455, 54.54545455, 69.23076923, 66.66666667, 68.75,       63.63636364, 58.33333333,
    83.33333333, 84.61538462, 89.47368421, 84.61538462, 83.33333333, 84.21052632, 80,          87.5,        83.33333333, 81.81818182, 88.23529412, 84.21052632, 83.33333333, 80,          83.33333333,
    83.33333333, 85.48387097, 84.48275862, 81.81818182, 86,          82,          84.48275862, 80,          83.33333333, 81.81818182, 80,          81.48148148, 86,          83.63636364, 84.48275862, 
	73.33333333, 71.42857143, 76.47058824, 70.83333333, 73.33333333, 78.94736842, 73.33333333, 77.14285714 ,80,          72,          75,          75,          71.42857143, 75,          76.66666667,
    72.22222222, 72.22222222, 71.42857143, 68.75,       64.70588235, 71.42857143, 75,          73.91304348, 68.75,       83.33333333, 77.41935484, 68,          82.75862069, 72.72727273, 70,
    76,	         72,          76.19047619, 78.57142857, 76.47058824, 76.92307692, 77.77777778, 80,          76.47058824, 73.07692308, 77.27272727, 75,          75,          75,          76.92307692,
    77.77777778, 78.78787879, 81.25,       85.71428571, 74.07407407, 77.77777778, 70.5882352,  78.94736842, 64.28571429, 81.81818182, 65,          71.42857143, 83.33333333, 70.58823529, 71.42857143,
    77.78,	     70.59,       86.36363636, 69.23076923, 83.33333333, 72.72727273, 80.76923077, 71.42857143, 73.68421053, 78.57142857, 76.92307692, 86.66666667, 68.75,       73.33333333, 73.33,
    57.14285714, 50,          62.5,        71.42857143,	50,          70,          72.72727273, 63.63636364, 75 ,         50,          66.66666667, 66.66666667, 66.66666667, 75,          66.66666667,
    86.79245283, 88.37209302, 86.66666667, 88.23529412, 84.21052632, 87.23404255, 86.44067797, 86,          87.5,        86.2745098,  85.45454545, 80,          87.75510204, 87.27272727, 88.67924528
]

# Create DataFrame
data = pd.DataFrame ({
    'Condition': conditions,
    'Time': times,
    'Shoot_Length': shoot_lengths,
    'Root_Length': root_lengths,
    'Fresh_Weight' : fresh_weight,
    'Dry_Weight' : dry_weight,
    '%_Water_Content' : water_content,   
})

df = pd.DataFrame 

data['Condition'] = data['Condition'].astype(str)
data['Time'] = data['Time'].astype(str)

# One-Hot Encode categorical variables
data = pd.get_dummies(data, columns=['Condition', 'Time'], drop_first=False)

print(data.head())

# Drop 'Time_Treatment' to avoid data leakage
X = data.drop(columns=['Condition_15 mM L-Met', 'Condition_20 mM L-Met', 'Condition_25 mM L-Met', 'Condition_10 mM L-Met', 
                       'Condition_5 mM L-Met', 'Condition_H2O Kontrol', 'Condition_HCl Kontrol'])

# Create target variable 'y' using one-hot encoded columns
y = data[['Condition_15 mM L-Met', 'Condition_20 mM L-Met', 'Condition_25 mM L-Met', 'Condition_10 mM L-Met', 
          'Condition_5 mM L-Met', 'Condition_H2O Kontrol', 'Condition_HCl Kontrol']]

# Convert y to categorical labels
y = y.idxmax(axis=1)  # This gets the column name (i.e., the condition) where the value is True

# Encode 'y' (since it's still categorical)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# İlk ağacı çizme
plt.figure(figsize=(40, 10))  # Görselleştirme boyutunu ayarladık
plot_tree(rf_model.estimators_[0], 
          feature_names=X.columns, 
          class_names=label_encoder.classes_, 
          filled=True, 
          rounded=True, 
          fontsize=6)

plt.savefig('Random_Forest_Tree.png', dpi=300) 

plt.show()



# Predict and evaluate
y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Feature importance
importances = rf_model.feature_importances_
feature_names = X.columns
feature_importance = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_importance = feature_importance.sort_values(by='Importance', ascending=False)
print(feature_importance)