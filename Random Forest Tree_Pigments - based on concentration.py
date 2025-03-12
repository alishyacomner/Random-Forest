# -*- coding: utf-8 -*-
"""
Created on Wed Mar 12 13:15:39 2025

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

conditions = ['10 mM L-Met'] * 9 + ['15 mM L-Met'] * 9 + ['20 mM L-Met'] * 9 + ['HCl Kontrol'] * 9 + ['H2O Kontrol'] * 9 + \
             ['10 mM L-Met'] * 9 + ['15 mM L-Met'] * 9 + ['20 mM L-Met'] * 9 + ['HCl Kontrol'] * 9 + ['H2O Kontrol'] * 9
times = ['Recovery'] * 45 + ['Treatment'] * 45
chlorophyll_a = [
    0.080661178, 0.083941518 ,0.082473537, 0.079913035, 0.081281528, 0.080507598, 0.081500988, 0.08080534, 0.08255417, # 10 mM REC
    0.077797178, 0.078661235, 0.078350724, 0.076668114, 0.076332969,0.076185925, 0.07754502, 0.078958942, 0.078041093, # 15 mM REC
    0.044738476, 0.04469229, 0.044423705, 0.04874952, 0.047964599, 0.048750527, 0.046199678, 0.04709342, 0.047756289, # 20 mM REC
    0.068124145, 0.067722052, 0.068596252, 0.073062484, 0.074509068, 0.073641631, 0.069919073, 0.070103536, 0.070309225, # HCl Kontrol REC
    0.152018099, 0.150521224, 0.150047975, 0.148495005, 0.151972723, 0.151091932, 0.145899952, 0.146304409, 0.146508246, # H2O Kontrol REC
    0.0511184, 0.05300608, 0.05102688, 0.051586333, 0.0499925, 0.04990866, 0.049293225, 0.049045278, 0.048797332, # 10 mM No REC
    0.044084817, 0.045023123, 0.044209762, 0.042336293, 0.042903718, 0.043431459, 0.041627575, 0.04223538, 0.042128243, # 15 mM No REC
    0.038819689, 0.038971462, 0.039008402, 0.035901667, 0.035901667, 0.035901667, 0.036414873, 0.036939545, 0.03656211, # 20 mM No REC
    0.042794092, 0.040870051, 0.041984443, 0.040325917, 0.039584131, 0.040207262, 0.039286203, 0.039198157, 0.039389447, # HCl Kontrol No REC
    0.052137132, 0.052328621, 0.051460731, 0.05009475, 0.050167944, 0.050967259, 0.048464019, 0.048464019, 0.048464019, # H2O Kontrol No REC
]
chlorophyll_b = [
    0.041169518, 0.043386114, 0.041591996, 0.035448965, 0.040262243, 0.040569292, 0.03609106, 0.035982898, 0.038359238, # 10 mM REC
    0.039454768, 0.040443155, 0.039904067, 0.035726419, 0.03628782, 0.035501724, 0.036505424, 0.039312553, 0.037006776, # 15 mM REC
    0.020136968, 0.023610707, 0.01983042, 0.023536544, 0.024956071, 0.026173096, 0.022857155, 0.024092301, 0.024607013, # 20 mM REC
    0.019357481, 0.020466204, 0.020959242, 0.022992218, 0.025315986, 0.025186716, 0.02509596, 0.025444351, 0.025328298, # HCl Kontrol REC
    0.058256977, 0.056885436, 0.055092204, 0.060156841, 0.06227964, 0.060864753, 0.058912802, 0.060272636, 0.060600077, # H2O Kontrol REC
    0.0387424, 0.03648816, 0.03544096, 0.034401, 0.033394167, 0.034065333, 0.036766592, 0.035986158, 0.035205725, # 10 mM No REC
    0.028291723, 0.028197686, 0.029328748, 0.025020251, 0.025478452, 0.023188178, 0.030080404, 0.031373918, 0.030808036, # 15 mM No REC
    0.019811159, 0.022033947, 0.020114303, 0.022759583, 0.022759583, 0.022759583, 0.018466716, 0.017587839, 0.017411959, # 20 mM No REC
    0.013392638, 0.01384436, 0.014748882, 0.018244317, 0.021454641, 0.021197938, 0.015619823, 0.016250563, 0.01617176, # HCl Kontrol No REC
    0.026868595, 0.028541407, 0.027147243, 0.030430472, 0.028608481, 0.030668315, 0.029390668, 0.029390668, 0.029390668, # H2O Kontrol No REC
]
carotenoids = [
    0.033615995, 0.035920067, 0.03530497,  0.034432253, 0.03653843,  0.0374603, 0.032514043, 0.033057456, 0.033355423,							
    0.033775691, 0.034848015, 0.03458270,  0.032346751, 0.032227601, 0.032124175, 0.03228728, 0.03286103, 0.031980034,								
    0.017610769, 0.018291453, 0.017101445, 0.022835313, 0.023431098, 0.024232084, 0.020291594, 0.020897748, 0.021201845,								
    0.023833199, 0.024066397, 0.023812352, 0.025696615, 0.026571467, 0.026962192, 0.02658695, 0.026699172, 0.026816998,								
    0.058477507, 0.059022386, 0.05629013,  0.051376874, 0.052414195, 0.051840397, 0.051349734, 0.05183027, 0.051935986,																
    0.029544022, 0.029904199, 0.029761177, 0.033039328, 0.033226074, 0.033382302, 0.02756917, 0.027895269, 0.027908109,								
    0.019661952, 0.019852343, 0.020209149, 0.023216693, 0.023207591, 0.02306719, 0.024780944, 0.024921068, 0.025092921,								
    0.020848917, 0.022221437, 0.022667468, 0.01967921,  0.01967921,  0.01967921, 0.018870138, 0.019238805, 0.019418671,								
    0.016419577, 0.017682675, 0.01766494,  0.016746199, 0.021558016, 0.021306622, 0.016338762, 0.016486676, 0.016487874,								
    0.028171498, 0.029071546, 0.028166506, 0.026977312, 0.026848404, 0.027926909, 0.024784427, 0.024784427, 0.024784427,
]
pheophytin_b = [
    0.066357194, 0.068872984, 0.067388754, 0.055530031, 0.065144176, 0.066717673, 0.057651526, 0.058578198, 0.059569466,								
    0.063228342, 0.064731004, 0.063906486, 0.057739335, 0.059050239, 0.057556367, 0.059899626, 0.064613563, 0.060389624,								
    0.032828266, 0.035335871, 0.03163557,  0.03929017,  0.038855144, 0.042153404, 0.036979946, 0.038366009, 0.038489115,								
    0.036585402, 0.038478481, 0.039682922, 0.043237722, 0.044976188, 0.042361821, 0.04187136,  0.04415367,  0.045149932,								
    0.090801535, 0.08847595,  0.085824038, 0.099549741, 0.104748281, 0.100408043, 0.094086267, 0.095396563, 0.095864738,																
    0.06693216,  0.06738224,  0.0626424,   0.065001667, 0.062943,    0.0657715,   0.058544766, 0.075938991, 0.065069829,								
    0.04519902,  0.042301868, 0.046647596, 0.028732457, 0.028842046, 0.025130933, 0.043902025, 0.043998721, 0.043353491,								
    0.035397777, 0.037877534, 0.035397777, 0.039453958, 0.039453958, 0.039453958, 0.027164486, 0.029269284, 0.026462887,								
    0.025545646, 0.026051093, 0.027548747, 0.030207566, 0.034307676, 0.033359455, 0.022697867, 0.0245387,   0.023956527,								
    0.046706169, 0.04750352,  0.042997322, 0.051557852, 0.047716639, 0.055936611, 0.045763835, 0.045763835, 0.045763835,
]
pheophytin_a = [
    0.137433774, 0.142259672, 0.139683078, 0.136449266, 0.137564123, 0.136513158, 0.138533802, 0.1378638, 0.140019954,								
    0.131689937, 0.133843074, 0.132606921, 0.130814793, 0.130162576, 0.130415832, 0.131863951, 0.133367806, 0.132932385,								
    0.07541451, 0.075771811, 0.076007914, 0.081867518, 0.081552985, 0.081770398, 0.078060413, 0.079795753, 0.081351137,								
    0.115979154, 0.114714735, 0.117341005, 0.124141255, 0.126317799, 0.126266758, 0.119614361, 0.118782836, 0.119503152,								
    0.263282512, 0.260849619, 0.259683683, 0.251711506, 0.257149921, 0.256531493, 0.250917189 ,0.251681393, 0.252019893,																
    0.08854944, 0.08600656,  0.08372688, 0.084046333, 0.083753, 0.0852005,   0.085693867, 0.082147415, 0.080404348,								
    0.0796936, 0.081628656, 0.078726072, 0.082126906, 0.08351152, 0.083439067, 0.075510455, 0.076732169, 0.076841554,								
    0.065181769, 0.065535102, 0.065181769, 0.059151042, 0.059151042, 0.059151042, 0.064727733, 0.064370908, 0.064846675,								
    0.073229439, 0.070373745, 0.07081235, 0.066670172, 0.065975083, 0.067119276, 0.069211573 ,0.0682957, 0.068998193,								
    0.086734752, 0.087311253, 0.087363511, 0.084683926, 0.084728472, 0.083334944, 0.083028913, 0.083028913, 0.083028913,
]

# Create DataFrame
data = pd.DataFrame ({
    'Condition': conditions,
    'Time': times,
    'chlorophyll_a': chlorophyll_a,
    'chlorophyll_b': chlorophyll_b,
    'carotenoids' : carotenoids,
    'pheophytin_b' : pheophytin_b,
    'pheophytin_a' : pheophytin_a,   
})

df = pd.DataFrame 

data['Condition'] = data['Condition'].astype(str)
data['Time'] = data['Time'].astype(str)

# One-Hot Encode categorical variables
data = pd.get_dummies(data, columns=['Condition', 'Time'], drop_first=False)

print(data.head())

# Drop 'Time_Treatment' to avoid data leakage
X = data.drop(columns=['Condition_15 mM L-Met', 'Condition_20 mM L-Met', 'Condition_10 mM L-Met', 
                       'Condition_H2O Kontrol', 'Condition_HCl Kontrol'])

# Create target variable 'y' using one-hot encoded columns
y = data[['Condition_15 mM L-Met', 'Condition_20 mM L-Met', 'Condition_10 mM L-Met', 
                       'Condition_H2O Kontrol', 'Condition_HCl Kontrol']]

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

plt.savefig('Pigment Tree.png', dpi=300) 

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