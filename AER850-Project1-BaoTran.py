# -*- coding: utf-8 -*-
import pandas as pd                 
import matplotlib.pyplot as plt    
import numpy as np                  
import seaborn as sns

from mpl_toolkits.mplot3d import Axes3D # step 2, 3d plot

from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, f1_score




################################################
######### Step 1: DATA PROCESSING #############
################################################

csv_file = 'Project 1 Data.csv'

# import data
df = pd.read_csv(csv_file)

# display info
print(df.info())

################################################
######### Step 2: DATA VISUALIZATION  ##########
################################################

# ------- Statistical analysis -------------
print(df.describe())                # I was

# -------- 2D Scatter Plots for each pair -------------

fig, axes = plt.subplots(1, 3, figsize=(12, 5))
fig.suptitle('2D Scatter Plots of Maintenance Step', fontsize=16)

# X vs Y
sns.scatterplot(ax=axes[0], data=df, x='X', y='Y', hue='Step', palette='viridis', s=50, alpha=0.3)
axes[0].set_title('X vs Y Coordinates')
axes[0].grid(alpha=0.3)
axes[0].get_legend().remove()

axes[0].set_xlabel('X Coordinate', fontsize=12)
axes[0].set_ylabel('Y Coordinate', fontsize=12)

# X vs Z
sns.scatterplot(ax=axes[1], data=df, x='X', y='Z', hue='Step', palette='viridis', s=50, alpha=0.3)
axes[1].set_title('X vs Z Coordinates')
axes[1].grid(alpha=0.3)
axes[1].get_legend().remove()

axes[1].set_xlabel('X Coordinate', fontsize=12)
axes[1].set_ylabel('Z Coordinate', fontsize=12)

# Y vs Z
sns.scatterplot(ax=axes[2], data=df, x='Y', y='Z', hue='Step', palette='viridis', s=50, alpha=0.3)
axes[2].set_title('Y vs Z Coordinates')
axes[2].grid(alpha=0.3)

axes[2].set_xlabel('Y Coordinate', fontsize=12)
axes[2].set_ylabel('Z Coordinate', fontsize=12)
    
# legend
handles, labels = axes[2].get_legend_handles_labels()
axes[2].get_legend().remove()
fig.legend(handles, labels, title='Steps', bbox_to_anchor=(1.0, 0.85), loc='upper left')

plt.tight_layout(rect=[0, 0, 0.95, 1]) # make room for legend
plt.show()

# -------------- 3D Graph ------------------

DataVivPicture = plt.figure(figsize=(10, 8)) 
ax = DataVivPicture.add_subplot(111, projection='3d')   # Add a 3D subplot (using ax for the axes object)

# Scatter plot using all data points (X, Y, Z) colored by 'Step'
plot = ax.scatter(df['X'], df['Y'], df['Z'], c=df['Step'], cmap='viridis', s=50, alpha=1)

color_bar = plt.colorbar(plot, label='Step')    
ax.set_xlabel('X Coordinate', fontsize=12)
ax.set_ylabel('Y Coordinate', fontsize=12)
ax.set_zlabel('Z Coordinate', fontsize=12)

ax.set_title('3D Visualization of Maintenance Steps', fontsize=14, fontweight='bold')

plt.tight_layout() 
plt.show()

# -------------- Histogarms ------------------
plt.figure(figsize=(10, 6))
# Use countplot for categorical data distribution visualization
sns.countplot(x='Step', data=df, color='blue') 
plt.title('Frequency of Maintenance Steps', fontsize=14, fontweight='bold')
plt.xlabel('Frequency of Data Points per Maintenance Step', fontsize=12)
plt.ylabel('Number of Samples', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

################################################
######### Step 3: CORRELATION ANAYSIS  #########
################################################

# printed out version
correlation_matrix = df.corr(method='pearson')
print("\nPearson Correlation Matrix:")
print(correlation_matrix)

# plot version
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, linecolor='black')
plt.title('Correlation Matrix (Coordinates vs Maintenance Steps)')
plt.show()

########################################################################
######### Step 4: CLASSIFICATION MODEL DEVELOPMENT/ENGINEERING #########
########################################################################

X = df[['X', 'Y', 'Z']]
y = df['Step']

# 80% train, 20% test sets
splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in splitter.split(X, y):
    # Use .iloc to slice the DataFrames/Series based on the generated indices
    X_train = X.iloc[train_index]
    X_test = X.iloc[test_index]
    y_train = y.iloc[train_index]
    y_test = y.iloc[test_index]

print(f"Training data size: {len(X_train)} samples, Testing data size: {len(X_test)} samples.")

# Scale the data (Crucial for distance-based algorithms like KNN)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

############### 3 MODELS ################
results = {}
best_estimators = {}

# 1. KNN with GridSearchCV
knn = KNeighborsClassifier()
knn_params = {'n_neighbors': range(1, 15), 'weights': ['uniform', 'distance']}
knn_grid = GridSearchCV(knn, knn_params, cv=5, scoring='f1_weighted')
knn_grid.fit(X_train_scaled, y_train)
best_estimators['KNN'] = knn_grid.best_estimator_
print(f"\nKNN Best Hyperparameters (GridSearch): {knn_grid.best_params_}")

# 2. Decision Tree with GridSearchCV
dtc = DecisionTreeClassifier(random_state=42)
dtc_params = {'max_depth': range(1, 10), 'min_samples_split': [2, 5, 10]}
dtc_grid = GridSearchCV(dtc, dtc_params, cv=5, scoring='f1_weighted')
dtc_grid.fit(X_train_scaled, y_train)
best_estimators['DTC'] = dtc_grid.best_estimator_
print(f"DTC Best Hyperparameters (GridSearch): {dtc_grid.best_params_}")

# 3. Random Forest with GridSearchCV
rfc = RandomForestClassifier(random_state=42)
# Limiting the parameter space for speed
rfc_params = {'n_estimators': [50, 100, 150], 'max_depth': [5, 10, None]} 
rfc_grid = GridSearchCV(rfc, rfc_params, cv=5, scoring='f1_weighted')
rfc_grid.fit(X_train_scaled, y_train)
best_estimators['RFC_Grid'] = rfc_grid.best_estimator_
print(f"RFC Best Hyperparameters (GridSearch): {rfc_grid.best_params_}")

########### Random Forest with RandomizedSearchCV ###############
rfc_random = RandomForestClassifier(random_state=42)
rfc_random_params = {'n_estimators': np.arange(50, 200),'max_depth': np.arange(3, 15),'min_samples_split': np.arange(2, 11)}
rfc_rand_search = RandomizedSearchCV(rfc_random, rfc_random_params, n_iter=10, cv=5, scoring='f1_weighted', random_state=42)
rfc_rand_search.fit(X_train_scaled, y_train)
best_estimators['RFC_Random'] = rfc_rand_search.best_estimator_
print(f"RFC Best Hyperparameters (RandomizedSearch): {rfc_rand_search.best_params_}")

########################################################################
#################### Step 5: MODEL PERFORMANCE ANALYSIS ################
########################################################################

final models

birojeabjioerbijoeijorijoregioerjgioaregjaeijorgjraiofeijoafreijofrijoarEFijofreijofrijofaewijoferwiojfewiojiijo
WEPAWIJEPWAEawewAEAWEaw CRAHSIGN CRAHEIOERATGO OTEAUH