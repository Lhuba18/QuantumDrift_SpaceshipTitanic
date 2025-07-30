#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection   import train_test_split, GridSearchCV
from sklearn.preprocessing    import StandardScaler
from sklearn.linear_model     import LogisticRegression
from sklearn.ensemble         import RandomForestClassifier
import xgboost                 as xgb
from sklearn.metrics          import (
    confusion_matrix, roc_curve, roc_auc_score,
    classification_report
)

# Seaborn styling
sns.set(style="whitegrid")

# Load your engineered training set (adjust path if needed)
df = pd.read_csv('../output/engineered_train.csv')
X  = df.drop(columns=['PassengerId','Transported'])
y  = df['Transported'].astype(int)

# Quick peek
df.head()



# In[3]:


# 1) Train/validation split
X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# 2) Scale numeric features for LR & XGBoost
num_cols = ['Age','CabinNum','TotalSpend','GroupSize']
scaler   = StandardScaler().fit(X_train[num_cols])

X_train[num_cols] = scaler.transform(X_train[num_cols])
X_val[num_cols]   = scaler.transform(X_val[num_cols])

print("Train shape:", X_train.shape, "Val shape:", X_val.shape)



# In[6]:


# -- Logistic Regression --
lr_params = {'C': [0.01,0.1,1,10], 'solver': ['liblinear']}
lr_grid = GridSearchCV(
    LogisticRegression(max_iter=500),
    lr_params, cv=5, scoring='f1'
)
lr_grid.fit(X_train, y_train)

# -- Random Forest --
xgb_params = {
    'n_estimators':   [100, 200],
    'max_depth':      [3, 5],
    'learning_rate':  [0.01, 0.1]
}

xgb_grid = GridSearchCV(
    xgb.XGBClassifier(
        # no use_label_encoder parameter any more
        eval_metric='logloss',
        random_state=42
    ),
    xgb_params,
    cv=5,
    scoring='f1',
    n_jobs=-1
)
xgb_grid.fit(X_train, y_train)

print("Best XGB params:", xgb_grid.best_params_)



# In[7]:


# pick your champion model
best_model = xgb_grid.best_estimator_

# predictions
y_pred = best_model.predict(X_val)
y_prob = best_model.predict_proba(X_val)[:, 1]

# 1) Confusion Matrix
cm = confusion_matrix(y_val, y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix – XGBoost')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig('../plots/confusion_matrix.png')
plt.show()
plt.close()

# 2) ROC Curve
fpr, tpr, _ = roc_curve(y_val, y_prob)
auc_score   = roc_auc_score(y_val, y_prob)
plt.figure(figsize=(5,4))
plt.plot(fpr, tpr, label=f'AUC = {auc_score:.3f}')
plt.plot([0,1], [0,1], '--', linewidth=1)
plt.title('ROC Curve – XGBoost')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig('../plots/roc_curve.png')
plt.show()
plt.close()

# 3) Feature Importance
importances = pd.Series(
    best_model.feature_importances_,
    index=X.columns
)
top10 = importances.sort_values(ascending=False).head(10)
plt.figure(figsize=(6,4))
sns.barplot(x=top10.values, y=top10.index)
plt.title('Top 10 Feature Importances – XGBoost')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.tight_layout()
plt.savefig('../plots/feature_importance.png')
plt.show()
plt.close()

print("✅ Plots saved to ../plots/")


# In[ ]:




