
# %% [markdown]
# # 🚀 Spaceship Titanic: Data Exploration Notebook
# This notebook performs visual data exploration for the Spaceship Titanic dataset.

# %% [markdown]
# ## 1. Import Libraries

# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

# %% [markdown]
# ## 2. Load and Clean the Data

# %%
df = pd.read_csv("output/cleaned_train.csv")

# Fill and convert types early to avoid visual errors
df['CryoSleep'] = df['CryoSleep'].fillna(False).astype(bool)
df['VIP'] = df['VIP'].fillna(False).astype(bool)
df['Age'] = df['Age'].fillna(df['Age'].median())
df['TotalSpend'] = df[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']].sum(axis=1)

# %% [markdown]
# ## 3. Histogram: Age Distribution

# %%
plt.figure(figsize=(8, 5))
sns.histplot(df['Age'], bins=30, kde=True)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Count")
plt.savefig("output/plot_age_histogram.png")
plt.show()

# %% [markdown]
# ## 4. Boxplot: TotalSpend

# %%
plt.figure(figsize=(8, 3))
sns.boxplot(x=df['TotalSpend'])
plt.title("Boxplot of TotalSpend")
plt.savefig("output/plot_totalspend_boxplot.png")
plt.show()

# %% [markdown]
# ## 5. Correlation Matrix

# %%
numeric_df = df.select_dtypes(include=np.number)
corr = numeric_df.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Matrix")
plt.savefig("output/plot_correlation_matrix.png")
plt.show()

# %% [markdown]
# ## 6. Bar Charts for Categorical Distributions

# %%
plt.figure(figsize=(6, 4))
sns.countplot(x='CryoSleep', data=df)
plt.title("CryoSleep Distribution")
plt.savefig("output/plot_cryo_count.png")
plt.show()

plt.figure(figsize=(6, 4))
sns.countplot(x='VIP', data=df)
plt.title("VIP Distribution")
plt.savefig("output/plot_vip_count.png")
plt.show()

# %% [markdown]
# ## 7. Stacked Bar Chart: Transported vs VIP

# %%
vip_transport = pd.crosstab(df['VIP'], df['Transported'], normalize='index')

vip_transport.plot(kind='bar', stacked=True, colormap='coolwarm')
plt.title("Transported vs VIP Status")
plt.xlabel("VIP")
plt.ylabel("Proportion")
plt.legend(title='Transported')
plt.savefig("output/plot_vip_vs_transported.png")
plt.show()
