#!/usr/bin/env python
# coding: utf-8

# In[79]:


import pandas as pd

df = pd.read_csv("Desktop/passes (3).csv", delimiter=';')
df.head()
df = df.dropna()


# In[80]:


df = df.groupby('game_id').filter(lambda group: not all(group['winner'] == 'No'))

# Now 'filtered_df' contains only groups where at least one 'winner' value is not 'No'


# In[81]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt
import seaborn as sns


# Part a: Descriptive Analysis - Decision Tree Visualization
X = df[['passing_quote']]
y = df['winner'].apply(lambda x: 1 if x == 'Yes' else 0)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build a decision tree classifier
tree_classifier = DecisionTreeClassifier(random_state=42)
tree_classifier.fit(X_train, y_train)



# Part b: Hypothesis Testing
# Winner's Passing Rate vs. Loser's Passing Rate
winners_passing_rate = df[df['winner'] == 'Yes']['passing_quote']
losers_passing_rate = df[df['winner'] == 'No']['passing_quote']

t_statistic, p_value = ttest_ind(winners_passing_rate, losers_passing_rate)
print(f"\nHypothesis Test 1 - Winner's Passing Rate vs. Loser's Passing Rate:")
print(f"T-statistic: {t_statistic}")
print(f"P-value: {p_value}")
print("Result:", "Reject Null Hypothesis" if p_value < 0.05 else "Fail to Reject Null Hypothesis")

# Difference in Passing Rate in Games with Winners vs. Draw
winners_passing_rate_diff = df[df['winner'] == 'Yes']['passing_quote'].diff().dropna()
draw_passing_rate_diff = df[df['winner'] == 'No']['passing_quote'].diff().dropna()

t_statistic, p_value = ttest_ind(winners_passing_rate_diff, draw_passing_rate_diff)
print(f"\nHypothesis Test 2 - Difference in Passing Rate in Games with Winners vs. Draw:")
print(f"T-statistic: {t_statistic}")
print(f"P-value: {p_value}")
print("Result:", "Reject Null Hypothesis" if p_value < 0.05 else "Fail to Reject Null Hypothesis")

