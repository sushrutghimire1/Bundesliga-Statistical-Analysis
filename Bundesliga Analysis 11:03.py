#!/usr/bin/env python
# coding: utf-8

# In[261]:


import pandas as pd

df = pd.read_csv("Desktop/passes (3).csv", delimiter=';')
df


# In[262]:


df = df.dropna()


# In[263]:


df.describe()


# In[264]:


df.game_id.nunique()


# In[265]:


df['winner'] = df.groupby('game_id')['winner'].transform(lambda x: 'Draw' if all(x == 'No') else x)


# In[266]:


df[df['winner']=='Yes']['passing_quote'].describe()


# In[267]:


df[df['winner']=='No']['passing_quote'].describe()  


# In[268]:


df[df['winner']=='Draw']['passing_quote'].describe()


# In[269]:


import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm
import numpy as np


# Set the mean and standard deviation
mean = df[df['winner']=='Yes']['passing_quote'].mean()
std_dev = df[df['winner']=='Yes']['passing_quote'].std()

# Generate data points for the normal distribution curve
x = np.linspace(mean - 3*std_dev, mean + 3*std_dev, 1000)
y = norm.pdf(x, mean, std_dev)

# Plot the normal distribution curve using Seaborn
sns.histplot(df[df['winner']=='Yes']['passing_quote'], kde=False, stat='density', color='green', label='Histogram')
plt.plot(x, y, label='Normal Distribution', color='blue')

# Add labels and a legend
plt.xlabel('Passing Quote of Winners')
plt.ylabel('Probability Density Function')
plt.title('Normal Distribution for Passing Quote of Winners')
plt.legend()

# Show the plot
plt.show()


# In[270]:


# Set the mean and standard deviation
mean = df[df['winner']=='No']['passing_quote'].mean()
std_dev = df[df['winner']=='No']['passing_quote'].std()

# Generate data points for the normal distribution curve
x = np.linspace(mean - 3*std_dev, mean + 3*std_dev, 1000)
y = norm.pdf(x, mean, std_dev)

# Plot the normal distribution curve using Seaborn
sns.histplot(df[df['winner']=='No']['passing_quote'], kde=False, stat='density', color='red', label='Histogram')
plt.plot(x, y, label='Normal Distribution', color='blue')

# Add labels and a legend
plt.xlabel('Passing Quote of Losers')
plt.ylabel('Probability Density Function')
plt.title('Normal Distribution of Passing Quote for Losers')
plt.legend()

# Show the plot
plt.show()


# In[271]:


# Set the mean and standard deviation
mean = df[df['winner']=='Draw']['passing_quote'].mean()
std_dev = df[df['winner']=='Draw']['passing_quote'].std()

# Generate data points for the normal distribution curve
x = np.linspace(mean - 3*std_dev, mean + 3*std_dev, 1000)
y = norm.pdf(x, mean, std_dev)

# Plot the normal distribution curve using Seaborn
sns.histplot(df[df['winner']=='Draw']['passing_quote'], kde=False, stat='density', color='orange', label='Histogram')
plt.plot(x, y, label='Normal Distribution', color='blue')

# Add labels and a legend
plt.xlabel('Passing Quote of Draws')
plt.ylabel('Probability Density Function')
plt.title('Normal Distribution of Passing Quote for Draws')
plt.legend()

# Show the plot
plt.show()


# In[272]:


import seaborn as sns
import matplotlib.pyplot as plt



# Create a box plot using Seaborn
sns.boxplot(x='winner', y='passing_quote', data=df)

# Add labels and a title
plt.xlabel('Is winner')
plt.title('Box Plot of Passing Quote')

# Show the plot
plt.show()


# In[273]:


df.sort_values(['game_id', 'winner'], inplace=True)

# Calculate passing rate difference for both winner and loser rows within each game
df['passing_rate_diff'] = df.groupby('game_id')['passing_quote'].diff()

# Filter data for each scenario
lower_passing_won = df.query('winner == "Yes" and passing_rate_diff < 0')
greater_equal_passing_won = df.query('winner == "Yes" and passing_rate_diff >= 0')
lower_passing_won_abs = lower_passing_won
lower_passing_won_abs['passing_rate_diff'] = lower_passing_won_abs['passing_rate_diff'].abs()

# Create box plots for each scenario
plt.figure(figsize=(12, 6))

# Box plot for games that had a greater passing rate than losers but still lost
plt.subplot(1, 2, 1)
sns.boxplot(x='winner', y='passing_rate_diff', data=lower_passing_won_abs)
plt.title('Passing Quote difference (Winners lower than Losers)')

# Box plot for games that had a greater/equal passing rate than losers and won
plt.subplot(1, 2, 2)
sns.boxplot(x='winner', y='passing_rate_diff', data=greater_equal_passing_won)
plt.title('Passing quote difference (Winners highers than Losers)')

# Adjust layout
plt.tight_layout()

# Show the plots
plt.show()


# In[274]:


import matplotlib.pyplot as plt
import seaborn as sns


lower_passing_lost = df.query('winner == "Yes" and passing_rate_diff < 0')
greater_equal_passing_lost = df.query('winner == "Yes" and passing_rate_diff >= 0')
lower_passing_lost_abs = lower_passing_lost
lower_passing_lost_abs['passing_rate_diff'] = lower_passing_lost_abs['passing_rate_diff'].abs()
# Aggregate data for each scenario
scenario_counts = {
    'Lower Passing Rate': len(lower_passing_lost_abs),
    'Higher Passing Rate': len(greater_equal_passing_lost)
}

# Create bar chart
plt.figure(figsize=(3, 6))
sns.barplot(x=list(scenario_counts.keys()), y=list(scenario_counts.values()))
plt.xlabel('Passing rate difference')
plt.ylabel('Count of Games won')
plt.title('Counts of Games won Winners compared to Losers')
plt.xticks(rotation=45, ha='right')

# Show the plot
plt.tight_layout()
plt.show()


# In[275]:


df


# In[276]:


import matplotlib.pyplot as plt
import seaborn as sns


draw_difference = df.query('winner == "Draw"')
win_difference = df.query('winner == "Yes"')
draw_difference['passing_rate_diff'] = draw_difference['passing_rate_diff'].abs()
win_difference['passing_rate_diff'] = win_difference['passing_rate_diff'].abs()



# Create box plots for each scenario
plt.figure(figsize=(12, 6))

# Box plot for games that had a greater passing rate than losers but still lost
plt.subplot(1, 2, 1)
sns.boxplot(x='winner', y='passing_rate_diff', data=win_difference)
plt.title('Win')

# Box plot for games that had a greater/equal passing rate than losers and won
plt.subplot(1, 2, 2)
sns.boxplot(x='winner', y='passing_rate_diff', data=draw_difference)
plt.title('Draw')

# Adjust layout
plt.tight_layout()

# Show the plots
plt.show()


# In[277]:


lower_passing_won_abs['game_id'].size


# In[278]:


greater_equal_passing_won['game_id'].size


# In[279]:


47/(47+67)


# In[280]:


67/(47+67)


# In[281]:


58-41


# In[282]:


plt.figure(figsize=(10, 6))

# Bar chart for games that had a greater passing rate than losers but still lost
sns.barplot(x='game_id', y='passing_rate_diff', data=lower_passing_won_abs, hue='winner')
plt.title('Greater Passing Rate Than Losers but Still Lost')

# Adjust layout
plt.tight_layout()

# Show the plot
plt.show()


# In[283]:


plt.figure(figsize=(10, 6))

# Bar chart for games that had a greater passing rate than losers but still lost
sns.barplot(x='game_id', y='passing_rate_diff', data=greater_equal_passing_won, hue='winner')
plt.title('Greater/Equal Passing Rate compared to Losers and Won')

# Adjust layout
plt.tight_layout()

# Show the plot
plt.show()


# In[284]:


import warnings

# To filter out all warnings
warnings.filterwarnings("ignore")

# Or, to filter out specific types of warnings
# warnings.filterwarnings("ignore", category=SomeSpecificWarning)

# Your code here

# Reset the warning filter if needed
# warnings.resetwarnings()


# In[285]:


# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# # Assuming you have a DataFrame 'df' with columns 'game_id', 'passing_quote', and 'winner'

# # Data preparation
# # Assuming you have a binary 'winner' column where 'Yes' indicates a win and 'No' indicates a loss
# y = df['winner'].apply(lambda x: 1 if x == 'Yes' else 0)
# df['passing_rate_gt_80'] = df['passing_quote'] < 80
# df['passing_rate_gt_85'] = df['passing_quote'] < 85
# df['passing_rate_gt_70'] = df['passing_quote'] < 70
# df['passing_rate_gt_75'] = df['passing_quote'] < 75
# df['passing_rate_gt_90'] = df['passing_quote'] < 90
# df['passing_rate_gt_60'] = df['passing_quote'] < 60
# df['passing_rate_gt_65'] = df['passing_quote'] < 65
# df['passing_rate_gt_95'] = df['passing_quote'] < 95



# df['passing_rate_gt_80'] = df['passing_rate_gt_80'].apply(lambda x: 1 if x == True else 0)
# df['passing_rate_gt_70'] = df['passing_rate_gt_70'].apply(lambda x: 1 if x == True else 0)
# df['passing_rate_gt_75'] = df['passing_rate_gt_75'].apply(lambda x: 1 if x == True else 0)
# df['passing_rate_gt_85'] = df['passing_rate_gt_85'].apply(lambda x: 1 if x == True else 0)
# df['passing_rate_gt_90'] = df['passing_rate_gt_90'].apply(lambda x: 1 if x == True else 0)
# df['passing_rate_gt_60'] = df['passing_rate_gt_60'].apply(lambda x: 1 if x == True else 0)
# df['passing_rate_gt_65'] = df['passing_rate_gt_65'].apply(lambda x: 1 if x == True else 0)
# df['passing_rate_gt_95'] = df['passing_rate_gt_95'].apply(lambda x: 1 if x == True else 0)



# # Splitting the data
# X = df[['passing_quote','passing_rate_gt_70','passing_rate_gt_80','passing_rate_gt_75',
#         'passing_rate_gt_85','passing_rate_gt_90','passing_rate_gt_60','passing_rate_gt_65','passing_rate_gt_95']]
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Logistic Regression model
# logreg = LogisticRegression()
# logreg.fit(X_train, y_train)

# # Predictions
# y_pred = logreg.predict(X_test)

# # Model evaluation
# accuracy = accuracy_score(y_test, y_pred)
# conf_matrix = confusion_matrix(y_test, y_pred)
# classification_rep = classification_report(y_test, y_pred)

# # Coefficient interpretation
# coefficients = logreg.coef_[0]
# intercept = logreg.intercept_[0]

# # Feature importance
# feature_importance = logreg.coef_[0]

# print(f'Coefficients: {coefficients}')
# print(f'Intercept: {intercept}')

# # Displaying results
# print(f'Accuracy: {accuracy}')
# print(f'Confusion Matrix:\n{conf_matrix}')
# print(f'Classification Report:\n{classification_rep}')

# # Plotting feature importance
# fig, ax = plt.subplots(figsize=(10, 6))
# plt.xticks(rotation=45)  # Adjust the rotation angle as needed
# ax.bar(X.columns, feature_importance)
# ax.set_ylabel('Coefficient Value')
# ax.set_title('Feature Importance Plot')
# plt.show()


# In[286]:


df.game_id.nunique()


# In[287]:


df.winner.nunique()


# In[288]:


df


# In[289]:


import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import plot_tree
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming you have a DataFrame 'df' with columns 'game_id', 'passing_quote', and 'winner'

# Data preparation
# Assuming you have a binary 'winner' column where 'Yes' indicates a win and 'No' indicates a loss
y = df['winner'].apply(lambda x: 0 if x == 'Yes' else (1 if x == 'No' else 2))

df2=df
df2['passing_rate_gt_80'] = (df['passing_quote'] > 80).astype(int)
df2['passing_rate_gt_85'] = (df['passing_quote'] > 85).astype(int)
df2['passing_rate_gt_70'] = (df['passing_quote'] > 70).astype(int)
df2['passing_rate_gt_75'] = (df['passing_quote'] > 75).astype(int)
df2['passing_rate_gt_90'] = (df['passing_quote'] > 90).astype(int)
df2['passing_rate_gt_60'] = (df['passing_quote'] > 60).astype(int)
df2['passing_rate_gt_65'] = (df['passing_quote'] > 65).astype(int)
df2['passing_rate_gt_95'] = (df['passing_quote'] > 95).astype(int)

# Splitting the data
X = df2[['passing_quote', 'passing_rate_gt_70', 'passing_rate_gt_80', 'passing_rate_gt_75',
        'passing_rate_gt_85', 'passing_rate_gt_90', 'passing_rate_gt_60', 'passing_rate_gt_65', 'passing_rate_gt_95',
        'passing_rate_diff']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=23)

# XGBoost model
xgb_model = xgb.XGBClassifier(
    learning_rate=0.1,
    objective='multi:softmax',
    num_class=3,
    n_estimators=100,
    max_depth=3,
    min_child_weight=1,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0,
    scale_pos_weight=1,
    reg_alpha=0,
    reg_lambda=1
)

xgb_model.fit(X_train, y_train)

# Predictions
y_pred = xgb_model.predict(X_test)

# Model evaluation
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

# Displaying results
print(f'Accuracy: {accuracy}')
print(f'Confusion Matrix:\n{conf_matrix}')
print(f'Classification Report:\n{classification_rep}')

# Plotting feature importance
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x=X.columns, y=xgb_model.feature_importances_, ax=ax)
plt.xticks(rotation=45)  # Adjust the rotation angle as needed
ax.set_ylabel('Feature Importance')
ax.set_title('Feature Importance Plot')
plt.show()

# Plotting decision tree diagram
plt.figure(figsize=(20, 10))
plot_tree(xgb_model, num_trees=0, rankdir='LR')
plt.show()


# In[299]:


from xgboost import plot_tree

# Plotting decision tree diagram
plt.figure(figsize=(20, 10))  # Adjust the size as needed
plot_tree(xgb_model, num_trees=50, rankdir='LR', ax=plt.gca())
plt.show()



# In[291]:


df['winner']


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[292]:


df = pd.read_csv("Desktop/passes (3).csv", delimiter=';')
df = df.dropna()
df = df.groupby('game_id').filter(lambda group: not all(group['winner'] == 'No'))

winners_passing_rate = df[df['winner'] == 'Yes']['passing_quote']
losers_passing_rate = df[df['winner'] == 'No']['passing_quote']

t_statistic, p_value = ttest_ind(winners_passing_rate, losers_passing_rate)
print(f"\nHypothesis Test 1 - Winner's Passing Rate vs. Loser's Passing Rate:")
print(f"T-statistic: {t_statistic}")
print(f"P-value: {p_value}")
print("Result:", "Reject Null Hypothesis" if p_value < 0.05 else "Fail to Reject Null Hypothesis")



# In[ ]:


import pandas as pd
from scipy.stats import mannwhitneyu

# Assuming 'passing_quote' is the column containing passing rates and 'winner' is the column indicating game winners
df = pd.read_csv("Desktop/passes (3).csv", delimiter=';')
df = df.dropna()
df['winner'] = df.groupby('game_id')['winner'].transform(lambda x: 'Draw' if all(x == 'No') else x)

winners_passing_rate = df[df['winner'] == 'Yes']['passing_quote']
draws_passing_rate = df[df['winner'] == 'Draw']['passing_quote']

# Mann-Whitney U test
u_statistic, p_value = mannwhitneyu(winners_passing_rate, draws_passing_rate)

# Display results
print("\nMann-Whitney U Test - Difference in Passing Rate between Games with Winners and Draws:")
print(f"U-Statistic: {u_statistic}")
print(f"P-value: {p_value}")
print("Result:", "Reject Null Hypothesis" if p_value < 0.05 else "Fail to Reject Null Hypothesis")

