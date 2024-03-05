#!/usr/bin/env python
# coding: utf-8

# In[55]:


import pandas as pd
from sklearn.tree import DecisionTreeClassifier


# In[56]:


df = pd.read_csv('passes (3).csv',delimiter=';')


# In[57]:


df.size


# In[58]:


import pandas as pd

# Assuming your dataframe is named df
df['winner'] = df.groupby('game_id')['winner'].transform(lambda x: 'Draw' if all(x == 'No') else x)

# Now you can drop duplicates to keep unique 'game_id' entries
# df = df.drop_duplicates(subset='game_id')

# Print the modified dataframe
print(df.size)


# In[59]:


from scipy.stats import ttest_ind

winners_passing_rates = df[df['winner'] == 'Yes']['passing_quote']
losers_passing_rates = df[df['winner'] == 'No']['passing_quote']



# In[60]:


losers_passing_rates


# In[61]:


t_stat, p_value = ttest_ind(winners_passing_rates, losers_passing_rates)

if p_value < 0.05:
    print("There is a significant difference between the passing rates of winners and losers.")
    if t_stat > 0:
        print("Winners have a higher passing rate than losers.")
    else:
        print("Losers have a higher passing rate than winners.")
else:
    print("There is no significant difference between the passing rates of winners and losers.")


# In[62]:


t_stat, p_value = ttest_ind(winners_passing_rates, draw_passing_rates)

if p_value < 0.05:
    print("There is a significant difference in passing rates between games with a winner and games that ended in a draw.")
    if t_stat > 0:
        print("Games with a winner have a higher passing rate than games that ended in a draw.")
    else:
        print("Games that ended in a draw have a higher passing rate than games with a winner.")
else:
    print("There is no significant difference in passing rates between games with a winner and games that ended in a draw.")


# In[63]:


# Create a violin plot
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6))
sns.violinplot(x='winner', y='passing_quote', data=df)
plt.title('Passing Rates: Winners, Losers and Draws')
plt.xlabel('Winner')
plt.ylabel('Passing Rate')
plt.show()


# In[68]:


# Create a swarm plot
plt.figure(figsize=(10, 6))
sns.swarmplot(x='winner', y='passing_quote', data=df[(df['winner'] == 'Yes') | (df['winner'] == 'Draw')])
plt.title('Passing Rates: Winners vs. Draws')
plt.xlabel('Winner')
plt.ylabel('Passing Rate')
plt.show()


# In[69]:


# Calculate means
mean_passing_rates = df.groupby('winner')['passing_quote'].mean()

# Create a bar plot
plt.figure(figsize=(8, 6))
sns.barplot(x=mean_passing_rates.index, y=mean_passing_rates.values)
plt.title('Average Passing Rates: Winners vs. Losers vs. Draws')
plt.xlabel('Winner')
plt.ylabel('Average Passing Rate')
plt.show()

