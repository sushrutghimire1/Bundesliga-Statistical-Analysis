#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd

def data_profiler(dataset_path):
    # Load the dataset
    df = pd.read_csv(dataset_path)

    # Display basic information
    print("Dataset Overview:")
    print(df.info())

    # Display summary statistics
    print("\nSummary Statistics:")
    print(df.describe())

    # Display missing values
    print("\nMissing Values:")
    print(df.isnull().sum())

    # Display unique values in each column
    print("\nUnique Values:")
    for column in df.columns:
        print(f"{column}: {df[column].nunique()} unique values")

# Replace 'your_dataset.csv' with the actual path to your dataset
data_profiler('Desktop/passes (3).csv')


# In[68]:


from ydata_profiling import ProfileReport
import pandas as pd

df = pd.read_csv("Desktop/passes (3).csv", delimiter=';')
df.head()


# In[58]:


df['winner'] = df.groupby('game_id')['winner'].transform(lambda x: 'Draw' if all(x == 'No') else x)


# In[59]:


df.head()


# In[61]:


import seaborn as sns
import matplotlib.pyplot as plt
import os


# Specify the desktop path
desktop_path = os.path.join(os.path.expanduser("~"), 'Desktop')

# Box Plot
plt.figure(figsize=(10, 6))
sns.boxplot(x='winner', y='passing_quote', data=df, order=['Yes', 'No', 'Draw'])
plt.title('Box Plot of Passing Quote by Winner')
plt.savefig(os.path.join(desktop_path, 'box_plot.png'))
plt.show()

# Bar Chart
plt.figure(figsize=(8, 5))
sns.countplot(x='winner', data=df, order=['Yes', 'No', 'Draw'])
plt.title('Bar Chart of Game Outcomes')
plt.savefig(os.path.join(desktop_path, 'bar_chart.png'))
plt.show()

# Histogram
plt.figure(figsize=(10, 6))
sns.histplot(df['passing_quote'], bins=20, kde=True)
plt.title('Histogram of Passing Quote')
plt.xlabel('Passing Quote')
plt.savefig(os.path.join(desktop_path, 'histogram.png'))
plt.show()


# In[32]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv("Desktop/passes (3).csv", delimiter=';')

# Box Plot
plt.figure(figsize=(10, 6))
sns.boxplot(x='winner', y='passing_quote', data=df, order=['Yes', 'No'])
plt.title('Box Plot of Passing Quote by Winner')
plt.savefig("Desktop/box_plot.png")
plt.show()

# Bar Chart
plt.figure(figsize=(8, 5))
sns.countplot(x='winner', data=df, order=['Yes', 'No'])
plt.title('Bar Chart of Game Outcomes')
plt.savefig("Desktop/bar_chart.png")
plt.show()

# Histogram
plt.figure(figsize=(10, 6))
sns.histplot(df['passing_quote'], bins=20, kde=True)
plt.title('Histogram of Passing Quote')
plt.xlabel('Passing Quote')
plt.savefig("Desktop/histogram.png")
plt.show()

# Quantile Plot
plt.figure(figsize=(10, 6))
sns.catplot(x='winner', y='passing_quote', kind='point', data=df, order=['Yes', 'No'])
plt.title('Quantile Plot of Passing Quote by Winner')
plt.savefig("Desktop/quantile_plot.png")
plt.show()


# In[42]:





# In[99]:


import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Assuming 'df' is your DataFrame containing the dataset
df['winning_team_passing_rate'] = df.apply(lambda row: row['passing_quote'] if row['winner'] == 'Yes' else None, axis=1)
df['losing_team_passing_rate'] = df.apply(lambda row: row['passing_quote'] if row['winner'] == 'No' else None, axis=1)

# Features and target variable
X = df[['winning_team_passing_rate', 'losing_team_passing_rate']]
y = df['winner']

X = X.fillna(0)
X


# In[97]:


# Drop rows with missing values


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the decision tree classifier
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")


# In[6]:


df


# In[66]:


import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Assuming 'df' is your DataFrame containing the dataset

# 1. Handling NaN values after the diff() operation
df['passing_rate_difference'] = df.groupby('game_id')['passing_quote'].diff().abs().dropna()

# 2. Determining draws based on both winner values containing 'No' for the same game_id
draw_game_ids = df[df['winner'] == 'No'].groupby('game_id').filter(lambda group: group['winner'].nunique() == 2)['game_id'].unique()
df['outcome'] = df['game_id'].apply(lambda game_id: 'Draw' if game_id in draw_game_ids else 'Winner')

# Features and target variable
X = df[['passing_rate_difference']]
y = df['outcome']

X


# In[ ]:


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=67)

# Train the decision tree classifier
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")


# In[54]:


y


# In[ ]:


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=34)

# Train the decision tree classifier
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")


# In[36]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X = df[['passing_quote']]
y = df['draw']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)

# Train the decision tree classifier
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")


# In[10]:


from sklearn.datasets import load_iris
from sklearn import tree


# In[19]:


from sklearn.datasets import load_iris
from sklearn import tree
clf = clf.fit(X_test, y_test)
tree.plot_tree(clf)


# In[ ]:




