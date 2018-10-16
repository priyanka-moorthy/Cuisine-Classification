import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import Perceptron
from sklearn.multiclass import OneVsRestClassifier

data = pd.read_json('train.json')
data.head()
data.info()

print("Bar Plot of various cusisines appearance frequency ")
# Bar Plot of various cusisines appearance frequency 
y = data['cuisine'].value_counts()
x = y/y.sum() * 100
y = y.index
sns.barplot(y, x, data=data, palette="BuGn_r")
plt.xticks(rotation=-60)
plt.show()

# The most popular ingredients
n = 6714 # total ingredients
frame= pd.DataFrame(Counter([i for sublist in data.ingredients for i in sublist]).most_common(n))
frame = frame.head(10)
frame

sns.barplot(frame[0], frame[1], palette="gist_heat")
plt.xticks(rotation=-60)
plt.show()

