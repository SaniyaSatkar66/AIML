import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
import re 
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules 
from mlxtend.preprocessing import TransactionEncoder 
from mpl_toolkits.mplot3d import Axes3D
import networkx as nx
basket = pd.read_csv("Groceries_dataset.csv") 
print(basket.head()) 
## Grouping into transactions
basket.itemDescription = basket.itemDescription.transform(lambda x: [x]) 
basket=basket.groupby(['Member_number','Date']).sum()['itemDescription'].reset_index(drop=True)
encoder = TransactionEncoder() 
##Apriori and Association Rules
transactions = pd.DataFrame(encoder.fit(basket).transform(basket), columns=encoder.columns_) 
print(transactions.head())  
 ##Apriori and Association Rules##
frequent_itemsets = apriori(transactions, min_support= 6/len(basket), use_colnames=True, max_len = 2)
rules = association_rules(frequent_itemsets, metric="lift", min_threshold = 1.5) 
print(rules.head()) 
print("Rules identified: ", len(rules))
 ##Visual
sns.set(style = "whitegrid")
fig = plt.figure(figsize=(12, 12)) 
ax = fig.add_subplot(projection = '3d')
x = rules['support'] 
y = rules['confidence']
z = rules['lift'] 
ax.set_xlabel("Support") 
ax.set_ylabel("Confidence")
ax.set_zlabel("Lift") 
ax.scatter(x, y, z) 
ax.set_title("3D Distribution of Association Rules")
plt.show()