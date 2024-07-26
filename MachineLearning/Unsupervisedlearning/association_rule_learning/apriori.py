import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

# Load the dataset
# Example data in the form of a list of transactions
dataset = [['milk', 'bread', 'butter'],
           ['bread', 'butter'],
           ['milk', 'bread'],
           ['milk', 'butter'],
           ['bread', 'butter']]

# Transform the dataset into a DataFrame suitable for the apriori function
te = TransactionEncoder()
te_ary = te.fit_transform(dataset)
df = pd.DataFrame(te_ary, columns=te.columns_)

# Apply Apriori algorithm
frequent_itemsets = apriori(df, min_support=0.6, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)

# Display frequent itemsets and rules
print("Frequent Itemsets:")
print(frequent_itemsets)
print("\nAssociation Rules:")
print(rules)

# Plot the frequent itemsets
plt.figure(figsize=(10, 7))
plt.bar(frequent_itemsets['itemsets'].astype(str), frequent_itemsets['support'])
plt.xlabel('Itemsets')
plt.ylabel('Support')
plt.title('Frequent Itemsets using Apriori')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('apriori_frequent_itemsets.png')
plt.close()
