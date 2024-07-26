import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori
from mlxtend.preprocessing import TransactionEncoder

# Load the dataset
# Example data in the form of a list of transactions
dataset = [['milk', 'bread', 'butter'],
           ['bread', 'butter'],
           ['milk', 'bread'],
           ['milk', 'butter'],
           ['bread', 'butter']]

# Transform the dataset into a DataFrame suitable for the eclat function
te = TransactionEncoder()
te_ary = te.fit_transform(dataset)
df = pd.DataFrame(te_ary, columns=te.columns_)

# Apply Eclat algorithm
frequent_itemsets = apriori(df, min_support=0.6, use_colnames=True)

# Display frequent itemsets
print("Frequent Itemsets using Eclat:")
print(frequent_itemsets)

# Plot the frequent itemsets
plt.figure(figsize=(10, 7))
plt.bar(frequent_itemsets['itemsets'].astype(str), frequent_itemsets['support'])
plt.xlabel('Itemsets')
plt.ylabel('Support')
plt.title('Frequent Itemsets using Eclat')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('eclat_frequent_itemsets.png')
plt.close()
