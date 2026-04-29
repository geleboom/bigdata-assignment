import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules

# -------------------------------
#  Load Dataset 
# -------------------------------
transactions = []

with open("market_basket_large.csv", "r") as file:
    for line in file:
        transaction = line.strip().split(",")
        transactions.append(transaction)

# -------------------------------
# Encoding
# -------------------------------
te = TransactionEncoder()
te_data = te.fit(transactions).transform(transactions)
df_encoded = pd.DataFrame(te_data, columns=te.columns_)

print("✅ Dataset loaded successfully!")
print(f"Total transactions: {len(transactions)}")
print("-" * 50)

# -------------------------------
#  APRIORI
# -------------------------------
frequent_apriori = apriori(df_encoded, min_support=0.02, use_colnames=True)

rules_apriori = association_rules(
    frequent_apriori,
    metric="lift",
    min_threshold=1
)

# Sort by Lift
rules_apriori = rules_apriori.sort_values(by='lift', ascending=False)

# -------------------------------
#  FP-GROWTH
# -------------------------------
frequent_fp = fpgrowth(df_encoded, min_support=0.02, use_colnames=True)

rules_fp = association_rules(
    frequent_fp,
    metric="lift",
    min_threshold=1
)

# Sort by Lift
rules_fp = rules_fp.sort_values(by='lift', ascending=False)

# -------------------------------
#  Display Top Rules
# -------------------------------
print("\n🔥 TOP 10 APRIORI RULES (by Lift)")
print("-" * 50)

for i, row in rules_apriori.head(10).iterrows():
    print(f"{set(row['antecedents'])} → {set(row['consequents'])}")
    print(f"Support: {row['support']:.3f} | Confidence: {row['confidence']:.3f} | Lift: {row['lift']:.3f}")
    print("-" * 40)

print("\n🔥 TOP 10 FP-GROWTH RULES (by Lift)")
print("-" * 50)

for i, row in rules_fp.head(10).iterrows():
    print(f"{set(row['antecedents'])} → {set(row['consequents'])}")
    print(f"Support: {row['support']:.3f} | Confidence: {row['confidence']:.3f} | Lift: {row['lift']:.3f}")
    print("-" * 40)

# -------------------------------
#  VISUALIZATION
# -------------------------------

top_rules = rules_fp.head(10).copy()


def clean_set(s):
    return ', '.join(list(s))

top_rules['rule'] = top_rules.apply(
    lambda row: f"{clean_set(row['antecedents'])} → {clean_set(row['consequents'])}",
    axis=1
)


plt.figure(figsize=(10, 6))

# Plot
plt.barh(top_rules['rule'], top_rules['lift'])

plt.xlabel("Lift")
plt.title("Top 10 Association Rules (FP-Growth)")

# Invert y-axis so best rule is on top
plt.gca().invert_yaxis()

# Add space for long labels
plt.tight_layout()

plt.show()