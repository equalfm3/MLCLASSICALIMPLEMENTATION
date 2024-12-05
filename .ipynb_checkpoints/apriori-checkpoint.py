import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations

def create_transaction_matrix(transactions):
    """
    Create a transaction matrix from a list of transactions.
    """
    return pd.DataFrame(transactions).reset_index().rename(columns={'index': 'TransactionID'})

def one_hot_encode(transactions, unique_items):
    """
    Perform one-hot encoding on the transaction data.
    """
    return pd.DataFrame([[1 if item in transaction else 0 for item in unique_items]
                         for transaction in transactions],
                        columns=unique_items)

def generate_itemsets(items, k):
    """
    Generate itemsets of size k from a list of items.
    """
    return list(combinations(items, k))

def calculate_support(itemset, transactions):
    """
    Calculate the support of an itemset in the transactions.
    """
    count = sum(1 for transaction in transactions if set(itemset).issubset(set(transaction)))
    return count / len(transactions)

def apriori(transactions, min_support, max_length):
    """
    Implement the Apriori algorithm.
    """
    unique_items = list(set(item for transaction in transactions for item in transaction))
    frequent_itemsets = []
    k = 1
    
    while k <= max_length:
        if k == 1:
            itemsets = [[item] for item in unique_items]
        else:
            itemsets = generate_itemsets(unique_items, k)
        
        frequent_k_itemsets = []
        for itemset in itemsets:
            support = calculate_support(itemset, transactions)
            if support >= min_support:
                frequent_k_itemsets.append((itemset, support))
        
        if not frequent_k_itemsets:
            break
        
        frequent_itemsets.extend(frequent_k_itemsets)
        k += 1
    
    return pd.DataFrame(frequent_itemsets, columns=['itemsets', 'support'])

def generate_rules(frequent_itemsets, min_confidence):
    """
    Generate association rules from frequent itemsets.
    """
    rules = []
    for itemset, support in frequent_itemsets.values:
        if len(itemset) > 1:
            for i in range(1, len(itemset)):
                for antecedent in combinations(itemset, i):
                    consequent = tuple(item for item in itemset if item not in antecedent)
                    antecedent_support = frequent_itemsets[frequent_itemsets['itemsets'].apply(lambda x: set(antecedent).issubset(set(x)))]['support'].values[0]
                    confidence = support / antecedent_support
                    if confidence >= min_confidence:
                        lift = confidence / (support / antecedent_support)
                        rules.append((antecedent, consequent, support, confidence, lift))
    
    return pd.DataFrame(rules, columns=['antecedents', 'consequents', 'support', 'confidence', 'lift'])

def apply_apriori(transactions, min_support, min_confidence, max_length=3):
    """
    Apply Apriori algorithm and generate association rules.
    """
    frequent_itemsets = apriori(transactions, min_support, max_length)
    rules = generate_rules(frequent_itemsets, min_confidence)
    return frequent_itemsets, rules

def visualize_rules(rules):
    """
    Visualize association rules.
    """
    plt.figure(figsize=(12, 8))
    plt.scatter(rules['support'], rules['confidence'], alpha=0.5, s=rules['lift']*100)
    plt.xlabel('Support')
    plt.ylabel('Confidence')
    plt.title('Association Rules - Support vs Confidence')

    for i, rule in rules.head(10).iterrows():
        plt.annotate(f"{rule['antecedents']} -> {rule['consequents']}",
                     (rule['support'], rule['confidence']),
                     xytext=(5, 5), textcoords='offset points')

    plt.tight_layout()
    plt.show()

def print_insights(rules):
    """
    Print business insights from the association rules.
    """
    print("\nTop 10 Association Rules by Lift:")
    print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(10))

    print("\nBusiness Insights from Association Rules:")
    print("1. Frequently bought together items:")
    for i, rule in rules.head(5).iterrows():
        print(f"   - {rule['antecedents']} are often bought with {rule['consequents']}")

    print("\n2. Potential for cross-selling:")
    for i, rule in rules.sort_values('confidence', ascending=False).head(5).iterrows():
        print(f"   - When customers buy {rule['antecedents']}, there's a {rule['confidence']:.2f} probability they'll also buy {rule['consequents']}")

    print("\n3. Product placement strategy:")
    print("   - Consider placing frequently associated items near each other in the store or on the website")

    print("\n4. Bundle offers:")
    for i, rule in rules.sort_values('lift', ascending=False).head(5).iterrows():
        print(f"   - Consider creating a bundle offer for {rule['antecedents'] + rule['consequents']}")

if __name__ == "__main__":
    # This section can be used for testing the functions
    pass