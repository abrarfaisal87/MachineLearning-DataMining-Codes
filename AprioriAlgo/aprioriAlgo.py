import pandas as pd
from apyori import apriori

store_data = pd.read_csv(r'D:\ML,NN,IP\MachineLearning-DataMining-Codes\AprioriAlgo\Online Retail.csv', header=None)

num_records = len(store_data)
print(num_records)

records = []
for i in range(0, num_records):
    records.append([str(store_data.values[i, j])for j in range(0, 5)])

association_rules = apriori(records, min_support=0.5, min_confidence=0.7, min_lift=1.2, min_length=2)
association_result = list(association_rules)
print(len(association_result))
print(association_result)

# extra
for item in association_rules:
    pair = item[0]
    items = [x for x in pair]
    print("rule: " + items[0] + "->" + items[1])
    print("Support: " + str(item[1]))
    print("Confidence: " + str(item[2][0][2]))
    print("Lift: " + str(item[2][0][3]))
    print("=====================================")