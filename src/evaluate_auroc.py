from sklearn import metrics
from datasets import Dataset

# test = Dataset.from_json('/home/tkdrnjs0621/work/kmel-reasoning2/result/bace_test.jsonl')
# y_true = [1 if ref == 'Yes.' else 0 for ref in test["label"]] 
# y_pred =  [1 if ref == 'Yes.' else 0 for ref in test["prediction"]] 

# auroc = metrics.roc_auc_score(y_true, y_pred)
# print(auroc)

test = Dataset.from_json('/home/tkdrnjs0621/work/kmel-reasoning3/result/hiv_reasoning.jsonl')
y_true = [1 if ref == 'Yes.' else 0 for ref in test["result"]] 
y_pred =  [1 if 'yes' in ref.lower().split('answer:')[-1] else 0 for ref in test["prediction"]] 

auroc = metrics.roc_auc_score(y_true, y_pred)
print(auroc)
