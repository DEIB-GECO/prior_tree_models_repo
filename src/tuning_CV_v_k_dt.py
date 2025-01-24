import numpy as np
import pandas as pd
import sys
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn import datasets, model_selection, metrics
from sklearn.metrics import make_scorer
from sklearn.metrics import recall_score, accuracy_score, precision_score, f1_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from utils import load_data_brca, load_penalties, load_DT_classifier, scoring
from pktree import tree
from joblib import Parallel, delayed
from joblib import Memory

# load dataset
file_path = "/data_tree/data_BRCA/BRCA_dataset.csv"
X_train, y_train,  X_test, y_test = load_data_brca(file_path, test_size = 0.2, random_state=42)

# load penalties
penalties_file = "/home/mongardi/tree-based-models/Biology-informed_sklearn/data_tree/penalties.txt"
gis_score = load_penalties(penalties_file)

def run_cv(seed, pk_configuration='on_feature_selection', w_prior=None, 
            values_k= None, values_v=None, criterion='gini'):
    
    print(f"Running iteration {seed}", flush=True)
    accuracy = []
    precision = []
    recall = []
    f1 = []

    if values_k is None:
        param = 'v'
        values_k = np.ones(len(values_v))
    
    elif values_v is None:
        param = 'k'
        values_v = np.ones(len(values_k))

    for value_k, value_v in zip(values_k, values_v):
        print(f"Training with k={value_k} and v={value_v}")
    

        tree_clf = load_DT_classifier(pk_configuration= pk_configuration,  w_prior = w_prior, k=value_k,
                    v=value_v, criterion=criterion, random_state=seed)
        
        all_scores = model_selection.cross_validate(tree_clf, X_train, y_train, cv=5, scoring=scoring)

        accuracy.append(all_scores['test_accuracy'].mean())
        recall.append(all_scores['test_rec'].mean())
        precision.append(all_scores['test_prec'].mean())
        f1.append(all_scores['test_f1'].mean())

        print(f"k={value_k}, v={value_v}: {all_scores['test_f1'].mean()}")

    return accuracy, recall, precision, f1



#tuning v

v =np.arange(0.25, 1.05, 0.05)
v = [np.round(i, 2) for i in v]


refit = False

accuracy_all = {str(key):[] for key in k}
recall_all = {str(key):[] for key in k}
precision_all = {str(key):[] for key in k}
f1_all = {str(key):[] for key in k}
accuracy_all_test = [] 
recall_all_test = []
precision_all_test = []
f1_all_test = []

# parallel implementation
print('-------------------Training-------------------')
n_jobs = 10

parallel = Parallel(
    n_jobs=n_jobs)

work = parallel(
    delayed(run_cv)(i, pk_configuration='on_impurity_improvment', w_prior=gis_score, 
            values_k=None, values_v=v, criterion='gini')
    for i in range(n_jobs)
)

print('-------------------Validation Results-------------------')

accuracy_all = {str(key):[] for key in k}
recall_all = {str(key):[] for key in k}
precision_all = {str(key):[] for key in k}
f1_all = {str(key):[] for key in k}

for accuracy, recall, precision, f1 in work:
    
    for m, value in enumerate(k):
        accuracy_all[str(value)].append(accuracy[m])
        recall_all[str(value)].append(recall[m])
        precision_all[str(value)].append(precision[m])
        f1_all[str(value)].append(f1[m])

accuracy_all = {key:np.array(value).mean() for key, value in accuracy_all.items()}
recall_all = {key:np.array(value).mean() for key, value in recall_all.items()}
precision_all = {key:np.array(value).mean() for key, value in precision_all.items()}
f1_all = {key:np.array(value).mean() for key, value in f1_all.items()}

print(f1_all)
f1_all_idx = np.argmax(list(f1_all.values()))
best_param_v = list(f1_all.keys())[f1_all_idx]   
print(f"Best v={best_param_v}: {np.max(list(f1_all.values()))}")

if refit:
    for i in range(10):
        tree_clf = Pipeline(steps=[('Scaler', MinMaxScaler()),
                    ('Model', tree.DecisionTreeClassifier( w_prior = gis_score,
                    pk_configuration="on_feature_selection", v=best_param_v, max_features=None,
                    criterion="gini", random_state=i))])
        tree_clf.fit(X_train, y_train)
        y_pred = tree_clf.predict(X_test)
        accuracy_all_test.append(accuracy_score(y_test, y_pred))
        recall_all_test.append(recall_score(y_test, y_pred, average= 'macro'))
        precision_all_test.append(precision_score(y_test, y_pred, average= 'macro')) 
        f1_all_test.append(f1_score(y_test, y_pred, average="macro"))

    print('-------------------Results-------------------')  
   
    print("Accuracy:", np.array(accuracy_all_test).mean())
    print("Recall:", np.array(recall_all_test).mean())
    print("Precision:", np.array(precision_all_test).mean())
    print("F1:", np.array(f1_all_test).mean())




# tuning k
k = np.arange(1.0, 5.25, 0.25)
k = [np.round(i, 2) for i in k]
accuracy_all = {str(key):[] for key in k}
recall_all = {str(key):[] for key in k}
precision_all = {str(key):[] for key in k}
f1_all = {str(key):[] for key in k}
accuracy_all_test = [] 
recall_all_test = []
precision_all_test = []
f1_all_test = []


# parallel implementation
print('-------------------Training-------------------')
n_jobs = 10

parallel = Parallel(
    n_jobs=n_jobs)

work = parallel(
    delayed(run_cv)(i, pk_configuration='on_feature_selection', w_prior=gis_score, 
            values_k=k, values_v=None, criterion='gini')
    for i in range(n_jobs)
)

print('-------------------Validation Results-------------------')

accuracy_all = {str(key):[] for key in k}
recall_all = {str(key):[] for key in k}
precision_all = {str(key):[] for key in k}
f1_all = {str(key):[] for key in k}

for accuracy, recall, precision, f1 in work:
    
    for m, value in enumerate(k):
        accuracy_all[str(value)].append(accuracy[m])
        recall_all[str(value)].append(recall[m])
        precision_all[str(value)].append(precision[m])
        f1_all[str(value)].append(f1[m])

accuracy_all = {key:np.array(value).mean() for key, value in accuracy_all.items()}
recall_all = {key:np.array(value).mean() for key, value in recall_all.items()}
precision_all = {key:np.array(value).mean() for key, value in precision_all.items()}
f1_all = {key:np.array(value).mean() for key, value in f1_all.items()}

print(f1_all)
f1_all_idx = np.argmax(list(f1_all.values()))
best_param_k = list(f1_all.keys())[f1_all_idx]   
print(f"Best v={best_param_v}: {np.max(list(f1_all.values()))}")

if refit:
    for i in range(10):
        tree_clf = Pipeline(steps=[('Scaler', MinMaxScaler()),
                    ('Model', tree.DecisionTreeClassifier( w_prior = gis_score,
                    pk_configuration="on_feature_selection", k=best_param_k, max_features=None,
                    criterion="gini", random_state=i))])
        tree_clf.fit(X_train, y_train)
        y_pred = tree_clf.predict(X_test)
        accuracy_all_test.append(accuracy_score(y_test, y_pred))
        recall_all_test.append(recall_score(y_test, y_pred, average= 'macro'))
        precision_all_test.append(precision_score(y_test, y_pred, average= 'macro')) 
        f1_all_test.append(f1_score(y_test, y_pred, average="macro"))

    print('-------------------Results-------------------')  
   
    print("Accuracy:", np.array(accuracy_all_test).mean())
    print("Recall:", np.array(recall_all_test).mean())
    print("Precision:", np.array(precision_all_test).mean())
    print("F1:", np.array(f1_all_test).mean())


