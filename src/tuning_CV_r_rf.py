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
from utils.utils import load_data_brca, load_penalties, load_DT_classifier, scoring
from pktree import ensemble

# load dataset
dire_repo = '/home/mongardi/tree-based-models/prior_tree_models_repo'
file_path = os.path.join(dire_repo,"data/data_BRCA/BRCA_dataset.csv")
dire_results = os.path.join(dire_repo,'results/BRCA/features_dt/')
X_train, y_train,  X_test, y_test = load_data_brca(file_path, test_size = 0.2, random_state=42)

#load penalties
gis_score = load_penalties(penalties_file)

print(X_train.shape)
# tuning v

# v =np.arange(0.25, 1.05, 0.05)
# v = [np.round(i, 2) for i in v]
# #v = [0.25, 0.50]
scoring = {'accuracy':'accuracy',
            'rec': make_scorer(recall_score, average= 'macro'),
            'prec': make_scorer(precision_score, average= 'macro'), 
            'f1':make_scorer(f1_score, average="macro")}

refit = False

which_gis = "all"
k = 2.0
v = 0.35

oob_score = True
on_oob = True
mfv = "sqrt"

def run_cv(seed, r):
    
    print(f"Running iteration {seed}", flush=True)
    accuracy = []
    precision = []
    recall = []
    f1 = []
    for value in r:
        print(f"Training with r={value}")


        tree_clf = Pipeline(steps=[('Scaler', MinMaxScaler()),
                    ('Model', ensemble.RandomForestClassifier(n_estimators=100, random_state=seed, criterion="gini",
                    w_prior=gis_score, pk_configuration=which_gis, max_features = mfv, k=k, v=v, r=value, 
                    oob_score = oob_score, on_oob = on_oob))])

        all_scores = model_selection.cross_validate(tree_clf, X_train, y_train, cv=5, scoring=scoring)

        accuracy.append(all_scores['test_accuracy'].mean())
        recall.append(all_scores['test_rec'].mean())
        precision.append(all_scores['test_prec'].mean())
        f1.append(all_scores['test_f1'].mean())

        print(f"r={value}: {all_scores['test_f1'].mean()}")
    return accuracy, recall, precision, f1


r = np.arange(1.0, 10.50, 0.50)
r = [np.round(i, 2) for i in r]

accuracy_all = {str(key):[] for key in r}
recall_all = {str(key):[] for key in r}
precision_all = {str(key):[] for key in r}
f1_all = {str(key):[] for key in r}
accuracy_all_test = [] 
recall_all_test = []
precision_all_test = []
f1_all_test = []


# parallel implementation
from joblib import Parallel, delayed
from joblib import Memory
print('-------------------Training-------------------')
n_jobs = 10

parallel = Parallel(
    n_jobs=n_jobs)
work = parallel(
    delayed(run_cv)(i, r)
    for i in range(n_jobs)
)

for accuracy, recall, precision, f1 in work:
    
    for m, value in enumerate(r):
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
print(f"Best r={best_param_v}: {np.max(list(f1_all.values()))}")

if refit:
    for seed in range(10):
        tree_clf = Pipeline(steps=[('Scaler', MinMaxScaler()),
                    ('Model', RandomForestClassifier(n_estimators=100, random_state=seed, criterion="gini",
                    w_prior=gis_score, pk_configuration=which_gis, max_features = mfv, k=k, v=v, r=best_param_v, 
                    oob_score = oob_score, on_oob = on_oob, pk_function=None))])

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



