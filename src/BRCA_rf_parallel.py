from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import f1_score, recall_score, precision_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
import pandas as pd
import json
from joblib import Parallel, delayed
from joblib import Memory
from utils import load_data_brca, load_penalties, load_DT_classifier, scoring
# parallel implementation
from joblib import Parallel, delayed
from joblib import Memory


# load dataset
file_path = "/home/mongardi/tree-based-models/Biology-informed_sklearn/data_tree/data_BRCA/BRCA_dataset.csv"
X_train, y_train,  X_test, y_test, gene_names= load_data_brca(file_path, test_size = 0.2, return_columns=True, random_state=42)

# load penalties
penalties_file = "/home/mongardi/tree-based-models/Biology-informed_sklearn/data_tree/penalties.txt"
gis_score = load_penalties(penalties_file)

print(X_train.shape)

min_max_scaler = MinMaxScaler()
X_train = min_max_scaler.fit_transform(X_train)
X_test = min_max_scaler.transform(X_test)

label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.fit_transform(y_test)

"
which_gis = "on_impurity_improvement"
oob_score = True
on_oob = True



max_features_values = ['sqrt', 1000, 2000, 5000, 10000, None]
print("--------------------------------------------------------------")
print(f"{which_gis}:")

df_p = pd.DataFrame(np.zeros((len(max_features_values), 8)), columns=['accuracy mean', 'accuracy std', 'f1 mean', 'f1 std', 'precision mean', 'precision std', 'recall mean', 'recall std'], index=max_features_values)

def run_exp_mfv(mfv):
    
    accuracy = []
    f1 = []
    precision = []
    recall = []

    max_samples = None

    if which_gis == "all":
        k = 2.0
        v = 0.35 
        r=1.0

    elif which_gis == "on_feature_selection":
        k = 2.0
        v = 1.0
        r=1.0
    
    elif which_gis == "no_gis":
        k = 1.0
        v = 1.0
        r=1.0

    elif which_gis == "on_impurity_improvement":
        k = 1.0
        v = 0.35
        r=1.0

    if oob_score:
        r = 10.0

    print('params:', k, v, r, max_samples)
    print("max_features = ", mfv)
    for i in range(10):
        classifier = ensemble.RandomForestClassifier(n_estimators=100, random_state=i, criterion="gini", w_prior=gis_score, pk_configuration=which_gis, max_features = mfv, k=k, v=v, r=r, oob_score = oob_score, on_oob = on_oob, max_samples=max_samples)
        classifier.fit(X_train, y_train)
        predictions = classifier.predict(X_test)
        accuracy.append(accuracy_score(y_test, predictions))
        f1.append(f1_score(y_test, predictions, average='macro'))
        precision.append(precision_score(y_test, predictions, average='macro'))
        recall.append(recall_score(y_test, predictions, average='macro'))

        for k, tree in enumerate(classifier.estimators_):
            used_genes = set([gene_names[j] for j in tree.tree_.feature if j != -2])
            with open(f"/home/mongardi/tree-based-models/Biology-informed_sklearn/results/BRCA/features_rf/{which_gis}_rf_{mfv}_{on_oob}_{i}.json", 'a') as f:
                json.dump(list(used_genes), f)
                f.write('\n')
            df_feat_importances = pd.DataFrame(classifier.feature_importances_, index=gene_names)
            df_feat_importances.to_csv(f"/home/mongardi/tree-based-models/Biology-informed_sklearn/results/BRCA/features_rf/{which_gis}_rf_{mfv}_{on_oob}_{i}_feat_importances.csv")
    print("MEAN:")
    print("accuracy: ", np.mean(accuracy))
    print("f1: ", np.mean(f1))
    print("precision: ", np.mean(precision))
    print("recall: ", np.mean(recall))
    print("")
    print("STD DEVIATION:")
    print("accuracy: ", np.std(accuracy))
    print("f1: ", np.std(f1))
    print("precision: ", np.std(precision))
    print("recall: ", np.std(recall))
    return [np.mean(accuracy), np.mean(f1), np.mean(precision), np.mean(recall), np.std(accuracy), np.std(f1), np.std(precision), np.std(recall)]
    
    
print('-------------------Training-------------------')
n_jobs = len(max_features_values)

parallel = Parallel(
    n_jobs=n_jobs)
work = parallel(
    delayed(run_exp_mfv)(mfv)
    for mfv in max_features_values)

df_p = pd.DataFrame(np.zeros((len(max_features_values), 8)), columns=['accuracy mean', 'accuracy std', 'f1 mean', 'f1 std', 'precision mean', 'precision std', 'recall mean', 'recall std'], index=max_features_values)
for i,values in enumerate(work):
                
    df_p.loc[max_features_values[i], 'accuracy mean'], df_p.loc[max_features_values[i], 'accuracy std'] = np.round(values[0], 4), np.round(values[4], 4)
    df_p.loc[max_features_values[i], 'f1 mean'], df_p.loc[max_features_values[i], 'f1 std'] = np.round(values[1], 4), np.round(values[5], 4)
    df_p.loc[max_features_values[i], 'precision mean'], df_p.loc[max_features_values[i], 'precision std'] = np.round(values[2], 4), np.round(values[6], 4)
    df_p.loc[max_features_values[i], 'recall mean'], df_p.loc[max_features_values[i], 'recall std'] = np.round(values[3], 4), np.round(values[7], 4)

df_p.to_csv(f"/home/mongardi/tree-based-models/Biology-informed_sklearn/results/BRCA/results_rf/{which_gis}_rf_{on_oob}.csv")

