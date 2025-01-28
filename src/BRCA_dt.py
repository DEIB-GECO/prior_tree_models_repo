
from pktree import tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
import os
import json
import pandas as pd
from sklearn.metrics import f1_score, recall_score, precision_score
from utils.utils import load_data_brca, load_penalties, load_DT_classifier, scoring

# load dataset
dire_repo = '/home/mongardi/tree-based-models/prior_tree_models_repo'
file_path = os.path.join(dire_repo,"data/data_BRCA/BRCA_dataset.csv")
dire_results = os.path.join(dire_repo,'results/BRCA/features_dt/')

X_train, y_train,  X_test, y_test, gene_names = load_data_brca(file_path, test_size = 0.2, return_columns=True, random_state=42)

# load penalties
penalties_file =  os.path.join(dire_repo,"data/penalties.txt")
gis_score = load_penalties(penalties_file)

print(X_train.shape)

min_max_scaler = MinMaxScaler()
#standard_scaler = StandardScaler()
X_train = min_max_scaler.fit_transform(X_train)
X_test = min_max_scaler.transform(X_test)
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.fit_transform(y_test)
## NO GIS
accuracy = []
f1 = []
precision = []
recall = []

for i in range(10):
    classifier = tree.DecisionTreeClassifier(random_state=i, criterion="gini", w_prior=gis_score, pk_configuration="standard", pk_function=None, max_features='sqrt')
    classifier.fit(X_train, y_train)
    predictions = classifier.predict(X_test)

    accuracy.append(accuracy_score(y_test, predictions))
    f1.append(f1_score(y_test, predictions, average='macro'))
    precision.append(precision_score(y_test, predictions, average='macro'))
    recall.append(recall_score(y_test, predictions, average='macro'))
    used_genes = set([gene_names[i] for i in classifier.tree_.feature if i != -2])
    print(i,": ",used_genes)
   
    with open(os.path.join(dire_results, 'no_gis_sqrt.json'), 'a') as file:
        json.dump(list(used_genes), file)
        file.write('\n')
    df_feat_importances = pd.DataFrame(classifier.feature_importances_, index=gene_names)
    df_feat_importances.to_csv(os.path.join(dire_results, 'no_gis_sqrt_importances_'+ str(i) +'.csv'))
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
accuracy = []
f1 = []
precision = []
recall = []
print
for i in range(10):
    classifier = tree.DecisionTreeClassifier(random_state=i, criterion="gini", w_prior=gis_score, pk_configuration="standard", pk_function=None, max_features=1000)
    classifier.fit(X_train, y_train)
    predictions = classifier.predict(X_test)
    accuracy.append(accuracy_score(y_test, predictions))
    f1.append(f1_score(y_test, predictions, average='macro'))
    precision.append(precision_score(y_test, predictions, average='macro'))
    recall.append(recall_score(y_test, predictions, average='macro'))

    used_genes = set([gene_names[i] for i in classifier.tree_.feature if i != -2])
    print(i,": ",used_genes)
    
    with open(os.path.join(dire_results, 'no_gis_1000.json'), 'a') as file:
        json.dump(list(used_genes), file)
        file.write('\n')
    df_feat_importances = pd.DataFrame(classifier.feature_importances_, index=gene_names)
    df_feat_importances.to_csv(os.path.join(dire_results, 'no_gis_1000_importances_'+ str(i) +'.csv'))

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
accuracy = []
f1 = []
precision = []
recall = []

for i in range(10):
    classifier = tree.DecisionTreeClassifier(random_state=i, criterion="gini", w_prior=gis_score, pk_configuration="standard", pk_function=None, max_features=2000)
    classifier.fit(X_train, y_train)
    predictions = classifier.predict(X_test)
    accuracy.append(accuracy_score(y_test, predictions))
    f1.append(f1_score(y_test, predictions, average='macro'))
    precision.append(precision_score(y_test, predictions, average='macro'))
    recall.append(recall_score(y_test, predictions, average='macro'))

    used_genes = set([gene_names[i] for i in classifier.tree_.feature if i != -2])
    print(i,": ",used_genes)
    
    with open(os.path.join(dire_results, 'no_gis_2000.json'), 'a') as file:
        json.dump(list(used_genes), file)
        file.write('\n')
    df_feat_importances = pd.DataFrame(classifier.feature_importances_, index=gene_names)
    df_feat_importances.to_csv(os.path.join(dire_results, 'no_gis_2000_importances_'+ str(i) +'.csv'))    

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
accuracy = []
f1 = []
precision = []
recall = []

for i in range(10):
    classifier = tree.DecisionTreeClassifier(random_state=i, criterion="gini", w_prior=gis_score, pk_configuration="standard", pk_function=None, max_features=5000)
    classifier.fit(X_train, y_train)
    predictions = classifier.predict(X_test)
    accuracy.append(accuracy_score(y_test, predictions))
    f1.append(f1_score(y_test, predictions, average='macro'))
    precision.append(precision_score(y_test, predictions, average='macro'))
    recall.append(recall_score(y_test, predictions, average='macro'))

    used_genes = set([gene_names[i] for i in classifier.tree_.feature if i != -2])
    print(i,": ",used_genes)
    
    with open(os.path.join(dire_results, 'no_gis_5000.json'), 'a') as file: 
        json.dump(list(used_genes), file)
        file.write('\n')

    df_feat_importances = pd.DataFrame(classifier.feature_importances_, index=gene_names)
    df_feat_importances.to_csv(os.path.join(dire_results, 'no_gis_5000_importances_'+ str(i) +'.csv'))   

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
accuracy = []
f1 = []
precision = []
recall = []

for i in range(10):
    classifier = tree.DecisionTreeClassifier(random_state=i, criterion="gini", w_prior=gis_score, pk_configuration="standard", pk_function=None, max_features=10000)
    classifier.fit(X_train, y_train)
    predictions = classifier.predict(X_test)
    accuracy.append(accuracy_score(y_test, predictions))
    f1.append(f1_score(y_test, predictions, average='macro'))
    precision.append(precision_score(y_test, predictions, average='macro'))
    recall.append(recall_score(y_test, predictions, average='macro'))

    used_genes = set([gene_names[i] for i in classifier.tree_.feature if i != -2])
    print(i,": ",used_genes)
    
    with open(os.path.join(dire_results, 'no_gis_10000.json'), 'a') as file:
        json.dump(list(used_genes), file)
        file.write('\n')
    df_feat_importances = pd.DataFrame(classifier.feature_importances_, index=gene_names)
    df_feat_importances.to_csv(os.path.join(dire_results, 'no_gis_10000_importances_'+ str(i) +'.csv'))

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
accuracy = []
f1 = []
precision = []
recall = []

for i in range(10):
    classifier = tree.DecisionTreeClassifier(random_state=i, criterion="gini", w_prior=gis_score, pk_configuration="standard", pk_function=None, max_features=None)
    classifier.fit(X_train, y_train)
    predictions = classifier.predict(X_test)
    accuracy.append(accuracy_score(y_test, predictions))
    f1.append(f1_score(y_test, predictions, average='macro'))
    precision.append(precision_score(y_test, predictions, average='macro'))
    recall.append(recall_score(y_test, predictions, average='macro'))

    used_genes = set([gene_names[i] for i in classifier.tree_.feature if i != -2])
    print(i,": ",used_genes)
    
    with open(os.path.join(dire_results, 'no_gis_None.json'), 'a') as file:
        json.dump(list(used_genes), file)
        file.write('\n')
    df_feat_importances = pd.DataFrame(classifier.feature_importances_, index=gene_names)
    df_feat_importances.to_csv(os.path.join(dire_results, 'no_gis_None_importances_'+ str(i) +'.csv'))  

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
accuracy = []
f1 = []
precision = []
recall = []

for i in range(10):
    classifier = tree.DecisionTreeClassifier(random_state=i, criterion="gini", w_prior=gis_score, pk_configuration="on_impurity_improvement", pk_function=None, max_features='sqrt', v=0.35)
    classifier.fit(X_train, y_train)
    predictions = classifier.predict(X_test)
    accuracy.append(accuracy_score(y_test, predictions))
    f1.append(f1_score(y_test, predictions, average='macro'))
    precision.append(precision_score(y_test, predictions, average='macro'))
    recall.append(recall_score(y_test, predictions, average='macro'))

    used_genes = set([gene_names[i] for i in classifier.tree_.feature if i != -2])
    print(i,": ",used_genes)
    
    with open(os.path.join(dire_results, 'imp_sqrt.json'), 'a') as file:
        json.dump(list(used_genes), file)
        file.write('\n')
    df_feat_importances = pd.DataFrame(classifier.feature_importances_, index=gene_names)
    df_feat_importances.to_csv(os.path.join(dire_results, 'imp_sqrt_importances_'+ str(i) +'.csv'))
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
accuracy = []
f1 = []
precision = []
recall = []

for i in range(10):
    classifier = tree.DecisionTreeClassifier(random_state=i, criterion="gini", w_prior=gis_score, pk_configuration="on_impurity_improvement", pk_function=None, max_features=1000, v=0.35)
    classifier.fit(X_train, y_train)
    predictions = classifier.predict(X_test)
    accuracy.append(accuracy_score(y_test, predictions))
    f1.append(f1_score(y_test, predictions, average='macro'))
    precision.append(precision_score(y_test, predictions, average='macro'))
    recall.append(recall_score(y_test, predictions, average='macro'))

    used_genes = set([gene_names[i] for i in classifier.tree_.feature if i != -2])
    print(i,": ",used_genes)
    
    with open(os.path.join(dire_results, 'imp_1000.json'), 'a') as file:
        json.dump(list(used_genes), file)
        file.write('\n')
    df_feat_importances = pd.DataFrame(classifier.feature_importances_, index=gene_names)
    df_feat_importances.to_csv(os.path.join(dire_results, 'imp_1000_importances_'+ str(i) +'.csv'))
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
accuracy = []
f1 = []
precision = []
recall = []

for i in range(10):
    classifier = tree.DecisionTreeClassifier(random_state=i, criterion="gini", w_prior=gis_score, pk_configuration="on_impurity_improvement", pk_function=None, max_features=2000, v=0.35)
    classifier.fit(X_train, y_train)
    predictions = classifier.predict(X_test)
    accuracy.append(accuracy_score(y_test, predictions))
    f1.append(f1_score(y_test, predictions, average='macro'))
    precision.append(precision_score(y_test, predictions, average='macro'))
    recall.append(recall_score(y_test, predictions, average='macro'))

    used_genes = set([gene_names[i] for i in classifier.tree_.feature if i != -2])
    print(i,": ",used_genes)
    
    with open(os.path.join(dire_results, 'imp_2000.json'), 'a') as file:
        json.dump(list(used_genes), file)
        file.write('\n')
    df_feat_importances = pd.DataFrame(classifier.feature_importances_, index=gene_names)
    df_feat_importances.to_csv(os.path.join(dire_results, 'imp_2000_importances_'+ str(i) +'.csv'))
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
accuracy = []
f1 = []
precision = []
recall = []

for i in range(10):
    classifier = tree.DecisionTreeClassifier(random_state=i, criterion="gini", w_prior=gis_score, pk_configuration="on_impurity_improvement", pk_function=None, max_features=5000, v=0.35)
    classifier.fit(X_train, y_train)
    predictions = classifier.predict(X_test)
    accuracy.append(accuracy_score(y_test, predictions))
    f1.append(f1_score(y_test, predictions, average='macro'))
    precision.append(precision_score(y_test, predictions, average='macro'))
    recall.append(recall_score(y_test, predictions, average='macro'))

    used_genes = set([gene_names[i] for i in classifier.tree_.feature if i != -2])
    print(i,": ",used_genes)
    
    with open(os.path.join(dire_results, 'imp_5000.json'), 'a') as file:
        json.dump(list(used_genes), file)
        file.write('\n')
    df_feat_importances = pd.DataFrame(classifier.feature_importances_, index=gene_names)
    df_feat_importances.to_csv(os.path.join(dire_results, 'imp_5000_importances_'+ str(i) +'.csv'))
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
accuracy = []
f1 = []
precision = []
recall = []

for i in range(10):
    classifier = tree.DecisionTreeClassifier(random_state=i, criterion="gini", w_prior=gis_score, pk_configuration="on_impurity_improvement", pk_function=None, max_features=10000, v=0.35)
    classifier.fit(X_train, y_train)
    predictions = classifier.predict(X_test)
    accuracy.append(accuracy_score(y_test, predictions))
    f1.append(f1_score(y_test, predictions, average='macro'))
    precision.append(precision_score(y_test, predictions, average='macro'))
    recall.append(recall_score(y_test, predictions, average='macro'))

    used_genes = set([gene_names[i] for i in classifier.tree_.feature if i != -2])
    print(i,": ",used_genes)
    with open(os.path.join(dire_results, 'imp_10000.json'), 'a') as file:
        json.dump(list(used_genes), file)
        file.write('\n')
    df_feat_importances = pd.DataFrame(classifier.feature_importances_, index=gene_names)
    df_feat_importances.to_csv(os.path.join(dire_results, 'imp_10000_importances_'+ str(i) +'.csv'))
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
accuracy = []
f1 = []
precision = []
recall = []

for i in range(10):
    classifier = tree.DecisionTreeClassifier(random_state=i, criterion="gini", w_prior=gis_score, pk_configuration="on_impurity_improvement", pk_function=None, v=0.35)
    classifier.fit(X_train, y_train)
    predictions = classifier.predict(X_test)
    accuracy.append(accuracy_score(y_test, predictions))
    f1.append(f1_score(y_test, predictions, average='macro'))
    precision.append(precision_score(y_test, predictions, average='macro'))
    recall.append(recall_score(y_test, predictions, average='macro'))

    used_genes = set([gene_names[i] for i in classifier.tree_.feature if i != -2])
    print(i,": ",used_genes)

    with open(os.path.join(dire_results, 'imp_None.json'), 'a') as file:
        json.dump(list(used_genes), file)
        file.write('\n')
    df_feat_importances = pd.DataFrame(classifier.feature_importances_, index=gene_names)
    df_feat_importances.to_csv(os.path.join(dire_results, 'imp_None_importances_'+ str(i) +'.csv'))
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
# ON FEATURE SELECTION
accuracy = []
f1 = []
precision = []
recall = []

for i in range(10):
    classifier = tree.DecisionTreeClassifier(random_state=i, criterion="gini", w_prior=gis_score, pk_configuration="on_feature_sampling", pk_function=None, k=2.0, max_features='sqrt')
    classifier.fit(X_train, y_train)
    predictions = classifier.predict(X_test)
    accuracy.append(accuracy_score(y_test, predictions))
    f1.append(f1_score(y_test, predictions, average='macro'))
    precision.append(precision_score(y_test, predictions, average='macro'))
    recall.append(recall_score(y_test, predictions, average='macro'))

    used_genes = set([gene_names[i] for i in classifier.tree_.feature if i != -2])
    print(i,": ",used_genes)

    with open(os.path.join(dire_results, 'fs_sqrt.json'), 'a') as file:
        json.dump(list(used_genes), file)
        file.write('\n')
    df_feat_importances = pd.DataFrame(classifier.feature_importances_, index=gene_names)
    df_feat_importances.to_csv(os.path.join(dire_results, 'fs_sqrt_importances_'+ str(i) +'.csv'))
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
accuracy = []
f1 = []
precision = []
recall = []

for i in range(10):
    classifier = tree.DecisionTreeClassifier(random_state=i, criterion="gini", w_prior=gis_score, pk_configuration="on_feature_sampling", pk_function=None, k=2.0, max_features=1000)
    classifier.fit(X_train, y_train)
    predictions = classifier.predict(X_test)
    accuracy.append(accuracy_score(y_test, predictions))
    f1.append(f1_score(y_test, predictions, average='macro'))
    precision.append(precision_score(y_test, predictions, average='macro'))
    recall.append(recall_score(y_test, predictions, average='macro'))

    used_genes = set([gene_names[i] for i in classifier.tree_.feature if i != -2])
    print(i,": ",used_genes)

    with open(os.path.join(dire_results, 'fs_1000.json'), 'a') as file:
        json.dump(list(used_genes), file)
        file.write('\n')
    df_feat_importances = pd.DataFrame(classifier.feature_importances_, index=gene_names)
    df_feat_importances.to_csv(os.path.join(dire_results, 'fs_1000_importances_'+ str(i) +'.csv'))     
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
accuracy = []
f1 = []
precision = []
recall = []

for i in range(10):
    classifier = tree.DecisionTreeClassifier(random_state=i, criterion="gini", w_prior=gis_score, pk_configuration="on_feature_sampling", pk_function=None, k=2.0, max_features=2000)
    classifier.fit(X_train, y_train)
    predictions = classifier.predict(X_test)
    accuracy.append(accuracy_score(y_test, predictions))
    f1.append(f1_score(y_test, predictions, average='macro'))
    precision.append(precision_score(y_test, predictions, average='macro'))
    recall.append(recall_score(y_test, predictions, average='macro'))

    used_genes = set([gene_names[i] for i in classifier.tree_.feature if i != -2])
    print(i,": ",used_genes)

    with open(os.path.join(dire_results, 'fs_2000.json'), 'a') as file:
        json.dump(list(used_genes), file)
        file.write('\n')
    df_feat_importances = pd.DataFrame(classifier.feature_importances_, index=gene_names)
    df_feat_importances.to_csv(os.path.join(dire_results, 'fs_2000_importances_'+ str(i) +'.csv'))
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
accuracy = []
f1 = []
precision = []
recall = []

for i in range(10):
    classifier = tree.DecisionTreeClassifier(random_state=i, criterion="gini", w_prior=gis_score, pk_configuration="on_feature_sampling", pk_function=None, k=2.0, max_features=5000)
    classifier.fit(X_train, y_train)
    predictions = classifier.predict(X_test)
    accuracy.append(accuracy_score(y_test, predictions))
    f1.append(f1_score(y_test, predictions, average='macro'))
    precision.append(precision_score(y_test, predictions, average='macro'))
    recall.append(recall_score(y_test, predictions, average='macro'))

    used_genes = set([gene_names[i] for i in classifier.tree_.feature if i != -2])
    print(i,": ",used_genes)

    with open(os.path.join(dire_results, 'fs_5000.json'), 'a') as file:
        json.dump(list(used_genes), file)
        file.write('\n')
    df_feat_importances = pd.DataFrame(classifier.feature_importances_, index=gene_names)
    df_feat_importances.to_csv(os.path.join(dire_results, 'fs_5000_importances_'+ str(i) +'.csv'))
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
accuracy = []
f1 = []
precision = []
recall = []

for i in range(10):
    classifier = tree.DecisionTreeClassifier(random_state=i, criterion="gini", w_prior=gis_score, pk_configuration="on_feature_sampling", pk_function=None, k=2.0, max_features=10000)
    classifier.fit(X_train, y_train)
    predictions = classifier.predict(X_test)
    accuracy.append(accuracy_score(y_test, predictions))
    f1.append(f1_score(y_test, predictions, average='macro'))
    precision.append(precision_score(y_test, predictions, average='macro'))
    recall.append(recall_score(y_test, predictions, average='macro'))

    used_genes = set([gene_names[i] for i in classifier.tree_.feature if i != -2])
    print(i,": ",used_genes)

    with open(os.path.join(dire_results, 'fs_10000.json'), 'a') as file:
        json.dump(list(used_genes), file)
        file.write('\n')
    df_feat_importances = pd.DataFrame(classifier.feature_importances_, index=gene_names)
    df_feat_importances.to_csv(os.path.join(dire_results, 'fs_10000_importances_'+ str(i) +'.csv'))
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
accuracy = []
f1 = []
precision = []
recall = []

for i in range(10):
    classifier = tree.DecisionTreeClassifier(random_state=i, criterion="gini", w_prior=gis_score, pk_configuration="on_feature_sampling", pk_function=None, k=2.0)
    classifier.fit(X_train, y_train)
    predictions = classifier.predict(X_test)
    accuracy.append(accuracy_score(y_test, predictions))
    f1.append(f1_score(y_test, predictions, average='macro'))
    precision.append(precision_score(y_test, predictions, average='macro'))
    recall.append(recall_score(y_test, predictions, average='macro'))

    used_genes = set([gene_names[i] for i in classifier.tree_.feature if i != -2])
    print(i,": ",used_genes)

    with open(os.path.join(dire_results, 'fs_None.json'), 'a') as file:
        json.dump(list(used_genes), file)
        file.write('\n')
    df_feat_importances = pd.DataFrame(classifier.feature_importances_, index=gene_names)
    df_feat_importances.to_csv(os.path.join(dire_results, 'fs_None_importances_'+ str(i) +'.csv'))   
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
# ALL
accuracy = []
f1 = []
precision = []
recall = []

for i in range(10):
    classifier = tree.DecisionTreeClassifier(random_state=i, criterion="gini", w_prior=gis_score, pk_configuration="all", pk_function=None, k=2.0, v=0.35, max_features='sqrt')
    classifier.fit(X_train, y_train)
    predictions = classifier.predict(X_test)
    accuracy.append(accuracy_score(y_test, predictions))
    f1.append(f1_score(y_test, predictions, average='macro'))
    precision.append(precision_score(y_test, predictions, average='macro'))
    recall.append(recall_score(y_test, predictions, average='macro'))

    used_genes = set([gene_names[i] for i in classifier.tree_.feature if i != -2])
    print(i,": ",used_genes)

    with open(os.path.join(dire_results, 'all_sqrt.json'), 'a') as file:
        json.dump(list(used_genes), file)
        file.write('\n')
    df_feat_importances = pd.DataFrame(classifier.feature_importances_, index=gene_names)
    df_feat_importances.to_csv(os.path.join(dire_results, 'all_sqrt_importances_'+ str(i) +'.csv')) 

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
accuracy = []
f1 = []
precision = []
recall = []

for i in range(10):
    classifier = tree.DecisionTreeClassifier(random_state=i, criterion="gini", w_prior=gis_score, pk_configuration="all", pk_function=None, k=2.0, v=0.35, max_features=1000)
    classifier.fit(X_train, y_train)
    predictions = classifier.predict(X_test)
    accuracy.append(accuracy_score(y_test, predictions))
    f1.append(f1_score(y_test, predictions, average='macro'))
    precision.append(precision_score(y_test, predictions, average='macro'))
    recall.append(recall_score(y_test, predictions, average='macro'))

    used_genes = set([gene_names[i] for i in classifier.tree_.feature if i != -2])
    print(i,": ",used_genes)

    with open(os.path.join(dire_results, 'all_1000.json'), 'a') as file:
        json.dump(list(used_genes), file)
        file.write('\n')
    df_feat_importances = pd.DataFrame(classifier.feature_importances_, index=gene_names)
    df_feat_importances.to_csv(os.path.join(dire_results, 'all_1000_importances_'+ str(i) +'.csv'))
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
accuracy = []
f1 = []
precision = []
recall = []

for i in range(10):
    classifier = tree.DecisionTreeClassifier(random_state=i, criterion="gini", w_prior=gis_score, pk_configuration="all", pk_function=None, k=2.0, v=0.35, max_features=2000)
    classifier.fit(X_train, y_train)
    predictions = classifier.predict(X_test)
    accuracy.append(accuracy_score(y_test, predictions))
    f1.append(f1_score(y_test, predictions, average='macro'))
    precision.append(precision_score(y_test, predictions, average='macro'))
    recall.append(recall_score(y_test, predictions, average='macro'))

    used_genes = set([gene_names[i] for i in classifier.tree_.feature if i != -2])
    print(i,": ",used_genes)
    with open(os.path.join(dire_results, 'all_2000.json'), 'a') as file:
        json.dump(list(used_genes), file)
        file.write('\n')
    df_feat_importances = pd.DataFrame(classifier.feature_importances_, index=gene_names)
    df_feat_importances.to_csv(os.path.join(dire_results, 'all_2000_importances_'+ str(i) +'.csv'))
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
accuracy = []
f1 = []
precision = []
recall = []

for i in range(10):
    classifier = tree.DecisionTreeClassifier(random_state=i, criterion="gini", w_prior=gis_score, pk_configuration="all", pk_function=None, k=2.0, v=0.35, max_features=5000)
    classifier.fit(X_train, y_train)
    predictions = classifier.predict(X_test)
    accuracy.append(accuracy_score(y_test, predictions))
    f1.append(f1_score(y_test, predictions, average='macro'))
    precision.append(precision_score(y_test, predictions, average='macro'))
    recall.append(recall_score(y_test, predictions, average='macro'))

    used_genes = set([gene_names[i] for i in classifier.tree_.feature if i != -2])
    print(i,": ",used_genes)

    with open(os.path.join(dire_results, 'all_5000.json'), 'a') as file:
        json.dump(list(used_genes), file)
        file.write('\n')
    df_feat_importances = pd.DataFrame(classifier.feature_importances_, index=gene_names)
    df_feat_importances.to_csv(os.path.join(dire_results, 'all_5000_importances_'+ str(i) +'.csv'))  
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
accuracy = []
f1 = []
precision = []
recall = []

for i in range(10):
    classifier = tree.DecisionTreeClassifier(random_state=i, criterion="gini", w_prior=gis_score, pk_configuration="all", pk_function=None, k=2.0, v=0.35, max_features=10000)
    classifier.fit(X_train, y_train)
    predictions = classifier.predict(X_test)
    accuracy.append(accuracy_score(y_test, predictions))
    f1.append(f1_score(y_test, predictions, average='macro'))
    precision.append(precision_score(y_test, predictions, average='macro'))
    recall.append(recall_score(y_test, predictions, average='macro'))

    used_genes = set([gene_names[i] for i in classifier.tree_.feature if i != -2])
    print(i,": ",used_genes)

    with open(os.path.join(dire_results, 'all_10000.json'), 'a') as file:
        json.dump(list(used_genes), file)
        file.write('\n')
    df_feat_importances = pd.DataFrame(classifier.feature_importances_, index=gene_names)
    df_feat_importances.to_csv(os.path.join(dire_results, 'all_10000_importances_'+ str(i) +'.csv'))  
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
accuracy = []
f1 = []
precision = []
recall = []

for i in range(10):
    classifier = tree.DecisionTreeClassifier(random_state=i, criterion="gini", w_prior=gis_score, pk_configuration="all", pk_function=None, k=2.0, v=0.35)
    classifier.fit(X_train, y_train)
    predictions = classifier.predict(X_test)
    accuracy.append(accuracy_score(y_test, predictions))
    f1.append(f1_score(y_test, predictions, average='macro'))
    precision.append(precision_score(y_test, predictions, average='macro'))
    recall.append(recall_score(y_test, predictions, average='macro'))

    used_genes = set([gene_names[i] for i in classifier.tree_.feature if i != -2])
    print(i,": ",used_genes)
    
    with open(os.path.join(dire_results, 'all_None.json'), 'a') as file:
        json.dump(list(used_genes), file)
        file.write('\n')
    df_feat_importances = pd.DataFrame(classifier.feature_importances_, index=gene_names)
    df_feat_importances.to_csv(os.path.join(dire_results, 'all_None_importances_'+ str(i) +'.csv')) 
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