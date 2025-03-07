import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import  MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer
from sklearn.utils import shuffle
from sklearn.metrics import recall_score, precision_score
from pktree import tree
import random
import json
import sys

dire_repo='/home/mongardi/tree-based-models/prior_tree_models_repo'
sys.path.append(os.path.join(dire_repo, "src"))
from utils.utils import load_txt, fisher_scores

penalties = np.loadtxt(os.path.join(dire_repo,'data/penalties_kidney.txt'))
genes_set = load_txt(os.path.join(dire_repo,'data/data_kidney/controlled_features.txt'))

dire_kidney = os.path.join(dire_repo,'data/data_kidney/Kidney_df_tr_coding_new.csv')

df = pd.read_csv(dire_kidney)
df = shuffle(df, random_state=42)
df.set_index(df.columns[0], inplace=True)

X = df.drop(['is_healthy'], axis=1)
y = df['is_healthy']

X_numeric = X.select_dtypes(include='number')
rpm = X_numeric.div(X_numeric.sum(axis=1).values, axis=0) * 1e6
rpm_log = np.log2(rpm + 1)

most_relevant_genes = ['PPAPDC1A','ABCA4','VEGFB','ANKRD13D','ZNF561']
genes_set = [x for x in genes_set if x not in most_relevant_genes]

relevant_features = [list(X.columns).index(gene) for gene in most_relevant_genes if gene in X.columns]
print(relevant_features)
rpm_log_new = pd.concat([rpm_log[genes_set],rpm_log[most_relevant_genes]],axis=1)

# training and test split
X_train, X_test,  y_train, y_test = train_test_split(rpm_log, y, random_state = 42,
                                        test_size = 0.2, stratify=y)

fi_scores = fisher_scores(X_train, y_train.to_numpy())
X_train, X_test,  y_train, y_test = train_test_split(rpm_log_new, y, random_state = 42,
                                        test_size = 0.2, stratify=y)

genes = list(fi_scores.iloc[0][most_relevant_genes].sort_values().index)
dict_genes = {g:i for i,g in enumerate(X_train.columns)}
genes_idx = [dict_genes[g] for g in genes]  
print('Normalizing')
norm = MinMaxScaler()
X_train = norm.fit_transform(X_train)
X_test = norm.transform(X_test)

all_genes = rpm_log_new.columns.tolist()

max_features = [1000]
max_leaf_nodes = None #100

for i in range(1000):
    
    print('---------------run #',i,'-----------------')
    print('--------------- Standard Decision Tree -----------------')

    classifier =  tree.DecisionTreeClassifier(random_state=i, criterion="gini",  w_prior=np.array(penalties), pk_configuration='standard', pk_function=None)
    classifier.fit(X_train, y_train)
   
    split_features = rpm_log_new.columns[classifier.tree_.feature[classifier.tree_.feature != -2]]
    print(classifier.tree_.feature)
    print("original splitting features: ",split_features.tolist())
    
    dct_features = {x: [] for x in genes}
    dct_n_features = {x: [] for x in genes}

    for f in genes_idx:
        dct_features[all_genes[f]].append(all_genes[f] in split_features)
        dct_n_features[all_genes[f]].append(len(split_features))

    for f in genes_idx:
        print('------------------------------------------------------', all_genes[f],'-------------------------------------------------')

        penalties_n1 = np.ones(len(penalties)) #penalties.copy()
        penalties_n1[f] = penalties[f]
        #print(penalties_n1)

        classifier = tree.DecisionTreeClassifier(random_state=i, criterion="gini", w_prior=np.array(penalties_n1), pk_configuration="on_impurity_improvement", v=1, k=1, pk_function=None)
        classifier.fit(X_train, y_train)

        split_features = rpm_log_new.columns[classifier.tree_.feature[classifier.tree_.feature != -2]]

        if f in genes_idx:            
            dct_features[all_genes[f]].append(all_genes[f] in split_features)
            dct_n_features[all_genes[f]].append(len(split_features))
   
        values = np.linspace(0.5, 1.0, 50)
    
        for value in values:

            penalties_new  = penalties_n1.copy()
            penalties_new[f] = value

            classifier = tree.DecisionTreeClassifier(random_state=i, criterion="gini", w_prior=np.array(penalties_new), pk_configuration="on_impurity_improvement", v=1, k=1, pk_function=None)
            classifier.fit(X_train, y_train)
        
            split_features = rpm_log_new.columns[classifier.tree_.feature[classifier.tree_.feature != -2]]
            print(f"{all_genes[f]} : {value} -> {split_features}")
            if f in genes_idx:
                dct_features[all_genes[f]].append(all_genes[f] in split_features)
                dct_n_features[all_genes[f]].append(len(split_features))
    
   
    print({x:np.sum(dct_features[x]) for x in dct_features})
