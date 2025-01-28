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
from utils.utils import load_txt

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

rpm_log_new = pd.concat([rpm_log[genes_set],rpm_log[most_relevant_genes]],axis=1)

X_train, X_test,  y_train, y_test = train_test_split(rpm_log_new, y, random_state = 42,
                                        test_size = 0.2, stratify=y)

norm = MinMaxScaler()
X_train = norm.fit_transform(X_train)
X_test = norm.transform(X_test)

penalties = np.ones_like(penalties)

for i in range(1,101):
    r = np.linspace(0.5, 1.0, 50)
    print("------------------------------------------------------------------------------------------------------------------------")
    for v in r:
        penalties[0] = v
        print("------------------------------------------------GIS[0] = ",penalties[0],"-----------------------------------")
        classifier = tree.DecisionTreeClassifier(random_state=3*i, criterion="gini", w_prior=np.array(penalties), pk_configuration="on_feature_sampling", v=1, k=1, max_features='sqrt', pk_function=None) 
        classifier.fit(X_train, y_train)