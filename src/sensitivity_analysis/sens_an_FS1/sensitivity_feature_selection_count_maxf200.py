import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from pktree import tree
from sklearn.utils import shuffle
import os
def load_txt(filename):
    content = []
    with open(filename)as f:
        for line in f:
            content.append(line.strip())
    return content

dire_kidney = '/home/mongardi/tree-based-models/Biology-informed_sklearn/data_tree/data_kidney/Kidney_df_tr_coding_new.csv'

df = pd.read_csv(dire_kidney)
df = shuffle(df, random_state=42)
df.set_index(df.columns[0], inplace=True)
penalties = np.loadtxt("/home/mongardi/tree-based-models/Biology-informed_sklearn/data_tree/penalties_kidney.txt")
genes_set = load_txt("/home/mongardi/tree-based-models/Biology-informed_sklearn/data_tree/data_kidney/controlled_features.txt")

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
        classifier = tree.DecisionTreeClassifier(random_state=3*i, criterion="gini", w_prior=np.array(penalties), pk_configuration="on_feature_sampling", v=1, k=1, max_features=200) 
        classifier.fit(X_train, y_train)