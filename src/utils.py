import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn import datasets, model_selection, metrics
from sklearn.metrics import make_scorer
from sklearn.metrics import recall_score, accuracy_score, precision_score, f1_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from pktree import tree

scoring = {'accuracy':'accuracy',
            'rec': make_scorer(recall_score, average= 'macro'),
            'prec': make_scorer(precision_score, average= 'macro'), 
            'f1':make_scorer(f1_score, average="macro")}

def load_data_brca(file_path, test_size = 0.2, return_columns= False, random_state=42):
    df = pd.read_csv(file_path)
    X = df.drop(columns=['sample_id', 'expert_PAM50_subtype'])
    y = df['expert_PAM50_subtype']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train)
    y_test = label_encoder.fit_transform(y_test)
    if return_columns:
        return X_train, y_train,  X_test, y_test, X.columns
    else:
        return X_train, y_train,  X_test, y_test

def load_penalties(file_path):
    penalties = np.loadtxt(file_path)
    return penalties


def load_DT_classifier(pk_configuration='no_gis', w_prior=None, k=1.0, v=1.0, 
                        max_features = None,
                        criterion = 'gini', 
                        scaler=MinMaxScaler(), random_state=42):
    tree_clf = Pipeline(steps=[('Scaler', scaler),
                    ('Model', tree.DecisionTreeClassifier(w_prior = w_prior,
                    pk_configuration=pk_configuration, k=k, v=v, max_features = max_features,
                    criterion=criterion, random_state=random_state))])

    return tree_clf

