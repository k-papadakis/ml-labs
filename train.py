import math

import joblib
import numpy as np
import matplotlib.pyplot as plt
import openml

from imblearn.pipeline import Pipeline as ILPipeline
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek, SMOTEENN

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score, f1_score, make_scorer, ConfusionMatrixDisplay
from sklearn.pipeline import Pipeline

from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning


openml.config.apikey = 'd3aa70dc3cdf0aeac05a5400172fd1e1'
RANDOM_STATE = 42


dataset = openml.datasets.get_dataset(41990)

X, y, categorical_indicator, attribute_names = dataset.get_data(
    dataset_format="array",
    target=dataset.default_target_attribute
    )

classes, class_counts = np.unique(y,return_counts=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3, shuffle=True,
    random_state=RANDOM_STATE
)

clf_classes = [
    DummyClassifier,
    GaussianNB,
    KNeighborsClassifier,
    LogisticRegression,
    MLPClassifier,
    SVC
]

pipes = {}
for clf_class in clf_classes[1:]:
    pipes[clf_class.__name__] = ILPipeline([
        ('balancer', RandomUnderSampler(random_state=RANDOM_STATE)),
        ('clf', clf_class()),
    ])

param_grids = {
    'GaussianNB': {
        'clf__priors': [
            # Uniform prior, same as `None` after balancing.
            np.full(len(classes), 1/len(classes)),
            # The prior before balancing.
            np.unique(y_train, return_counts=True)[1] / y_train.shape[0]
        ],
        # This is isn't really a hyperparameter. It's for numerical stability. 
        'clf__var_smoothing': [1e-09]  
    },
    'KNeighborsClassifier': {
        'clf__n_neighbors': [5, 20],
        'clf__weights': ['uniform', 'distance'],
        'clf__p': [1, 2],
        # 'clf__n_jobs': [-1]
    },
    'LogisticRegression': {
        'clf__penalty': ['l2'],
        'clf__C': [0.1, 1., 10.],
        'clf__max_iter': [400],
        # 'clf__n_jobs': [-1],
        'clf__random_state': [RANDOM_STATE]
    },
    'MLPClassifier': {
        'clf__hidden_layer_sizes': [(64,), (32, 64)],
        'clf__activation': ['logistic', 'relu'],
        'clf__alpha': [0.0001, 0.01],
        'clf__learning_rate_init': [0.001, 0.0001],
        'clf__batch_size': [256],
        'clf__max_iter': [400],
        'clf__random_state': [RANDOM_STATE]
    },
    'SVC': {
        'clf__C': [0.1, 1., 10.],
        'clf__kernel': ['linear', 'rbf'],
        'clf__max_iter': [400],
        'clf__random_state': [RANDOM_STATE]
    }
}

def cv_train(scoring):
    cv_clfs = {}
    for name, pipe in pipes.items():
        print('\n'*2 + '-'*30 + name + '-'*30)
        estimator = GridSearchCV(
            pipe, param_grids[name], cv=10,
            scoring=scoring,
            error_score='raise',
            verbose=2, n_jobs=-1
        )
        estimator.fit(X_train, y_train)
        cv_clfs[name] = estimator
    return cv_clfs

print('='*40 + 'ACCURACY' + '='*40)
cv_clfs_acc = cv_train(scoring='accuracy')
print('\n'*3 + '='*40 + 'F1 MACRO' + '='*40)
cv_clfs_f1 = cv_train(scoring='f1_macro')
models = (cv_clfs_acc, cv_clfs_f1)
joblib.dump(models, './output/models.joblib')
