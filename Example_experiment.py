import argparse
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score


parser = argparse.ArgumentParser(description='Predict breast cancer')
parser.add_argument('-m', '--model',
                    type=str,
                    help='The model to use',
                    action='store')
parser.add_argument('-f', '--fold',
                    type=int,
                    help='Number of splits in the Cross Validation',
                    action='store',
                    default=5)
parser.add_argument('-t', '--test_size',
                    type=float,
                    help='Size of the test samples (%)',
                    action='store',
                    default=0.2)
parser.add_argument('-v', '--verbosity',
                    type=int,
                    action='store',
                    default=1)                    
# arguments for Logistic Regression
parser.add_argument('--penalty',
                    type=str,
                    nargs='+',
                    default=['l2'],
                    help='Penalty to the loss function (l1, l2, elasticnet)')
parser.add_argument('--C',
                    type=float,
                    nargs='+',
                    default=[1.0],
                    help='Inverse of regularization strength; must be a positive float.')
# arguments for SGD
parser.add_argument('--alpha',
                    type=float,
                    nargs='+',
                    default=[0.0001])
# arguments for KNN               
parser.add_argument('--neighbors',
                    type=int,
                    nargs='+',
                    default=[5])
parser.add_argument('--knn_weight',
                    type=str,
                    nargs='+',
                    default=['uniform'],
                    help='weight function used in prediction (uniform, distance)')
# arguments for Random Forest
parser.add_argument('--n_estimator',
                    type=int,
                    nargs='+',
                    default=[100])  
parser.add_argument('--min_split',
                    type=int,
                    nargs='+',
                    default=[2],
                    help='The minimum number of samples required to split an internal node')
parser.add_argument('--min_leaf',
                    type=int,
                    nargs='+',
                    default=[1],
                    help='The minimum number of samples required to be at a leaf node')
# arguments for Adaboost
parser.add_argument('-lr', '--learning_rate',
                    type=float,
                    nargs='+',
                    default=[1])


args = parser.parse_args()

model_names = ['LogisticRegression',
               'SGDClassifier',
               'KNeighborsClassifier',
               'RandomForestClassifier',
               'AdaBoostClassifier']
assert args.model in model_names, f'Model not in the list \
    {" ".join(model_names)}'

model_list = [LogisticRegression(),
              SGDClassifier(), 
              KNeighborsClassifier(),
              RandomForestClassifier(),
              AdaBoostClassifier()]

model_dict = {n:model for n, model in zip(model_names, model_list)}



if args.model == 'LogisticRegression':
    params = {'penalty': args.penalty,
              'C': args.C}
    clf = GridSearchCV(LogisticRegression(),
                       param_grid=params,
                       cv=args.fold, scoring='accuracy',
                       verbose=args.verbosity)
elif args.model == 'SGDClassifier':
    params = {'penalty': args.penalty,
              'alpha': args.alpha}
    clf = GridSearchCV(SGDClassifier(),
                       param_grid=params,
                       cv=args.fold, scoring='accuracy',
                       verbose=args.verbosity)
elif args.model == 'KNeighborsClassifier':
    params = {'n_neighbors': args.neighbors,
              'weights': args.knn_weight}
    clf = GridSearchCV(KNeighborsClassifier(),
                       param_grid=params,
                       cv=args.fold, scoring='accuracy',
                       verbose=args.verbosity)
elif args.model == 'RandomForestClassifier':
    params = {'n_estimators': args.n_estimator,
              'min_samples_split': args.min_split,
              'min_samples_leaf': args.min_leaf}
    clf = GridSearchCV(RandomForestClassifier(),
                       param_grid=params,
                       cv=args.fold, scoring='accuracy',
                       verbose=args.verbosity)
elif args.model == 'AdaBoostClassifier':
    params = {'n_estimators': args.n_estimator,
              'learning_rate': args.learning_rate}
    clf = GridSearchCV(AdaBoostClassifier(),
                       param_grid=params,
                       cv=args.fold, scoring='accuracy',
                       verbose=args.verbosity)

X, y = load_breast_cancer(return_X_y=True)
X, X_test, y, y_test = train_test_split(X, y, test_size=args.test_size)


clf.fit(X, y)
print(pd.DataFrame(clf.cv_results_))

y_pred = clf.predict(X_test)
print(accuracy_score(y_test, y_pred))