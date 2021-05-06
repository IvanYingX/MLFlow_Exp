from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier


def choice(args):
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
    
    model_params = [{'penalty': args.penalty, 'C': args.C},
                    {'penalty': args.penalty, 'alpha': args.alpha},
                    {'n_neighbors': args.neighbors, 'weights': args.knn_weight},
                    {'n_estimators': args.n_estimator, 'min_samples_split': args.min_split, 'min_samples_leaf': args.min_leaf},
                    {'n_estimators': args.n_estimator, 'learning_rate': args.learning_rate}]

    model_dict = {n: [model, p] for n, model, p in zip(model_names, model_list, model_params)}
    clf = GridSearchCV(model_dict[args.model][0],
                       param_grid=model_dict[args.model][1],
                       cv=args.fold, scoring='accuracy',
                       verbose=args.verbosity)
    return clf