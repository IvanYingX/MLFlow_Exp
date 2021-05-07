import arguments
import model
import data
import pandas as pd
from sklearn.metrics import accuracy_score
import mlflow
from mlflow.models.signature import infer_signature
def run(clf, X, y):
    clf.fit(X, y)
    return clf

if __name__ == "__main__":
    args = arguments.parse()
    print(args)
    # mlflow.set_experiment()
    clf = model.choice(args)
    X, X_test, y, y_test = data.get(args)
    clf = run(clf, X, y)
    with mlflow.start_run():
        # print(pd.DataFrame(clf.cv_results_))
        best_model = clf.best_estimator_
        best_score = clf.best_score_
        best_params = clf.best_params_
        y_pred = clf.predict(X_test)

        signature = infer_signature(X, clf.predict(X))
    # mlflow.sklearn.save_model(best_model,
    #                     f"{args.model}_model",
    #                     signature=signature)

        mlflow.sklearn.log_model(best_model,
                                    f"{args.model}_model",
                                    signature=signature)
        # mlflow.log_dict('best_params', best_params)
        mlflow.log_metric('Train_Acc', best_score)
        mlflow.log_metric('Val_Acc', accuracy_score(y_test, y_pred))
        print(f'Accuracy on test set: {accuracy_score(y_test, y_pred)}')
    