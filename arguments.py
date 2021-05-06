import argparse


def parse():
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

    return args