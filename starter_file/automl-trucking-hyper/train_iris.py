# Modified from https://www.geeksforgeeks.org/multiclass-classification-using-scikit-learn/

import argparse
import os

# importing necessary libraries
import numpy as np
from azureml.core import Dataset
from sklearn import datasets
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.ensemble import BaggingClassifier
from sklearn.multiclass import OneVsRestClassifier
import joblib

from azureml.core.run import Run
#run = Run.get_context()
from azureml.core import Workspace, Dataset

subscription_id = '473e7e3b-e0e5-4760-92a4-0b9e3c06f260'
resource_group = 'MachineLearning'
workspace_name = 'MLTraining'
print("Imported libraries")
ws = Workspace(subscription_id, resource_group, workspace_name)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--tol', type=str, default='0.0001',
                        help='Tolerance for stopping')
    parser.add_argument('--coefficient', type=float, default=1,
                        help='Coefficient parameter')
    parser.add_argument(
        '--input-data',
        type=str,
        help='Path to the training data',
    )

    args = parser.parse_args()
    #run.log('Kernel type', np.str(args.kernel))
    #run.log('Penalty', np.float(args.penalty))
    print("arguments parsed")
    # loading the dataset
    #ws = run.experiment.workspace
    #dataset = Dataset.get_by_id(ws, id=args.input_data)
    dataset = Dataset.get_by_name(ws, name='Trucking Apps Cleaned')
    print("processing dataset")
    # X -> features, y -> label
    df = dataset.to_pandas_dataframe()
    df.drop('cre_militarydischargedon', axis=1, inplace=True)
    from sklearn import preprocessing
    le = preprocessing.LabelEncoder()
    for column_name in df.columns:
        if df[column_name].dtype == object:
            df[column_name] = le.fit_transform(df[column_name])
        else:
            pass

    y_col = ['workmonths']
    X = df.loc[:, df.columns != 'workmonths']
    y = df.loc[:, y_col]
    print("scaling sparse data")
    max_scaler = preprocessing.MaxAbsScaler()
    X_maxabs = max_scaler.fit_transform(X, y)
    print("dividing into test/train")
    # dividing X, y into train and test data
    X_train, X_test, y_train, y_test = train_test_split(X_maxabs, y, random_state=42)

    # training a linear SVM classifier
    from sklearn.svm import LinearSVC
    clf = OneVsRestClassifier(LinearSVC(class_weight='balanced', verbose=1, max_iter=2000, C=args.coefficient, tol=args.tol), n_jobs=-1)
    #score = cross_val_score(clf, X, y, cv=5, scoring='roc_auc_ovr_weighted').mean()
    #print('AUC Weighted of SVM classifier {:.2f}'.format(score))
    #run.log('AUC Weighted', np.float(score))
    print("fitting model")
    svm_model_linear = clf.fit(X_train, np.ravel(y_train,order='C'))
    print("making predictions")
    svm_predictions = svm_model_linear.predict(X_test)
    print("scoring model")
    # model accuracy for X_test
    accuracy = svm_model_linear.score(X_test, y_test)
    print('Accuracy of SVM classifier on test set: {:.2f}'.format(accuracy))
    #run.log('Accuracy', np.float(accuracy))
    # creating a confusion matrix
    cm = confusion_matrix(y_test, svm_predictions)
    print(cm)

    os.makedirs('outputs', exist_ok=True)
    # files saved in the "outputs" folder are automatically uploaded into run history
    joblib.dump(svm_model_linear, 'outputs/model.joblib')


if __name__ == '__main__':
    main()
