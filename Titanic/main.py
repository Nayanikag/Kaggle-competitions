import pandas as pd
import csv
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn import preprocessing
from sklearn import metrics
from sklearn.model_selection import KFold
import numpy as np
from scipy import stats

# Feature engineering and selection
def get_trainable_features(df):
    df = df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Fare', 'Cabin'], axis=1)
    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Age_band'] = pd.cut(df['Age'], 4)
    df['Age_band'] = pd.Categorical(df['Age_band'])
    df['Age'] = df['Age_band'].cat.codes
    df = df.drop(columns='Age_band')
    df['Sex'] = pd.Categorical(df['Sex'])
    df['Sex'] = df['Sex'].cat.codes
    df['Embarked'] = pd.Categorical(df['Embarked'])
    df['Embarked'] = df['Embarked'].cat.codes
    df['Family_size'] = df['SibSp'] + df['Parch'] + 1
    df['Family_band'] = pd.cut(df['Family_size'], 3)
    df['Family_band'] = pd.Categorical(df['Family_band'])
    df['Family_size'] = df['Family_band'].cat.codes
    df = df.drop(columns=['Family_band', 'SibSp', 'Parch'])
    return df

def train_and_predict():
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    test_pid = test_df['PassengerId']
    train_Y = train_df['Survived']
    train_df.drop(columns='Survived', inplace=True)
    train_si = 0
    train_ei = len(train_df)
    test_si = train_ei
    test_ei = train_ei + len(test_df) + 2
    merged = pd.concat([train_df, test_df], copy=True)
    merged_processed = get_trainable_features(merged)
    train_df = merged_processed[train_si:train_ei]
    test_df = merged_processed[test_si:test_ei + 1]
    kf = KFold(n_splits=5)
    kf.get_n_splits(train_df)
    total_error = 0.0
    models = []
    # 5-fold cross validation
    for train_index, val_index in kf.split(train_df):
        X_train, X_val = train_df.iloc[train_index], train_df.iloc[val_index]
        y_train, y_val = train_Y.iloc[train_index], train_Y.iloc[val_index]
        model = RandomForestClassifier(random_state=0).fit(X_train, y_train)
        model = svm.SVC().fit(X_train, y_train)
        models.append(model)
        y_pred = model.predict(X_val)
        total_error += metrics.accuracy_score(y_pred, y_val)
    print("test(evaluation error) after 5-fold cross validation is", float(total_error) / 5)

    test_ys = []
    for i in range(len(models)):
        test_y = models[i].predict(test_df)
        test_ys.append(test_y)

    # Take the vote of each of the model trained in cross validation step and take the prediction class with maximum votes
    test_y_fin = np.full_like(test_y, 0)
    for j in range(len(test_y)):
        test_y_fin[j] = stats.mode(np.array([test_ys[0][j], test_ys[1][j], test_ys[2][j], test_ys[3][j], test_ys[4][j]]))[0][0]

    # Write predictions to Submission.csv

    with open("Submission.csv", 'w') as csvfile:
        headers = ['PassengerId', 'Survived']
        writer = csv.DictWriter(csvfile, delimiter=',', lineterminator='\n', fieldnames=headers)
        writer.writeheader()
        for j in range(len(test_df)):
            test_res = [test_pid[j], test_y[j]]
            writer.writerow(
            {headers[i]: test_res[i] for i in range(2)})

def main():
    train_and_predict()

main()