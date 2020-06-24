import pandas as pd
import os
from sklearn import preprocessing
from sklearn import metrics
from .dispatcher import TRAIN_MODELS
import joblib

TRAINING_DATA = os.environ.get('TRAINING_DATA')
TEST_DATA = os.environ.get('TEST_DATA')
FOLD = int(os.environ.get('FOLD'))
MODEL = os.environ.get('MODEL')

FOLD_MAPPING = {
    0: [1, 2, 3, 4],
    1: [0, 2, 3, 4],
    2: [0, 1, 3, 4],
    3: [0, 1, 2, 4],
    4: [0, 1, 2, 3]
}


def fill_na_str(df, column):
    df.loc[:, c] = df.loc[:, c].astype(str).fillna("NONE")
    return df


if __name__ == '__main__':
    df = pd.read_csv(TRAINING_DATA)
    test_df = pd.read_csv(TEST_DATA)
    # select trainning part and validation part according to fold mapping
    train_df = df[df['kfold'].isin(FOLD_MAPPING.get(FOLD))]
    valid_df = df[df['kfold'] == FOLD]

    ytrain = train_df['target'].values
    yvalid = valid_df['target'].values

    # drop kfold column
    train_df = train_df.drop(['id', 'target', 'kfold'], axis=1)
    valid_df = valid_df.drop(['id', 'target', 'kfold'], axis=1)

    label_encoders = {}
    for c in train_df.columns:
        lbl = preprocessing.LabelEncoder()
        # fill the na
        train_df = fill_na_str(train_df, c)
        valid_df = fill_na_str(valid_df, c)
        test_df = fill_na_str(test_df, c)

        # fit catelogarical values
        lbl.fit(train_df[c].values.tolist() +
                valid_df[c].values.tolist() + test_df[c].values.tolist())
        train_df.loc[:, c] = lbl.transform(train_df[c].values.tolist())
        valid_df.loc[:, c] = lbl.transform(valid_df[c].values.tolist())
        label_encoders[c] = lbl

    # data is ready to train
    clf = TRAIN_MODELS[MODEL]
    clf.fit(train_df, ytrain)
    preds = clf.predict_proba(valid_df)[:, 1]
    print(metrics.roc_auc_score(yvalid, preds))

    # save modle
    joblib.dump(label_encoders,
                f'models/{MODEL}_label_encoder_{FOLD}.pkl')
    joblib.dump(clf, f'models/{MODEL}_model_{FOLD}.pkl')
    joblib.dump(train_df.columns, f'models/column.pkl')
