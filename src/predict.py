import pandas as pd
import numpy as np
import os
from sklearn import preprocessing
from sklearn import metrics
from .dispatcher import TRAIN_MODELS
import joblib

TEST_DATA = os.environ.get('TEST_DATA')
MODEL = os.environ.get('MODEL')


def predict():
    df = pd.read_csv(TEST_DATA)
    test_idx = df['id'].values

    label_encoders = joblib.load(os.path.join(
        'models', f'{MODEL}_label_encoder.pkl'))
    clf = joblib.load(os.path.join('models', f'{MODEL}.pkl'))
    cols = joblib.load(os.path.join('models', 'column.pkl'))

    df = df[cols]
    for c in cols:
        lbl = label_encoders[c]
        df.loc[:, c] = lbl.transform(df[c].values.tolist())

    preds = clf.predict_proba(df)[:, 1]
    result = pd.DataFrame(np.column_stack(
        (test_idx, preds)), columns=['id', 'target'])
    result['id'] = result['id'].astype(int)
    return result


if __name__ == '__main__':
    result = predict()
    result.to_csv(f'./models/{MODEL}.csv', index=False)
