import pandas as pd
import numpy as np
import os
from sklearn import preprocessing
from sklearn import metrics
import joblib
from datetime import datetime


def predict(test_data_path, model_type, model_path):
    df = pd.read_csv(test_data_path)
    test_idx = df['id'].values
    predictions = None

    for FOLD in range(5):
        df = pd.read_csv(test_data_path)
        label_encoders = joblib.load(os.path.join(
            model_path, f'{model_type}_label_encoder_{FOLD}.pkl'))
        cols = joblib.load(os.path.join(
            'models', 'column.pkl'))
        for c in cols:
            lbl = label_encoders[c]
            df.loc[:, c] = df.loc[:, c].astype(str).fillna("NONE")
            df.loc[:, c] = lbl.transform(df[c].values.tolist())

        clf = joblib.load(os.path.join(
            model_path, f'{model_type}_model_{FOLD}.pkl'))
        df = df[cols]
        preds = clf.predict_proba(df)[:, 1]

        if FOLD == 0:
            predictions = preds
        else:
            predictions += preds

    predictions /= 5

    result = pd.DataFrame(np.column_stack(
        (test_idx, predictions)), columns=['id', 'target'])
    result['id'] = result['id'].astype(int)
    return result


if __name__ == '__main__':
    model_type = 'extratrees'
    submission = predict(test_data_path='input/cat_II/test.csv',
                         model_type=model_type,
                         model_path='models')
    datestr = datetime.now().date()
    submission.to_csv(
        f'output/submission_cat_II_{model_type}_{datestr}.csv',
        index=False)
