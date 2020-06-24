import pandas as pd
import os
from sklearn import model_selection

TRAINING_DATA = os.environ.get('TRAINING_DATA')
OUTPUT_DIR = os.environ.get('OUTPUT_DIR')

if __name__ == '__main__':
    df = pd.read_csv(TRAINING_DATA)
    df['kfold'] = -1

    df = df.sample(frac=1).reset_index(drop=True)

    kf = model_selection.StratifiedKFold(n_splits=5, shuffle=False)

    for fold, (train_idx, validate_idx) in enumerate(kf.split(X=df, y=df['target'].values)):
        print(len(train_idx), len(validate_idx))
        df.loc[validate_idx, 'kfold'] = fold

    df.to_csv(f'{OUTPUT_DIR}/train_folds.csv', index=False)
