import pandas as pd
from sklearn import model_selection


class CrossValidation:
    """
    - binary classification
    - multi class classification
    - multi label classification
    - single column regression
    - multi column regression
    - holdout
    """

    def __init__(
        self,
        df: pd.DataFrame,
        target_cols: list,
        problem_type='binary_classification',
        num_folds=5,
        shuffle=True
    ):
        self.df = df
        self.target_cols = target_cols
        self.num_targets = len(target_cols)
        self.problem_type = problem_type
        self.num_folds = num_folds
        self.shuffle = shuffle
        # create a new column for kfold
        self.df['kfold'] = -1

    def split(self):
        """
        fold data according to problem type
        """

        # binary classification, use StratifiedKFold
        if self.problem_type in ['binary_classification', 'multiclass_classification']:
            unique_values = self.df[self.target_cols[0]].nunique()
            if unique_values < 2:
                raise Exception(f'unique value {unique_values} < 2')
            else:
                target = self.target_cols[0]
                # init kfold
                kf = model_selection.StratifiedKFold(
                    n_splits=self.num_folds,
                    shuffle=self.shuffle)
                # fold dataset
                for fold, (_, validate_idx) in enumerate(kf.split(X=self.df, y=self.df[target].values)):
                    self.df.loc[validate_idx, 'kfold'] = fold
        return self.df


# test the class
if __name__ == '__main__':
    df = pd.read_csv('./input/train.csv')
    cv = CrossValidation(df, target_cols=['target'])
    df_split = cv.split()
    print(f'number of reocords: {len(df_split)}')
    print(df_split['kfold'].value_counts())
