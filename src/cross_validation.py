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
        multilabel_delimiter=',',
        shuffle=True
    ):
        self.df = df
        self.target_cols = target_cols
        self.num_targets = len(target_cols)
        self.problem_type = problem_type
        self.num_folds = num_folds
        self.shuffle = shuffle
        self.multilabel_delimiter = multilabel_delimiter
        # create a new column for kfold
        self.df['kfold'] = -1

    def split(self):
        """
        fold data according to problem type
        """

        # binary or multi class classification, use StratifiedKFold
        if self.problem_type in ['binary_classification', 'multi_class_classification']:
            if self.num_targets != 1:
                raise Exception(
                    'Invalid number of targets for this problem type')
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
        # single and multi column regression
        elif self.problem_type in ['single_col_regression', 'multi_col_regression']:
            if (self.num_targets != 1 and self.problem_type == 'single_col_regression') \
                    or (self.num_targets < 2 and self.problem_type == 'multi_col_regression'):
                raise Exception(
                    'Invalid number of targets for this problem type')
            kf = model_selection.KFold(n_splits=self.num_folds)
            for fold, (_, validate_idx) in enumerate(kf.split(X=self.df)):
                self.df.loc[validate_idx, 'kfold'] = fold
        # holdout_5 holdout_10
        elif self.problem_type.startswith('holdout_'):
            holdout_percentage = int(self.problem_type.split('_')[1])
            self.df.loc[:int(
                len(self.df) * holdout_percentage / 100), 'kfold'] = 0
            self.df.loc[int(
                len(self.df) * holdout_percentage / 100):, 'kfold'] = 1
        # multi label
        elif self.problem_type == 'multi_label_classification':
            if self.num_targets != 1:
                raise Exception(
                    'Invalid number of targets for this problem type')
            # split the labels on target column
            num_targets = self.df[self.target_cols[0]].apply(
                lambda x: len(str(x).split(self.multilabel_delimiter)))
            # init kfold
            kf = model_selection.StratifiedKFold(
                n_splits=self.num_folds,
                shuffle=self.shuffle)
            # fold dataset
            for fold, (_, validate_idx) in enumerate(kf.split(X=self.df, y=num_targets)):
                self.df.loc[validate_idx, 'kfold'] = fold

        else:
            raise Exception('Problem type not found')

        return self.df


# test the class
if __name__ == '__main__':
    df = pd.read_csv('./input/train_imet.csv')
    cv = CrossValidation(
        df,
        target_cols=['attribute_ids'],
        problem_type='multi_label_classification',
        multilabel_delimiter=' '
    )
    df_split = cv.split()
    print(f'number of reocords: {len(df_split)}')
    print(df_split['kfold'].value_counts())
