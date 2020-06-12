import pandas as pd
from sklearn import preprocessing


class CategoricalFeatures:
    def __init__(self, df, categorical_features, encode_type, fill_na=False):
        """
        df: pandas dataframe
        categorical_features (list(string)): list of features
        fill_na(boolean): if fill NaN
        encode_type: one of label, ohe (one hot value)
        """
        self.features = categorical_features
        self.encode_type = encode_type
        self.label_encoders = {}
        self.binary_encoders = {}
        self.fill_na = fill_na
        self.df = self._fill_na(df)

    def _fill_na(self, df):
        if self.fill_na:
            for col in self.features:
                df.loc[:, col] = df.loc[:, col].astype(
                    str).fillna('-9999999')
        return df

    def _lable_encoding(self):
        for col in self.features:
            le = preprocessing.LabelEncoder()
            le.fit(self.df[col].values)
            # update value with transformed one
            self.df.loc[:, col] = le.transform(self.df[col].values)
            self.label_encoders[col] = le
        return self.df

    def _label_binarization(self):
        for col in self.features:
            le = preprocessing.LabelBinarizer()
            le.fit(self.df[col].values)
            val = le.transform(self.df[col].values)
            self.binary_encoders[col] = le

            # for each 'new' column created by transformer
            for j in range(val.shape[1]):
                new_col_name = col + f"__bin_{j}"
                self.df[new_col_name] = val[:, j]

            # drop current column
            self.df = self.df.drop(col, axis=1)
        return self.df

    def fit_transform(self):
        if self.encode_type == 'label':
            return self._lable_encoding()
        elif self.encode_type == 'binary':
            return self._label_binarization()
        else:
            raise Exception('unknown encoding type')

    def transform_only(self, df):
        df = self._fill_na(df)
        if self.encode_type == 'label':
            for col in self.features:
                df.loc[:, col] = self.label_encoders[col].transform(
                    df[col].values)
            return df
        elif self.encode_type == 'binary':
            for col in self.features:
                val = self.binary_encoders[col].transform(df[col].values)
                for j in range(val.shape[1]):
                    new_col_name = col + f"_bin_{j}"
                    df[new_col_name] = val[:, j]
                df = df.drop(col, axis=1)
            return df
        else:
            raise Exception('unknown encoding type')


if __name__ == '__main__':
    # cat = 'binary'
    cat = 'label'
    if cat == 'binary':
        train_df = pd.read_csv('./input/train_cat_II.csv').head(500)
        features = [
            col for col in train_df.columns if col not in ['id', 'target']]
        temp_df = train_df.copy(deep=True)
        cat_encode = CategoricalFeatures(
            temp_df, features, cat, fill_na=True)
        train_df_encoded = cat_encode.fit_transform()
        test_df = pd.read_csv('./input/train_cat_II_test.csv').head(500)
        temp_df = test_df.copy(deep=True)
        test_df_encoded = cat_encode.transform_only(temp_df)
    elif cat == 'label':     # need full data for handle unseen labels: 'a885aacec'
        train_df = pd.read_csv('./input/train_cat_II.csv')
        train_idx = train_df['id'].values
        test_df = pd.read_csv('./input/train_cat_II_test.csv')
        test_idx = test_df['id'].values
        test_df['target'] = -1
        full_df = pd.concat([train_df, test_df])
        temp_df = full_df.copy(deep=True)
        features = [
            col for col in full_df.columns if col not in ['id', 'target']]
        cat_encode = CategoricalFeatures(
            temp_df, features, cat, fill_na=True)
        full_df_encoded = cat_encode.fit_transform()
        train_df_encoded = full_df_encoded[full_df_encoded['id'].isin(train_idx)].reset_index(
            drop=True)
        test_df_encoded = full_df_encoded[full_df_encoded['id'].isin(test_idx)].reset_index(
            drop=True)
        test_df = test_df.drop('target', axis=1)
        test_df_encoded = test_df_encoded.drop('target', axis=1)

    print(train_df)
    print(train_df_encoded)
    print(test_df)
    print(test_df_encoded)
