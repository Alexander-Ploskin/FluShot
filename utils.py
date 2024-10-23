import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from category_encoders import OrdinalEncoder
import xgboost as xgb


def apply_preprocessor(
    preprocessor: ColumnTransformer, one_hot: bool, cat_cols, ord_cols, num_cols
):
    train_features_transformed = preprocessor.fit_transform(train_features)
    test_features_transformed = preprocessor.fit_transform(test_features)

    # Get feature names for transformed data
    if one_hot:
        cat_feature_names = (
            preprocessor.named_transformers_["cat"]
            .named_steps["onehot"]
            .get_feature_names_out(cat_cols)
        )
    else:
        cat_feature_names = np.array(cat_cols)
    ord_feature_names = np.array(ord_cols)

    # Combine all feature names into a single list
    all_feature_names = np.concatenate((num_cols, cat_feature_names, ord_feature_names))

    train_features_transformed_df = pd.DataFrame(
        train_features_transformed, columns=all_feature_names
    )
    test_features_transformed_df = pd.DataFrame(
        test_features_transformed, columns=all_feature_names
    )

    return train_features_transformed_df, test_features_transformed_df


def prepare_data_for_xgboost(num_cols, cat_cols, ord_cols):
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value=-1)),
            ("scaler", MinMaxScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    ordinal_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("ordinal", OrdinalEncoder()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols),
            ("ord", ordinal_transformer, ord_cols),
        ]
    )

    return apply_preprocessor(preprocessor, True)


def prepare_data_for_catboost():
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value=-1)),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ]
    )

    ordinal_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("ordinal", OrdinalEncoder()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols),
            ("ord", ordinal_transformer, ord_cols),
        ]
    )

    return apply_preprocessor(preprocessor, False)
