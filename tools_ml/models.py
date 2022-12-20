from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer



def get_pipeline(features_num: list, features_cat: list, classifier, random_state: int = None):
    pipe_tr_features_num = Pipeline([
        ('tr_imput_mean', SimpleImputer(strategy='mean')),
        ('tr_min_max', StandardScaler())
    ])

    pipe_tr_features_cat = Pipeline([
        ('tr_input_frequent', SimpleImputer(strategy='most_frequent')),
        ('tr_dummy', OneHotEncoder(drop='first', handle_unknown='ignore'))
    ])

    pre_processor = ColumnTransformer([
        ('tr_num', pipe_tr_features_num, features_num),
        ('tr_cat', pipe_tr_features_cat, features_cat)
    ])

    pipe_final = Pipeline([
        ('pre_processor', pre_processor),
        ('classifier', classifier)
    ])

    return pipe_final