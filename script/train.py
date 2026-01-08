# %%
import pandas as pd
import pickle
import numpy as np
import xgboost as xgb
from sklearn.feature_extraction import DictVectorizer






def load_data():
    df = pd.read_csv('../data/data.csv')

    # rename all columns so all columns have uniform variable naming style
    df.columns = df.columns.str.lower().str.replace(' ', '_')

    # collect all columns whose data types are strings
    columns_list = list(df.dtypes[df.dtypes == 'object'].index)

    # rename data in each column to have uniform variable naming style
    for col in columns_list:
        df[col] = df[col].str.lower().str.replace(' ', '_')
    
    # Map target variable values to binary value
    target_variable = 'chronic_medical_conditions'
    df[target_variable] = df[target_variable].map({'yes': 1, 'no': 0})

    # drop name column. it's not needed
    del df['name']

    # After repeated kernel crash during parameter tuning for random forest, I realized the dataset (248k rows) is too large to fit above 100 tree random forest in a remote environment (code space). Memory usage grows with number of trees. your RAM usage spikes massively. Linux detects this and kills the Python process (kernel crash). The cut down of the data into smaller size down in the following cell below is not necessary if the computation resources are available.
    df = df.sample(frac=0.075, random_state=42)


    return df


def train_model(df):
    categorical_features = list(df.dtypes[df.dtypes == 'object'].index)
    numerical_features = list(df.dtypes[df.dtypes != 'object'].index)

    # From the output below, numerical_fetures includes converted variable which is our target therefore the tagert variable has to be removed
    features = (categorical_features + numerical_features)
    features.remove('chronic_medical_conditions')


    X_train_dict = df[features].to_dict(orient='records')

    y_train = df.chronic_medical_conditions



    # One Hot Encoding
    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(X_train_dict)

    # Training
    dtrain = xgb.DMatrix(X_train, label=y_train)
    xgb_params = {
    'eta': 0.1, 
    'max_depth': 3,
    'min_child_weight': 1,
    
    'objective': 'binary:logistic',
    'nthread': 8,
    
    'seed': 1,
    'verbosity': 1,
    }

    model = xgb.train(xgb_params, dtrain, num_boost_round=10)

    return dv, model

def save_model(dv, model, output_file):
    with open(output_file, 'wb') as f_out:
        pickle.dump((dv, model), f_out)

df = load_data()
dv, model = train_model(df)
save_model(dv, model, 'model.bin')
