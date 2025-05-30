import pandas as pd

def prepare_data():
    df = pd.read_csv('../data/intent_Train.csv')
    df_valid = pd.read_csv('../data/intent_Valid.csv')
    df_test = pd.read_csv('../data/intent_Test.csv')

    df = df.sample(frac=1, random_state=26)

    df.drop(['category','tags'], axis=1, inplace=True)
    df_valid.drop(['category','tags'], axis=1, inplace=True)
    # df_valid.head()
    df_test.drop(['category','tags'], axis=1, inplace=True)

    X_train = df['utterance']
    X_valid = df_valid['utterance']
    X_test = df_test['utterance']
    y_train = df['intent']
    y_valid = df_valid['intent']
    y_test = df_valid['intent']

    return X_train,X_valid,X_test,y_train,y_valid,y_test


