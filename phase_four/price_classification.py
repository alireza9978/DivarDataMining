import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score


def inverse_convert_category(temp_df: pd.DataFrame, columns):
    for col in columns:
        my_dict = np.load("./datasets/converting_map_{}.npy".format(col), allow_pickle='TRUE').item()
        temp_df[col] = temp_df[col].map(my_dict)
    return temp_df


def one():
    temp_df = pd.read_csv("./datasets/divar_posts_dataset_cleaned.csv", low_memory=False)
    columns_to_use = ['cat1', 'cat2', 'cat3', 'city', 'price']
    temp_df = temp_df[columns_to_use]
    temp_df = temp_df[temp_df.price.notna()].reset_index(drop=True)
    temp_df["label"] = 0

    temp_index = temp_df.sample(frac=0.05).index
    small_df = temp_df.loc[temp_index]
    temp_df = temp_df.drop(index=temp_index)
    small_df["label"] = 1
    small_df["price"] = small_df["price"] * 0.2

    temp_index = temp_df.sample(frac=0.05).index
    small_df_two = temp_df.loc[temp_index]
    temp_df = temp_df.drop(index=temp_index)
    small_df_two["label"] = 1
    small_df_two["price"] = small_df_two["price"] * 5

    temp_index = temp_df.sample(frac=0.05).index
    small_df_three = temp_df.loc[temp_index]
    temp_df = temp_df.drop(index=temp_index)
    small_df_three["label"] = 1
    small_df_three["price"] = 0

    temp_df = temp_df.reset_index(drop=True).append(small_df.reset_index(drop=True))
    temp_df = temp_df.reset_index(drop=True).append(small_df_two.reset_index(drop=True))
    temp_df = temp_df.reset_index(drop=True).append(small_df_three.reset_index(drop=True))

    x = temp_df.drop(columns=["label"]).to_numpy()
    y = temp_df["label"].to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20, shuffle=True)

    x_scaler = MinMaxScaler()
    X_train = x_scaler.fit_transform(X_train)
    X_test = x_scaler.transform(X_test)

    y_scaler = MinMaxScaler()
    y_train = y_scaler.fit_transform(y_train.reshape(-1, 1))
    y_test = y_scaler.transform(y_test.reshape(-1, 1))

    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_pred = y_pred.reshape(-1, 1)
    print(confusion_matrix(y_test, y_pred))
    print("accuracy_score ", accuracy_score(y_test, y_pred))
    print("precision_score ", precision_score(y_test, y_pred))
    print("recall_score ", recall_score(y_test, y_pred))


one()
