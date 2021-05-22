import ast

import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.preprocessing import TransactionEncoder


def inverse_convert_category(temp_df: pd.DataFrame, columns):
    for col in columns:
        my_dict = np.load("./datasets/converting_map_{}.npy".format(col), allow_pickle='TRUE').item()
        temp_df[col] = temp_df[col].map(my_dict)
    return temp_df


def divar_one():
    df = pd.read_csv("./datasets/divar_posts_dataset_cleaned.csv", low_memory=False)
    df = df.drop(columns=["desc", "year", "title", "mileage", "price", "archive_by_user", "brand", "type"])
    temp_columns = ["cat2", "city"]
    test_one_df = inverse_convert_category(df[temp_columns], temp_columns)
    print(test_one_df)
    encoder = TransactionEncoder()
    encoded = encoder.fit_transform(test_one_df.to_numpy())
    df = pd.DataFrame(encoded, columns=encoder.columns_)

    ans = apriori(df, min_support=0.01, use_colnames=True)
    ans['length'] = ans['itemsets'].apply(lambda x: len(x))
    ans = ans[ans.length > 1]
    print(ans.sort_values("support", ).tail(5))


def digikala_one():
    df = pd.read_csv("./datasets/clean_orders.csv", converters={"ID_Item": ast.literal_eval}, low_memory=False,
                     index_col=0)

    df['length'] = df.ID_Item.apply(lambda x: len(x))
    df = df[df['length'] > 1]
    encoder = TransactionEncoder()
    encoded_df = encoder.fit_transform(df.ID_Item)
    sparse_df = pd.DataFrame(encoded_df, columns=encoder.columns_)
    # sparse_df.columns = [str(i) for i in sparse_df.columns]

    ans = apriori(sparse_df, min_support=0.00001, low_memory=True)
    ans['length'] = ans['itemsets'].apply(lambda x: len(x))
    ans = ans[ans['length'] > 1]
    print(ans.sort_values("support", ascending=False))


# divar_one()
digikala_one()
