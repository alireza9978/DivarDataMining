import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from mlxtend.preprocessing import TransactionEncoder

orders_df = pd.read_csv("./datasets/orders.csv")


def one():
    # استخراج الگو‌های پرتکرار محصولات در سبد خرید مشتریان
    df = orders_df.groupby("ID_Order")["ID_Item"].apply(lambda x: x.to_list()).reset_index()
    transactions = list(df.ID_Item)
    transaction_encoder = TransactionEncoder()
    encoded_df = transaction_encoder.fit_transform(transactions, sparse=True)
    df = pd.DataFrame.sparse.from_spmatrix(encoded_df, columns=transaction_encoder.columns_)
    df.columns = [str(i) for i in df.columns]

    apriori_item_set = apriori(df, min_support=0.001, use_colnames=True)
    print(apriori_item_set)
    apriori_rules = association_rules(apriori_item_set, metric='confidence', min_threshold=0.0001)
    print(apriori_rules)


def two():
    # استخراج الگو‌های پرتکرار قیمت محصولات در سبد خرید و شهر
    column_name = "Amount_Gross_Order"
    temp_df = orders_df[["Amount_Gross_Order", "city_name_fa"]]
    Q1 = temp_df[column_name].quantile(0.25)
    Q3 = temp_df[column_name].quantile(0.75)
    IQR = Q3 - Q1  # IQR is inter_quartile range.

    noise_filter = (temp_df[column_name] >= Q1 - 1.5 * IQR) & (temp_df[column_name] <= Q3 + 1.5 * IQR) | (
        temp_df[column_name].isna())
    temp_df = temp_df.loc[noise_filter]
    min_amount = temp_df.Amount_Gross_Order.min()
    max_amount = temp_df.Amount_Gross_Order.max()
    bins = list(np.arange(min_amount, max_amount, (max_amount - min_amount) / 4))
    temp_df.Amount_Gross_Order = pd.cut(temp_df.Amount_Gross_Order, bins, labels=["low", "mid", "high"])
    temp_df = temp_df.dropna()

    transaction_encoder = TransactionEncoder()
    encoded_df = transaction_encoder.fit_transform(temp_df.to_numpy())
    df = pd.DataFrame(encoded_df, columns=transaction_encoder.columns_)

    apriori_item_set = apriori(df, min_support=0.1, use_colnames=True)
    print(apriori_item_set)
