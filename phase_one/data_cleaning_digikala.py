import pandas as pd

# comments_df = pd.read_excel("./datasets/comment.xlsx")
# quality_df = pd.read_excel("./datasets/keifiat.xlsx")
products_df = pd.read_excel("./datasets/product.xlsx")
orders_df = pd.read_csv("./datasets/orders.csv")
# shopping_df = pd.read_csv("./datasets/tarikhche kharid.csv")


orders_df.groupby("ID_Order")["ID_Item"].apply(lambda x: x.to_list()).reset_index().to_csv(
    "./datasets/clean_orders.csv")
