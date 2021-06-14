import arabic_reshaper
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from bidi import algorithm as bidialg
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering


def inverse_convert_category(temp_df: pd.DataFrame, columns):
    for col in columns:
        my_dict = np.load("./datasets/converting_map_{}.npy".format(col), allow_pickle='TRUE').item()
        temp_df[col] = temp_df[col].map(my_dict)
    return temp_df


def plot_dendrogram(model, names, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_, counts]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, labels=names, **kwargs)


def one():
    temp_df = pd.read_csv("./datasets/divar_posts_dataset_cleaned.csv", low_memory=False)
    columns_to_cluster = ['cat1', 'cat2', 'cat3']
    temp_df = inverse_convert_category(temp_df, columns_to_cluster + ['city'])
    temp_df = temp_df[columns_to_cluster + ['city']]

    def my_count(inner_df: pd.DataFrame):
        inner_df = inner_df.reset_index(drop=True)
        return inner_df.drop(columns=['city'])[['cat1']].value_counts()

    temp_df = temp_df.groupby('city').apply(my_count)
    temp_df = temp_df.unstack(level=-1)
    temp_df = temp_df.fillna(0)

    model = AgglomerativeClustering(distance_threshold=0, n_clusters=None)
    model = model.fit(temp_df)
    plt.title('Hierarchical Clustering Dendrogram')
    plot_dendrogram(model, temp_df.index, truncate_mode='level', p=30)
    plt.xlabel("Number of points in node (or index of point if no parenthesis).")

    ax = plt.gca()
    xlbls = ax.get_xmajorticklabels()
    for lbl in xlbls:
        lbl.set_rotation(90)
    plt.tight_layout()
    plt.show()


def two():
    temp_df = pd.read_csv("./datasets/orders.csv")
    columns_to_cluster = ['ID_Item', 'Quantity_item']
    temp_df = temp_df[columns_to_cluster + ['city_name_fa']]

    def my_count(inner_df: pd.DataFrame):
        inner_df = inner_df.reset_index(drop=True)
        return inner_df.drop(columns=['city_name_fa']).groupby("ID_Item").sum()

    temp_df = temp_df.groupby('city_name_fa').apply(my_count)
    temp_df = temp_df.unstack(level=-1)
    temp_df = temp_df.fillna(0)
    labels = [bidialg.get_display(arabic_reshaper.reshape(x)) for x in temp_df.index]

    model = AgglomerativeClustering(distance_threshold=0, n_clusters=None)
    model = model.fit(temp_df)
    plt.title('Hierarchical Clustering Dendrogram')
    plot_dendrogram(model, labels, truncate_mode='level', p=10)
    plt.xlabel("Number of points in node (or index of point if no parenthesis).")

    ax = plt.gca()
    xlbls = ax.get_xmajorticklabels()
    for lbl in xlbls:
        lbl.set_rotation(90)
    plt.tight_layout()
    plt.show()
