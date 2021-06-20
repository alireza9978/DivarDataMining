import arabic_reshaper
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from bidi import algorithm as bidialg
from scipy.cluster.hierarchy import dendrogram
from sklearn import metrics
from sklearn.cluster import AgglomerativeClustering, KMeans, DBSCAN, MeanShift


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
    plt.savefig("./result/divar_city_clustering.jpg")
    plt.close()


def one_v2():
    temp_df = pd.read_csv("./datasets/divar_posts_dataset_cleaned.csv", low_memory=False)
    columns_to_cluster = ['cat1', 'cat2', 'cat3']
    temp_df = inverse_convert_category(temp_df, ['city'])
    temp_df = temp_df[columns_to_cluster + ['city']]

    def my_count(inner_df: pd.DataFrame):
        inner_df = inner_df.reset_index(drop=True)
        return inner_df.groupby(["cat1", "cat2", "cat3"]).count()

    temp_df = temp_df.groupby('city').apply(my_count)
    temp_df = temp_df.unstack(level=[1, 2, 3])
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
    plt.savefig("./result/divar_city_clustering_v2.jpg")
    plt.close()


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
    plt.savefig("./result/digikala_city_clustering.jpg")
    plt.close()


def four():
    temp_df = pd.read_csv("./datasets/divar_posts_dataset_cleaned.csv", low_memory=False)
    temp_df = temp_df[temp_df.price.notna()]

    models = [
        KMeans(n_clusters=2),
        KMeans(n_clusters=3),
        KMeans(n_clusters=4),
        KMeans(n_clusters=5),
        KMeans(n_clusters=6),
        KMeans(n_clusters=7),
        KMeans(n_clusters=8),
        KMeans(n_clusters=9),
    ]

    x_train = temp_df[['price']].to_numpy()

    best = None
    best_value = 0
    best_index = -1
    silhouette_scores = []
    for i, model in enumerate(models):
        y_prediction = model.fit_predict(x_train)
        y_set = set(y_prediction)
        if len(y_set) > 1:
            sil = metrics.silhouette_score(x_train, y_prediction, metric='sqeuclidean')
            silhouette_scores.append(sil)
            if best is None or sil > best_value:
                best = model
                best_value = sil
                best_index = i
        else:
            print("one cluster")

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.plot(range(2, len(silhouette_scores) + 2), silhouette_scores, '#7eb5fc', marker="o", linewidth=0.8)
    ax.plot([best_index + 2], [silhouette_scores[best_index]], 'r', marker="o", linewidth=2)
    ax.set_xticks(range(2, len(silhouette_scores) + 2))
    ax.set_xticklabels(["2", "3", "4", "5", "6", "7", "8", "9"])
    ax.set_ylabel('silhouette', fontsize=14)
    ax.set_xlabel('number of clusters', fontsize=14)
    plt.savefig("result/silhouette_scores.jpg")
    plt.close()


def four_v2():
    temp_df = pd.read_csv("./datasets/divar_posts_dataset_cleaned.csv", low_memory=False)
    temp_df = temp_df[temp_df.price.notna()]

    names = [
        "DBSCAN_1",
        "DBSCAN_2",
        "DBSCAN_3",
        "MeanShift",
    ]
    models = [
        DBSCAN(eps=0.1, min_samples=500),
        DBSCAN(eps=0.2, min_samples=500),
        DBSCAN(eps=0.1, min_samples=1000),
        MeanShift(bandwidth=0.2)
    ]

    x_train = temp_df[['price']].to_numpy()

    best = None
    best_value = 0
    best_index = -1
    silhouette_scores = []
    silhouette_scores_name = []
    for i, model in enumerate(models):
        y_prediction = model.fit_predict(x_train)
        y_set = set(y_prediction)
        if len(y_set) > 1:
            sil = metrics.silhouette_score(x_train, y_prediction, metric='sqeuclidean')
            silhouette_scores.append(sil)
            silhouette_scores_name.append(names[i] + "_{}".format(len(y_set)))
            if best is None or sil > best_value:
                best = model
                best_value = sil
                best_index = i
        else:
            print("one cluster for {}".format(names[i]))

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.plot(list(range(len(silhouette_scores))), silhouette_scores, '#7eb5fc', marker="o", linewidth=0.8)
    ax.plot([best_index], [silhouette_scores[best_index]], 'r', marker="o", linewidth=2)
    ax.set_xticks(list(range(len(silhouette_scores))))
    ax.set_xticklabels(silhouette_scores_name)
    ax.set_ylabel('silhouette', fontsize=14)
    ax.set_xlabel('models', fontsize=14)
    plt.savefig("result/silhouette_scores_v2.jpg")
    plt.close()


# one()
# one_v2()
# two()
# four()
four_v2()
