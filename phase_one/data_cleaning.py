import random

import matplotlib.pyplot as plt
import pandas as pd
from dateutil.parser import parse


def fix_time(temp_df: pd.DataFrame):
    temp_df.created_at = temp_df.created_at.apply(lambda x: parse(x))
    temp_df['created_at_hour'] = temp_df.created_at.apply(lambda x: x.time().hour)
    temp_df['created_at_weekday'] = temp_df.created_at.apply(lambda x: (x.weekday() + 2) % 7)
    temp_df = temp_df.drop(columns=["created_at"])
    return temp_df


def fix_year(temp_df: pd.DataFrame):
    temp_df.loc[temp_df.year == '<1366', 'year'] = temp_df.loc[temp_df.year == '<1366', 'year'].apply(
        lambda x: float(random.randint(1350, 1365)))
    temp_df.year = temp_df.year.astype('float')
    return temp_df


def categorize(temp_df: pd.DataFrame):
    def categorize_inner(inner_df: pd.DataFrame, name: str):
        inner_df[name] = inner_df[name].astype('category')
        inner_df[name] = inner_df[name].cat.codes

    columns = ['cat1', 'cat2', 'cat3', 'city', 'platform']
    for column in columns:
        categorize_inner(temp_df, column)

    return temp_df


def reduction(temp_df: pd.DataFrame):
    temp_df = temp_df.drop(columns=['id'])
    return temp_df


def save_box_plot(temp_df: pd.DataFrame):
    columns = ['image_count', 'mileage', 'price', 'year']
    temp_df[columns].agg(['mean', 'median', 'max', 'min', 'count']).to_csv("result/statistic.csv")

    fig, axes = plt.subplots(nrows=1, ncols=len(columns), )
    for i, column in enumerate(columns):
        # rectangular box plot
        box_plot = axes[i].boxplot(temp_df[column].dropna(),
                                   vert=True,  # vertical box alignment
                                   patch_artist=True,  # fill with color
                                   labels=[column])  # will be used to label x-ticks

        colors = ['pink', 'lightblue', 'lightgreen']
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
        axes[i].yaxis.grid(True)

    plt.tight_layout()
    plt.savefig('figures/box_plots.jpg')
    plt.close()


def remove_outlier_box_plot(temp_df: pd.DataFrame):
    column_name = 'price'

    def remove_outliers(inner_df: pd.DataFrame):
        fig, axes = plt.subplots(nrows=1, ncols=1, )
        # rectangular box plot
        box_plot = axes.boxplot(inner_df['price'].dropna(),
                                vert=True,  # vertical box alignment
                                patch_artist=True,  # fill with color
                                labels=['-->'.join(inner_df[['cat1', 'cat2', 'cat3']].head(1).values[
                                                       0])])  # will be used to label x-ticks

        colors = ['pink', 'lightblue', 'lightgreen']
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
        axes.yaxis.grid(True)
        plt.tight_layout()
        plt.savefig('figures/box_plots_{}.jpg'.format('-->'.join(inner_df[['cat1', 'cat2', 'cat3']].head(1).values[0])))
        plt.close()

        Q1 = inner_df[column_name].quantile(0.25)
        Q3 = inner_df[column_name].quantile(0.75)
        IQR = Q3 - Q1  # IQR is inter_quartile range.

        noise_filter = (inner_df[column_name] >= Q1 - 1.5 * IQR) & (inner_df[column_name] <= Q3 + 1.5 * IQR)
        return inner_df.loc[noise_filter, column_name]

    my_df = temp_df.groupby(['cat1', 'cat2', 'cat3']).apply(remove_outliers)
    my_df = my_df.reset_index(level=['cat1', 'cat2', 'cat3'], drop=False)
    return my_df


def remove_outlier_price(temp_df: pd.DataFrame):
    column_name = 'price'

    def remove_outliers(inner_df: pd.DataFrame):
        Q1 = inner_df[column_name].quantile(0.25)
        Q3 = inner_df[column_name].quantile(0.75)
        IQR = Q3 - Q1  # IQR is inter_quartile range.

        noise_filter = (inner_df[column_name] >= Q1 - 1.5 * IQR) & (inner_df[column_name] <= Q3 + 1.5 * IQR) | (
            inner_df[column_name].isna())
        inner_df = inner_df.loc[noise_filter]
        return inner_df

    return temp_df.groupby(['cat1', 'cat2', 'cat3']).apply(remove_outliers)


def remove_outlier(temp_df: pd.DataFrame):
    columns = ['image_count', 'mileage', 'year']
    for i, column_name in enumerate(columns):
        Q1 = temp_df[column_name].quantile(0.25)
        Q3 = temp_df[column_name].quantile(0.75)
        IQR = Q3 - Q1  # IQR is inter_quartile range.

        noise_filter = (temp_df[column_name] >= Q1 - 1.5 * IQR) & (temp_df[column_name] <= Q3 + 1.5 * IQR) | (
            temp_df[column_name].isna())
        temp_df = temp_df.loc[noise_filter]

    return temp_df


def save_null_description(temp_df: pd.DataFrame, state):
    (100 * temp_df.isnull().sum() / temp_df.shape[0]).to_csv('result/columns_null_percent_{}.csv'.format(state))


if __name__ == '__main__':
    data_set_path = 'datasets/divar_posts_dataset.csv'
    main_df = pd.read_csv(data_set_path, index_col=0)
    print("loaded", main_df.shape)
    main_df = fix_year(main_df)
    print("fix year done", main_df.shape)
    main_df = fix_time(main_df)
    print("fix time done", main_df.shape)
    main_df = reduction(main_df)
    print("reduction done", main_df.shape)
    # حذف ردیف‌ هایی که قیمت توافقی دارند
    main_df.price.where(~(main_df.price == -1), None, inplace=True)
    print("remove -1 price done", main_df.shape)

    main_df = remove_outlier_price(main_df)
    print("remove outlier price done", main_df.shape)
    main_df = remove_outlier(main_df)
    print("remove other outlier done", main_df.shape)
    main_df = categorize(main_df)
    print("categorize done", main_df.shape)
    main_df.to_csv("datasets/divar_posts_dataset_cleaned.csv")
