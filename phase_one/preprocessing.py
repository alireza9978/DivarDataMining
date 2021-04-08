import matplotlib.pyplot as plt
import pandas as pd
from dateutil.parser import parse


def fix_time(temp_df: pd.DataFrame):
    temp_df.created_at = temp_df.created_at.apply(lambda x: parse(x))
    temp_df['created_at_hour'] = temp_df.created_at.apply(lambda x: x.time().hour)
    temp_df['created_at_weekday'] = temp_df.created_at.apply(lambda x: (x.weekday() + 2) % 7)
    temp_df = temp_df.drop(columns=["created_at"])
    return temp_df


def categorize(temp_df: pd.DataFrame):
    def categorize_inner(inner_df: pd.DataFrame, name: str):
        inner_df[name] = inner_df[name].astype('category')
        inner_df[name] = inner_df[name].cat.codes

    columns = ['cat1', 'cat2', 'cat3', 'city']
    for column in columns:
        categorize_inner(temp_df, column)

    return temp_df


def reduction(temp_df: pd.DataFrame):
    temp_df = temp_df.drop(columns=['id'])
    return temp_df


def save_box_plot(temp_df: pd.DataFrame):
    columns = ['image_count', 'mileage']
    fig, axes = plt.subplots(nrows=1, ncols=2, )

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


def remove_outlier_box_plot(temp_df: pd.DataFrame, ):
    columns = ['image_count', 'price', 'mileage']
    output = pd.DataFrame()
    for column_name in columns:
        def remove_outliers(inner_df: pd.DataFrame):
            Q1 = inner_df[column_name].quantile(0.25)
            Q3 = inner_df[column_name].quantile(0.75)
            IQR = Q3 - Q1  # IQR is inter_quartile range.

            noise_filter = (inner_df[column_name] >= Q1 - 1.5 * IQR) & (inner_df[column_name] <= Q3 + 1.5 * IQR)
            return inner_df.loc[noise_filter, column_name]

        my_df = temp_df.groupby(['cat1', 'cat2', 'cat3']).apply(remove_outliers)
        my_df = my_df.reset_index(level=['cat1', 'cat2', 'cat3'], drop=True)
        output = pd.concat([output, my_df], axis=1)
    print(output.agg(['mean', 'median', 'max', 'min']))
    return output


data_set_path = 'datasets/divar_posts_dataset.csv'
reader = pd.read_csv(data_set_path, chunksize=10000, index_col=0)
df = next(reader)
df.price.where(~(df.price == -1), None, inplace=True)
save_box_plot(df)
df = remove_outlier_box_plot(df)
# save_box_plot(df)
# df = reduction(df)
# df = fix_time(df)
# df = categorize(df)
