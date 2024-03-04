# import necessary libraries

# for work with data
import numpy as np
import pandas as pd
# for check rest time of cycles
from tqdm.auto import tqdm
# for work with pkl files
import pickle
# for work with csv files
import csv
# for check time consum by code running
import time
# for speed up download csv files
import pyarrow
# for work with files paths
import os
# for use internal data files
import importlib.resources


# function for reduce mem usage by pandas dataset
# we get it by change type of variables
# source of this func https://www.kaggle.com/code/ragnar123/very-fst-model/notebook

def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024 ** 2
    if verbose:

        if start_mem - end_mem > 0:
            print('Mem. usage decreased from {:5.2f} Mb to {:5.2f} Mb ({:.1f}% reduction)'.format(start_mem,
                                                                                                  end_mem, 100 * (
                                                                                                              start_mem - end_mem) / start_mem))

    return df

class Insta:

    def __init__(self):
        pass

    # function for load dataset

    def load_data(self, file_path: object) -> object:
        """
        loading dataset from file csv or pkl
        return pandas dataset.
        showing minimum info about dataset also reduce mem usage
        """
        if not os.path.isfile(file_path):
            print('file is not exist!')

            return

        print('loading file...')

        if file_path[-3:] == 'csv':

            df = pd.read_csv(file_path, engine='pyarrow')

        elif file_path[-3:] == 'pkl':

            df = pd.read_pickle(file_path)

        else:

            print('extention of this file is not support by this func!')

            return

        print('optimize file size...')
        df = reduce_mem_usage(df)

        print('rows of loaded dataset:', df.shape[0])
        print('columns of loaded dataset:', df.shape[1])
        print('----------------------------------------')
        print('quantity of NaN:')
        print(df.isna().sum())

        return df

    def load_sample(self):

        with importlib.resources.path('instarecsys', 'Data') as df_path:
            final_path = df_path / 'df_sample.csv'
            df_sample = pd.read_csv(final_path)
        return df_sample



    # main function, building features

    def feature(self, df, columns=['user_id', 'product_id', 'add_to_cart_order']):
        """
        in argument 'columns':
        first - user_id
        second - product_id
        third - add_to_cart_order
        in your dataset can be other names - but with same purpose
        """
        # take neccesary columns from loaded dataset
        df = df[columns]

        # make dataset with only user_id and product_id
        df_final = df[[columns[0], columns[1]]]

        # make second dataset with first features - quantity of purchases each product by each user
        print('build first features...')
        start_time = time.time()
        df_final = df_final.groupby([columns[0], columns[1]]).size().reset_index(name='counts')
        print('time consum:', time.time() - start_time)

        # sort both datasets by users and products
        print('sorting datasets...')
        df = df.sort_values(by=[columns[0], columns[1]], ascending=[True, True])
        df_final = df_final.sort_values(by=[columns[0], columns[1]], ascending=[True, True])

        # build second feature
        print('build second feature...')

        # add empty column for second feature
        df_final['most_freq'] = ""

        count = 0

        # in this cycle we build second feature and also check time consum with tqdm
        for user in tqdm(df['user_id'].unique()):

            ds_temp = df[df['user_id'] == user]

            for item in ds_temp['product_id'].unique():
                most_freq = ds_temp[ds_temp['product_id'] == item]['add_to_cart_order'].value_counts().idxmax()

                df_final.iloc[count, 3] = most_freq

                count += 1

        return df_final


    # function for recommendation

    def recommend(self, df, users=[], k=10, all_users=True):
        """
        this function give recommendation
        can be three cases:
         - for one user (users = [user],  all_users = False)
           in this case return dict where key is user and value is recommendations
         - set of users (give a list of users users = [user_1, user_2, ...]
           in this case return dataset with set of users
         - all users (all_users = True)
           in this case return csv file with recommendation for all users in loaded dataset
        k - quantity of recommended products
        """
        # sort dataset by user_id, count, most_freq
        df = df.sort_values(by=['user_id', 'counts', 'most_freq'], ascending=[True, False, True])

        # case for all users
        if all_users == True:

            # make a list of all users
            array_users = df['user_id'].unique()

            # create top_k products
            df_2 = df.groupby('user_id').head(k)
            top_k = df_2.groupby('user_id')['product_id'].apply(lambda x: x.to_numpy()).values

            # create zip for loading to the file
            rows = zip(array_users, top_k)

            # write zip to the file
            with open('submission.csv', "w") as f:
                writer = csv.writer(f)
                writer.writerow(['user_id', 'product_id'])
                for row in rows:
                    writer.writerow(row)

        # case for one user
        if len(users) == 1:
            dict_user[users[0]] = df[df['user_id'] == users[0]]['product_id'].head(k).values

            return dict_user

        # case for list of users
        if len(users) > 1:

            df_users = pd.DataFrame(columns=['user_id', 'product_id'])

            for user in users:
                df_temp = df[df['user_id'] == user][['user_id', 'product_id']].head(k)

                df_users = pd.concat([df_users, df_temp])

            return df_users