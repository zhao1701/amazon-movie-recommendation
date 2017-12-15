import pandas as pd
import numpy as np
import sys
import os
import pickle as pkl
import time
import warnings

from collections import defaultdict
from scipy.sparse import csr_matrix, csc_matrix, hstack, vstack

class CosineLSH():
    def __init__(self, p = 5, q = 4, user = 'user', item = 'item', rating = 'rating', seed = None):
        
        # store column names for user, item, and rating data
        self.user = user
        self.item = item
        self.rating = rating
        
        # p - hash signature length, q - number of signatures
        self.p = p
        self.q = q
        
        # user-item ratings matrix
        self.data_matrix = None
        
        # list of unique items in the training set
        self.item_list = None
        self.hyperplanes = None
        
        # basic statistics
        self.n_users = None
        self.n_items = None
        
        # list of signature dictionaries, each dictionary storing all item names with the same signature
        self.dicts_list = list()
        
        # average rating of training data that is returned as prediction when CF prediction not possible
        self.avg_rating = None
        
        if isinstance(seed, int):
            np.random.seed(seed)
    
    # convert a series of characters into a single string
    def char_series_to_string(self, series):
        string = ''
        for char in series:
            string += char
        return string
    
    def fit(self, data):
        # convert data into vector form
        df = data.pivot_table(index=self.user, columns=self.item)
        self.data_matrix = df.fillna(0)
        self.data_matrix.columns = self.data_matrix.columns.droplevel()
        self.item_list = df.columns.levels[-1]
        self.n_users, self.n_items = self.data_matrix.shape
        
        # create set of random hyperplanes
        hyperplanes = np.random.randn(self.p * self.q, self.n_users)
        hyperplanes_pos = (hyperplanes >= 0).astype(int)
        hyperplanes_neg = (hyperplanes < 0) * -1
        self.hyperplanes = hyperplanes_pos + hyperplanes_neg
        assert((self.hyperplanes >= 0).sum() + (self.hyperplanes < 0).sum() == self.hyperplanes.size)
        
        # build signatures and bands
        signature_matrix = np.dot(self.hyperplanes, self.data_matrix)
        signature_matrix = (signature_matrix >= 0).astype(int)
        self.signature_matrix = pd.DataFrame(signature_matrix, columns = self.item_list)
        
        # for each band, store neighbors in same bucket
        for i in range(self.q):
            band_start = i * self.p
            band_end = band_start + self.p
            signature_df = self.signature_matrix.iloc[band_start:band_end, :].T.astype(str)
            signature_df = pd.DataFrame(signature_df.apply(self.char_series_to_string, axis = 1))
            signature_dict = signature_df.groupby(0)
            self.dicts_list.append(signature_dict)
            
        # calculate average rating
        self.avg_rating = data[self.rating].mean()
            
    def get_neighbors(self, item):
        
        # generate signature string
        item_vector = self.data_matrix.loc[:,item]
        signatures = np.dot(self.hyperplanes, item_vector)
        signatures = (signatures >= 0).astype(int).astype(str)
        signatures = ''.join(list(signatures))
        
        # collect neighbors from each band
        neighbors = set()
        for i in range(self.q):
            band_start = i * self.p
            band_end = band_start + self.p
            signature = signatures[band_start:band_end]
            band_neighbors = self.dicts_list[i].get_group(signature).index
            neighbors = neighbors.union(list(band_neighbors))
            
        neighbors = neighbors - set([item])     
        return neighbors
    
    # given a user and item, return all item-neighbors that the user has rated
    def get_relevant_items(self, user, item):
        user_vector = self.data_matrix.loc[user, :]
        user_vector = user_vector[user_vector > 0]
        user_items = set(user_vector.index) - set([item])
        item_neighbors = self.get_neighbors(item)
        relevant_items = user_items.intersection(item_neighbors)
        
        # if 5 or fewer relevant items found, return all items user has rated
        if len(relevant_items) <= 5:
            relevant_items = user_items
        
        return relevant_items
    
    # calculate all cosine similarities between a target item and its neighbors
    def cosine_similarities(self, target_item, relevant_items):
        target_item_vector = self.data_matrix.loc[:, target_item].values
        relevant_items_matrix = self.data_matrix.loc[:, relevant_items].values
        dot_products = np.dot(target_item_vector, relevant_items_matrix)
        norm_products = np.linalg.norm(relevant_items_matrix, axis = 0) * np.linalg.norm(target_item_vector)
        similarities = dot_products / norm_products
        return similarities
    
    # predict a rating for a user-item pair
    def predict_rating(self, user, item):
        relevant_items = list(self.get_relevant_items(user, item))
        user_ratings = self.data_matrix.loc[user,:][relevant_items]
        item_similarities = self.cosine_similarities(item, relevant_items)
        
        # if an item is new and prediction cannot be made, return average rating from training set
        if len(item_similarities) == 0 or item_similarities.sum() == 0:
            prediction = self.avg_rating
        else:
            prediction = np.dot(item_similarities, user_ratings) / item_similarities.sum()

        return prediction
       
    # predict ratings for all input data
    def predict(self, data):
        df = data[[self.user, self.item]]
        
        preds_list = list()
        print('=' * 100)
        for i in range(len(df)):
            if i % int(len(df) / 100) == 0:
                print('-', end='')
            user = df.iloc[i,0]
            item = df.iloc[i,1]
            preds_list.append(self.predict_rating(user, item))
            
        return np.array(preds_list)