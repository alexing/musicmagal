"""
This module contains a single class that is focused in making group 
recommendations. It should receive a utility matrix and train a model capable
of making group predictions.
"""
import pickle
import implicit
import pandas as pd
import numpy as np
from scipy import sparse


class GroupRecommender():
    def __init__(self, utility_matrix, dataset, is_pickled=True):
    """
    :utility_matrix: scipy sparse matrix with tracks as rows and users as 
    columns. Each entry shows how many times a track was listened by a user.
    :is_pickled: flag telling if the utility matrix is the object itself or if 
    it is the filepath to the pickled matrix.
    :dataset: the original last.fm dataset.
    
    This function initializes the group recommender object by loading the 
    utility matrix and training the ALS model with it.
    """
        if is_pickled:
            with open(utility_matrix, 'rb') as pickle_file:
                self.utility_matrix = pickle.loads(pickle_file.read())
        else:
            self.utility_matrix = utility_matrix
        self.dataset = dataset
        self.algo = implicit.als.AlternatingLeastSquares()
        self.algo.fit(self.utility_matrix.astype(np.double))
        self.num_of_tracks = self.utility_matrix.shape[0]
    
    
    def recommend(self, users, max_recommendations, method='naive'):
    """
    :users: the user indices in the utility matrix for which the 
    recommendations should be made.
    :max_recommendations: the max amount of recommendations to be made for the 
    group.
    :method: The group recommendation method that should be applied.
    
    :return: a list of the indices of the rows of the utility matrix for the 
    recommended tracks.
    """
        single_recommendations = []
        if method == 'naive':
            for user in users:
                recommendations = self.algo.recommend(user,
                                                      self.utility_matrix,
                                                      max_recommendations)
                single_recommendations.append([x[0] for x in recommendations])
                
            group_recommendations = set(single_recommendations[0])
            for recommendation in single_recommendations[1:]:
                group_recommendations = group_recommendations.intersection(
                    group_recommendations,
                    recommendation
                )
            group_recommendations = list(group_recommendations)
        elif method == 'mean':
            recommendation_vector = np.zeros()
        else:
            print("Not yet implemented!")
            group_recommendations = None
        
        return group_recommendations
    
    
    def full_recommendation(self, user_ids, max_recommendations, df, 
                            method='naive'):
        """
        :user_ids: the last.fm user ids from the original dataset for which we
        wish to make group recommendations.
        :max_recommendations: maximum number of tracks to recommend to the 
        group.
        :method: the group recommendation method that should be applied.
        
        :return: a list containing the artist and track names for the
        recommended tracks for the group.
        """
        users = np.where(np.in1d(self.dataset['user_id'].unique(), user_ids))[0]
        recommendations = self.recommend(users, max_recommendations, method)
        if recommendations:
            recommended_track_ids = self.dataset['track_id'].unique() \
                                    [recommendations]
            playlist = []
            for track in recommended_track_ids:
                playlist.append(
                    self.dataset[self.dataset['track_id'] == track] \
                    [['artist_name', 'track_name']].iloc[0, : ]
                )
        else:
            print("No songs found for this group.")
            playlist = None
        return playlist


    def evaluate(self, users_indexes, track_indexes):
        """
        Based on the evaluation method proposed in
        Collaborative Filtering for Implicit Feedback Datasets by Hu, Koren & Volinsky
        to use recall-oriented features.

        :return: rank
        Lower values of rank are more desirable, as they indicate ranking actually watched shows closer to the top of
        the rec- ommendation lists. Notice that for random predictions, the expected value of rankui is 50%
        (placing i in the middle of the sorted list).
        Thus, rank   50% indicates an algorithm no better than random.
        """
        length_recommendation = len(track_indexes)
        numerator = 0
        denominator = 0
        for recommendation_index, a_track in enumerate(track_indexes):
            for a_user in users_indexes:
                r_iu = self.utility_matrix[a_track, a_user]
                rank_iu = (recommendation_index / length_recommendation) * 100
                numerator = numerator + (rank_iu * r_iu)  # accumulator
                denominator = denominator + r_iu    # accumulator
        rank = numerator / denominator
        return rank
