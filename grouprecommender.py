import pickle
import implicit
import pandas as pd
import numpy as np
from scipy import sparse


class GroupRecommender():
    def __init__(self, utility_matrix, is_pickled=True):
        if is_pickled:
            with open(utility_matrix, 'rb') as pickle_file:
                self.utility_matrix = pickle.loads(pickle_file.read())
        else:
            self.utility_matrix = utility_matrix
        self.algo = implicit.als.AlternatingLeastSquares()
        self.algo.fit(self.utility_matrix.astype(np.double))
    
    
    def recommend(self, users, max_recommendations, method='naive'):
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
        else:
            print("Not yet implemented!")
            group_recommendations = None
        
        return group_recommendations
    
    
    def full_recommendation(self, user_ids, max_recommendations, df, 
                            method='naive'):
        
        users = np.where(np.in1d(df['user_id'].unique(), user_ids))[0]
        recommendations = self.recommend(users, max_recommendations, method)
        if recommendations:
            recommended_track_ids = df['track_id'].unique()[recommendations]
            playlist = []
            for track in recommended_track_ids:
                playlist.append(
                    df[df['track_id'] == track] \
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
