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

