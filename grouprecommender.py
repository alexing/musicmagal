"""
This module contains a single class that is focused in making group 
recommendations. It should receive a utility matrix and train a model capable
of making group predictions.
"""
import pickle
import implicit
import pandas as pd
import numpy as np
from scipy import sparse, spatial
from operator import itemgetter


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
        group_recommendations = []
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
            score_dict = {}
            for user in users:
                recommendation = self.algo.recommend(user,
                                                     self.utility_matrix,
                                                     self.num_of_tracks)
                for track, score in recommendation:
                    if track in score_dict.keys():
                        score_dict[track] += score
                    else:
                        score_dict[track] = score
                
            group_recommendations = sorted(score_dict.items(),
                                           key=itemgetter(1),
                                           reverse=True)[:max_recommendations]
            group_recommendations = [x[0] for x in group_recommendations]
            
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
        if np.array(recommendations).size > 0:
            recommended_track_ids = self.dataset['track_id'].unique() \
                                    [recommendations]
            playlist = []
            for track in recommended_track_ids:
                playlist.append(
                    self.dataset[self.dataset['track_id'] == track] \
                    [['artist_name', 'track_name', 'track_id']].iloc[0, : ]
                )
        else:
            print("No songs found for this group.")
            playlist = None
        return playlist

    
    def __cosine_sim__(self, user1, user2, alpha=1):
        """
        Computes the cosine similarity between two users, that is, how similar
        their tastes are.
        
        :user1: column index of the first user to be compared
        :user2: column index of the second user to be compared
        :alpha: scaling factor
        
        :return: the similarity between the users.
        """
        user1_array = self.utility_matrix[:, user1].toarray()
        user2_array = self.utility_matrix[:, user2].toarray()
        mean1 = user1_array.mean()
        mean2 = user2_array.mean()
        
        user1_array = np.array([0 if x == 0 else 1 for x in user1_array] + \
                               [mean1 * alpha])
        user2_array = np.array([0 if y == 0 else 1 for y in user2_array] + \
                               [mean2 * alpha])
        return 1 - spatial.distance.cosine(user1_array, user2_array)
    

    def avg_group_similarity(self, group_ids, alpha=1):
        """
        For each user in the group, this function will compute an array with 
        the similarities between them and each of the other users and then it
        computes the average similarity of the whole group.
        
        :group_ids: The user ids for which the average similarity is to be 
        computed.
        :alpha: Scaling factor
        
        :return: arrays of the similarities and the average similarity.
        """
        user_similarities = np.zeros(len(group_ids))
        for user1 in group_ids:
            curr_similarities = []
            for user2 in group_ids:
                similarity = self.__cosine_sim__(user1, user2)
                curr_similarities.append(similarity)
            user_similarities += np.array(curr_similarities)
        user_similarities /= len(group_ids)
        avg_similarity = np.mean(user_similarities)
        return user_similarities, avg_similarity
        
        
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
    
    
    def user_friendly_evaluation(self, user_indexes, track_indexes, top_n=10):
        """
        Shows if each track is in the user's top N recommendations and how many 
        of them are in top N's overall.
        
        :user_indexes: the users indexes for the users in the group
        :track_indexes: the indexes of the tracks recommended for the group
        :top_n: check if the recommendations are in the top N songs for each 
        user
        """
        n_users = len(user_indexes)
        n_tracks = len(track_indexes)
        tracks_in_top_n = []
        for track in track_indexes:
            curr_total = 0.0
            in_top_n = False
            for user in user_indexes:
                recs = self.algo.recommend(user,
                                           self.utility_matrix,
                                           top_n)
                rec_tracks = [x[0] for x in recs]
                if track in rec_tracks:
                    curr_total += 1
                    in_top_n = True
            percentage_of_users = 100 * (curr_total / n_users)
            print("Track " + str(track) + " is in " + str(percentage_of_users) \
                  + "% of the user's top " + str(top_n) + " recommendations.")
            tracks_in_top_n.append(in_top_n)
        percentage_of_tracks = 100 * (sum(tracks_in_top_n) / n_tracks)
        print("--------------------------")
        print(str(percentage_of_tracks) + "% of recommended tracks are in" + \
              " the users' top " + str(top_n) + ".")
        
    def get_songs(self):
        '''
        Returns a pandas series with track names and artist names.
        Mostly for integration with last.fm api
        :return:
        '''
        return self.dataset[['track_name','artist_name']]