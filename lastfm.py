from __future__ import (absolute_import, division, print_function, unicode_literals)

import os

from surprise import SVD
from surprise import Dataset
from surprise import Reader
from surprise import KNNBasic
from surprise import KNNWithMeans
from surprise import KNNWithZScore
from surprise.model_selection import KFold
from collections import defaultdict
from surprise.model_selection import cross_validate

import numpy as np
import pandas as pd

class Lastfm:
    def __init__(self):
        self.init = True

    def get_data(self):
        f = open('new_user_artists.txt', 'r')
        data = []
        for s in f:
            data.append(s.split('\t'))
        for i in range(len(data)):
            data[i][0] = int(data[i][0])
            data[i][1] = int(data[i][1])
            data[i][2] = np.log(int(data[i][2])+1)
        df = pd.DataFrame(data, columns=['userID', 'artistID', 'weight'])

        return df

    def get_max_weight(self, df):
        max_weight = df['weight'].max()

        return max_weight

    def get_scaled_data(self):
        df = self.get_data()
        max_weight = self.get_max_weight(df)

        for i in range(len(df)):
            df['weight'][i] = (df['weight'][i] / max_weight) * 5

        return df

    def get_user_item_matrix(self, df):
        matrix = df.pivot(index='userID', columns='artistID', values='weight')
        matrix.fillna(0, inplace=True)

        return matrix

    def COS(self, mat):
        mat = np.array(mat)
        NumUsers = np.size(mat, axis = 0)
        Sim = np.full((NumUsers, NumUsers), 0.0)
        print("Cos calculation start!")
        for u in range(0, NumUsers):
            arridx_u = np.where(mat[u,] == 0)
            for v in range(u+1, NumUsers):
                arridx_v = np.where(mat[v,] == 0)
                arridx = np.unique(np.concatenate((arridx_u, arridx_v), axis = None))
                
                U = np.delete(mat[u, ], arridx)
                V = np.delete(mat[v, ], arridx)
                
                if(np.linalg.norm(U) == 0 or np.linalg.norm(V) == 0):
                    Sim[u, v] = 0
                else:
                    Sim[u, v] = np.dot(U, V) / (np.linalg.norm(U) * np.linalg.norm(V))
                Sim[v, u] = Sim[u, v]
                
        print("COS calculation end!")
        
        return Sim

    def PCC(self, mat):
        mat = np.array(mat)
        NumUsers = np.size(mat, axis = 0)
        Sim = np.full((NumUsers, NumUsers), -1.0)
        
        mean = np.nanmean(np.where(mat!=0.0, mat, np.nan), axis=1)
        print("PCC calculation start!")
        for u in range(0, NumUsers):
            arridx_u = np.where(mat[u, ] == 0)
            for v in range(u+1, NumUsers):
                arridx_v = np.where(mat[v, ] == 0)
                arridx = np.unique(np.concatenate((arridx_u, arridx_v), axis=None))
                
                U = np.delete(mat[u, ], arridx) - mean[u]
                V = np.delete(mat[v, ], arridx) - mean[v]
                
                if(np.linalg.norm(U) == 0 or np.linalg.norm(V) == 0):
                    Sim[u, v] = 0
                else:
                    Sim[u, v] = np.dot(U, V) / (np.linalg.norm(U) * np.linalg.norm(V))
                
                Sim[v, u] = Sim[u, v]
                
        print("PCC calculation end!")
        
        return Sim

    def CPCC(self, a):
        a = np.array(a)
        NumUsers = a.shape[0]
        Sim = np.zeros((NumUsers, NumUsers))

        median = np.nanmedian(np.where(a!=0, a, np.nan), axis=1)
        print("cPCC calculation start!")
        for u in range(0, NumUsers):
            for v in range(u, NumUsers):
                arridx_u = np.where(a[u, ] == 0)
                arridx_v = np.where(a[v, ] == 0)
                arridx = np.concatenate((arridx_u, arridx_v), axis=None)

                U = np.delete(a[u, ], arridx)
                V = np.delete(a[v, ], arridx)
                U = U - median[u]
                V = V - median[v]

                InnerDot = np.dot(U, V)
                NormU = np.linalg.norm(U)
                NormV = np.linalg.norm(V)
                Sim[u, v] = InnerDot/(NormU * NormV)
                Sim[v, u] = Sim[u, v]
                Sim[np.isnan(Sim)] = -1
        print("cPCC calculation end!")
        
        return Sim

    def basic_CF(self, mat, sim, k):
        mat = np.array(mat)
        predicted_rating = np.zeros((mat.shape[0], mat.shape[1]))
        
        if(sim == 'COS'):
            Sim = self.COS(mat)
        elif(sim == 'PCC'):
            Sim = self.PCC(mat)
            
        k_neighbors = np.argsort(-Sim)
        k_neighbors = np.delete(k_neighbors, np.s_[k:], 1)
        
        NumUsers = np.size(mat, axis=0)
        
        for u in range(0, NumUsers):
            list_sim = Sim[u, k_neighbors[u,]]
            list_rating = mat[k_neighbors[u,]].astype('float64')
            
            predicted_rating[u, ] = np.sum(list_sim.reshape(-1, 1) * list_rating, axis=0) / np.sum(list_sim)
            
        return predicted_rating

    def basic_mean(self, mat, sim, k):
        mat = np.array(mat)
        predicted_rating = np.zeros((mat.shape[0], mat.shape[1]))
        
        mean = np.nanmean(np.where(mat != 0, mat, np.nan), axis=1)
        
        if(sim == 'COS'):
            Sim = self.COS(mat)
        elif(sim == 'PCC'):
            Sim = self.PCC(mat)
            
        k_neighbors = np.argsort(-Sim)
        k_neighbors = np.delete(k_neighbors, np.s_[k:], 1)
        
        NumUsers = np.size(mat, axis = 0)
        
        for u in range(0, NumUsers):
            list_sim = Sim[u, k_neighbors[u, ]]
            list_rating = mat[k_neighbors[u, ], ].astype('float64')
            list_mean = mean[k_neighbors[u, ], ]
            
            denominator = np.sum(list_sim)
            numerator = np.sum(list_sim.reshape(-1, 1) * (list_rating - list_mean.reshape(-1, 1)), axis=0)
            predicted_rating[u, ] = mean[u] + numerator / denominator
            
        return predicted_rating

    def basic_zscore(self, mat, sim, k):
        mat = np.array(mat)
        predicted_rating = np.zeros((mat.shape[0], mat.shape[1]))
        
        mean = np.nanmean(np.where(mat != 0, mat, np.nan), axis=1)
        std = np.nanstd(np.where(mat != 0, mat, np.nan), axis=1)
        
        if(sim == 'COS'):
            Sim = self.COS(mat)
        elif(sim == 'PCC'):
            Sim = self.PCC(mat)
            
        k_neighbors = np.argsort(-Sim)
        k_neighbors = np.delete(k_neighbors, np.s_[k:], 1)
        
        NumUsers = np.size(mat, axis=0)
        
        for u in range(0, NumUsers):
            list_sim = Sim[u, k_neighbors[u, ]]
            list_rating = mat[k_neighbors[u, ], ].astype('float64')
            list_mean = mean[k_neighbors[u, ], ]
            list_std = std[k_neighbors[u, ], ]
            
            denominator = np.sum(list_sim)
            numerator = np.sum(list_sim.reshape(-1, 1) * ((list_rating - list_mean.reshape(-1, 1)) / list_std.reshape(-1, 1)), axis=0)
            predicted_rating[u, ] = mean[u] + std[u] * numerator / denominator
            
        return predicted_rating

    def basic_baseline(self, mat, sim, k):
        mat = np.array(mat)
        predicted_rating = np.zeros((mat.shape[0], mat.shape[1]))
        
        mean = np.nanmean(np.where(mat != 0, mat, np.nan), axis=1)
        std = np.nanstd(np.where(mat != 0, mat, np.nan), axis=1)
        
        if(sim == 'COS'):
            Sim = self.COS(mat)
        elif(sim == 'PCC'):
            Sim = self.PCC(mat)
            
        k_neighbors = np.argsort(-Sim)
        k_neighbors = np.delete(k_neighbors, np.s_[k:], 1)
        
        NumUsers = np.size(mat, axis=0)
        
        for u in range(0, NumUsers):
            list_sim = Sim[u, k_neighbors[u, ]]
            list_rating = mat[k_neighbors[u, ], ].astype('float64')
            list_mean = mean[k_neighbors[u, ], ]
            list_std = std[k_neighbors[u, ], ]
            all_mean = np.mean(mat)
            
            denominator = np.sum(list_sim)
            numerator = np.sum(list_sim.reshape(-1, 1) * ((list_rating - list_mean.reshape(-1, 1))))
            predicted_rating[u, ] = (mean[u] - all_mean) + numerator / denominator
            
        return predicted_rating

    def load_file_by_surprise(self, file_name='new_user_artists_scaled.txt'):
        file_path = os.path.expanduser(file_name)
        reader = Reader(line_format='user item rating', sep='\t')
        data = Dataset.load_from_file(file_path, reader=reader)
        trainset = data.build_full_trainset()
        df = pd.DataFrame(data.raw_ratings, columns=['uid', 'iid', 'rate', 'timestamp'])

        return data, df
    
    def surprise_basicCF(self, df):
        sim_options = {'name': 'cosine', 'user_based': True}
        algo = KNNBasic()
        algo.fit(trainset)

        CF_pred_ratings = []

        for j in range(0, 1000):
            uid = str(df['uid'][j])
            iids = df[df.uid == uid]
            for i in range(1, len(iids)):
                iid = iids[i-1:i].iid.values[0]
                r_ui = iids[i-1:i].rate.values[0]
                pred = algo.predict(uid, iid, r_ui, verbose=True)
                CF_pred_ratings.append([pred.uid, pred.iid, pred.r_ui, pred.est])

        return CF_pred_ratings

    def surprise_CFwMean(self, df):
        sim_options = {'name': 'cosine', 'user_based': True}
        algo = KNNWithMeans(k = 40, min_k = 1, sim_options = sim_options)
        algo.fit(trainset)

        CFM_pred_ratings = []

        for j in range(0, 1000):
            uid = str(df['uid'][j])
            iids = df[df.uid == uid]
            for i in range(1, len(iids)):
                iid = iids[i-1:i].iid.values[0]
                r_ui = iids[i-1:i].rate.values[0]
                pred = algo.predict(uid, iid, r_ui, verbose=True)
                CFM_pred_ratings.append([pred.uid, pred.iid, pred.r_ui, pred.est])

        return CFM_pred_ratings

    def surprise_CFwZscore(self, df):
        sim_options = {'name': 'cosine', 'user_based': True}
        algo = KNNWithZScore(k = 40, min_k = 1, sim_options = sim_options)
        algo.fit(trainset)

        CFZ_pred_ratings = []

        for j in range(0, 1000):
            uid = str(df['uid'][j])
            iids = df[df.uid == uid]
            for i in range(1, len(iids)):
                iid = iids[i-1:i].iid.values[0]
                r_ui = iids[i-1:i].rate.values[0]
                pred = algo.predict(uid, iid, r_ui, verbose=True)
                CFZ_pred_ratings.append([pred.uid, pred.iid, pred.r_ui, pred.est])

        return CFZ_pred_ratings

    def surprise_SVD(self, df):
        algo = SVD(n_factors = 100, n_epochs = 20, biased = False, lr_all = 0.005, reg_all = 0)
        algo.fit(trainset)

        SVD_pred_ratings = []

        for j in range(0, 1000):
            uid = str(df['uid'][j])
            iids = df[df.uid == uid]
            for i in range(1, len(iids)):
                iid = iids[i-1:i].iid.values[0]
                r_ui = iids[i-1:i].rate.values[0]
                pred = algo.predict(uid, iid, r_ui, verbose=True)
                SVD_pred_ratings.append([pred.uid, pred.iid, pred.r_ui, pred.est])

        return SVD_pred_ratings

    def precision_recall_at_k(self, predictions, k=10, threshold=3.5):
        '''Return precision and recall at k metrics for each user.'''

        # First map the predictions to each user.
        user_est_true = dict()
        for uid, _, true_r, est, _ in predictions:
            if user_est_true.get(uid) == None:
                user_est_true[uid] = []
                user_est_true[uid].append((est, true_r))
            else:
                user_est_true[uid].append((est, true_r))

        precisions = dict() # 정확도
        recalls = dict() # 검출율
        
        for uid, user_ratings in user_est_true.items():

            # Sort user ratings by estimated value
            user_ratings.sort(key=lambda x: x[0], reverse=True)

            # Number of relevant items
            n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)

            # Number of recommended items in top k
            n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])

            # Number of relevant and recommended items in top k
            n_rel_and_rec_k = sum(((true_r >= threshold) and (est >= threshold))
                                for (est, true_r) in user_ratings[:k])

            # Precision@K: Proportion of recommended items that are relevant
            precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 1

            # Recall@K: Proportion of relevant items that are recommended
            recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 1

        return precisions, recalls

    def kfold_score(self):
        data, _ = self.load_file_by_surprise()
        kf = KFold(n_splits=5)

        algo_list = [KNNBasic(), KNNWithMeans(), KNNWithZScore(), SVD()]
        algo_names = ['KNNBasic', 'KNNWithMeans', 'KNNWithZScore', 'SVD']

        for trainset, testset in kf.split(data):
            for algo, algo_name in zip(algo_list, algo_names):
                
                algo.fit(trainset)
                predictions = algo.test(testset)
                precisions, recalls = self.precision_recall_at_k(predictions, k=5, threshold=4)

                P = sum(prec for prec in precisions.values()) / len(precisions)
                R = sum(rec for rec in recalls.values()) / len(recalls)
                F1 = 2 * P * R / (P + R)
                
                print("*****%s*****" %(algo_name))
                print("precision : ", P)
                print("recall : ", R)
                print("F1 : ", F1)
                print("")

    def ndcg(self, r_ui_list, pred_list):
        dcg = float(pred_list[0][2])
        for idx in range(1, len(r_ui_list)):
            
            dcg += (float(pred_list[idx][2]) / np.log2(idx+1))
        
        idcg = float(r_ui_list[0][2])
        for idx in range(1, len(r_ui_list)):
            idcg += (float(r_ui_list[idx][2]) / np.log2(idx+1))
            
        ndcg = dcg/idcg
        return ndcg

    def calculate_ndcg(self, ratings):
        ratings = np.array(ratings)
        user_id = np.unique(ratings[:, 0])

        ndcg_list = []
        for user in user_id:
            cnt = 0
            user_pred_list = []
            for i in range(len(ratings)):
                if ratings[i][0] == user:
                    user_pred_list.append(ratings[i])
                    cnt += 1
                    if cnt > 10:
                        break
                        
            r_ui_list = sorted(user_pred_list, key=lambda x : x[2], reverse=True)
            pred_list = sorted(user_pred_list, key=lambda x : x[3], reverse=True)
            
            ndcg = calculate_ndcg(r_ui_list, pred_list)
            ndcg_list.append(ndcg)

        print("NDCG calculation ended!")
        print(sum(ndcg_list) / len(ndcg_list))
        
        return sum(ndcg_list) / len(ndcg_list)

    