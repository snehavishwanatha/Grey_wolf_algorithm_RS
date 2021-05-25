#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 20:55:31 2019

@authors: Sneha V and Shrinidhi KR
"""

# Importing Required Libraries
import pandas as pd
import numpy  as np
import GWO
import pickle

# Function: Global Centering
def find_global_mean(Xdata_matrix):
    ''' Calculates the global mean of the matrix 
        For each column, Sum of column / Count in column
    '''
    Xdata = Xdata_matrix.copy(deep = True)
    
    ''' Mean of the entire dataset is calculated 
        Missing values are discarded when calculating the mean
    '''
    global_mean = sum(Xdata.sum()) / sum(Xdata.count()) 

    ''' Mean is subtracted from each value in the matrix
        Nan is filled with zero
    '''
    for i in range(0, Xdata.shape[0]):
        for j in range(0, Xdata.shape[1]):
            if (pd.isnull(Xdata.iloc[i, j])):
                Xdata.iloc[i, j] = 0.0
            elif (pd.isnull(Xdata.iloc[i, j]) == False):
                Xdata.iloc[i, j] = Xdata.iloc[i, j] - global_mean      
    return Xdata, global_mean


# Function: Weight Matrix for User
def generate_user_weight_matrix(Xdata, w_list_user):
    ''' Create a users * users matrix
        Assign weights to each index excluding diagonal (self to self)
    '''
    k = 0
    weight_matrix = pd.DataFrame(np.zeros((Xdata.shape[1], Xdata.shape[1])))
    for i in range(0, weight_matrix.shape[0]):
        for j in range(0, weight_matrix.shape[1]):           
            if (i == j):
                weight_matrix.iloc[i, j] = 0.0
            else:
                weight_matrix.iloc[i, j] = w_list_user[k]
                k = k + 1            
    return weight_matrix


# Function: Weight Matrix for Item
def generate_item_weight_matrix(Xdata, w_list_item):
    '''Create a items * items matrix
       Assign weights to each index excluding diagonal (self to self)
    '''
    
    k = 0
    weight_matrix = pd.DataFrame(np.zeros((Xdata.shape[0], Xdata.shape[0])))
    for i in range(0, weight_matrix.shape[0]):
        for j in range(0, weight_matrix.shape[1]):           
            if (i == j):
                weight_matrix.iloc[i, j] = 0.0
            else:
                weight_matrix.iloc[i, j] = w_list_item[k]
                k = k + 1            
    return weight_matrix


# Function: Ratings Prediction
def predict_ratings(original, weight_matrix_user, weight_matrix_item, global_mean, p_user_list, q_user_list, bias_user_list, bias_item_list):
    bias = original.copy(deep = True)
    prediction = original.copy(deep = True)
    for i in range(0, original.shape[0]):
        for j in range(0, original.shape[1]):
            bias.iloc[i, j] = bias_user_list[j] + bias_item_list[i]
            prediction.iloc[i, j] = 0
    for i in range(0, original.shape[0]):
        for j in range(0, original.shape[1]):
            for k in range(0,  weight_matrix_user.shape[1]):
                if ((not (pd.isnull(original.iloc[i, k]))) and (not (pd.isnull(original.iloc[i, j]))) and k != j):
                    prediction.iloc[i, j] = prediction.iloc[i, j] + weight_matrix_user.iloc[k,j]*(original.iloc[i, k]+ (-bias_user_list[k] - bias_item_list[i]))
            prediction.iloc[i, j] = prediction.iloc[i, j]/p_user_list[i]**(1/2)
    for i in range(0, original.shape[0]):
        for j in range(0, original.shape[1]):
            for k in range(0,  weight_matrix_item.shape[1]):
                if ((not (pd.isnull(original.iloc[k, j]))) and (not (pd.isnull(original.iloc[i, j]))) and k != i):
                    prediction.iloc[i, j] = prediction.iloc[i, j] + weight_matrix_item.iloc[k,i]*(original.iloc[k, j]+ (-bias_user_list[j] - bias_item_list[k]))
            prediction.iloc[i, j] = prediction.iloc[i, j]/q_user_list[j]**(1/2)
    for i in range(0, original.shape[0]):
        for j in range(0, original.shape[1]):
            prediction.iloc[i, j] = prediction.iloc[i, j] + bias.iloc[i, j] + global_mean
    return prediction


# Function: RSME calculator
def rmse_calculator(original, prediction):   
    mse = prediction.copy(deep = True)   
    for i in range (0, original.shape[0]):
        for j in range (0, original.shape[1]):
            if (not (pd.isnull(original.iloc[i, j]))):
                mse.iloc[i][j] = (original.iloc[i][j] - prediction.iloc[i][j])**2 
            else:
                mse.iloc[i][j] = 0
    rmse  = sum(mse.sum())/sum(mse.count())
    rmse = (rmse)**(1/2)    
    return rmse    


# Function: Separate Lists
def separate_lists(Xdata, variable_list):
    ''' Separating the variable list which contains random positions of the wolf as weights and bias
        for each user and each item
    '''
    
    w_list_user = [0]*(Xdata.shape[1]**2 - Xdata.shape[1])
    w_list_item = [0]*(Xdata.shape[0]**2 - Xdata.shape[0])
    bias_user_list = [0]*Xdata.shape[1]
    bias_item_list = [0]*Xdata.shape[0]
    
    r = len(w_list_user)
    s = r + len(w_list_item)
    t = s + len(bias_user_list)
    u = t + len(bias_item_list)
    
    # Weights set for users
    w_list_user = list(variable_list.iloc[0:r])
    
    # Weights set for items
    w_list_item = list(variable_list.iloc[r:s])
    
    # Bias set for users
    bias_user_list = list(variable_list.iloc[s:t]) 
    
    # Bias set for items
    bias_item_list = list(variable_list.iloc[t:u])

    return w_list_user, w_list_item, bias_user_list, bias_item_list


# Function: Loss Function
def loss_function(original, variable_list):
    Xdata, global_mean = find_global_mean(original)
    
    # Number of users who watched a movie
    p_user_list = original.count(axis = 1) 
    
    # Number of movies watched by a user 
    q_user_list = original.count(axis = 0) 
    
    # Obtaining lists that contain weights and bias for each item and each movie
    w_list_user, w_list_item, bias_user_list, bias_item_list = separate_lists(Xdata, variable_list)

    # Assign values from ww_list_user into a matrix 
    weight_matrix_user = generate_user_weight_matrix(Xdata, w_list_user)
    weight_matrix_item = generate_item_weight_matrix(Xdata, w_list_item)
    
    # Predict ratings 
    prediction = predict_ratings(original, weight_matrix_user, weight_matrix_item, global_mean, p_user_list, q_user_list, bias_user_list, bias_item_list)
    
    # Calculation of rmse
    rmse = rmse_calculator(original, prediction)
    return prediction, rmse


# Function: Model
def model(Xdata, pack_size = 25, iterations = 75):
    n = Xdata.shape[0]**2 + Xdata.shape[1]**2
    
    # GWO Objective Function
    def target_function (variables_values = [0]): 
        _ = loss_function(Xdata, variable_list = variables_values)
        return _[1]
    
    gwo = GWO.grey_wolf_optimizer(target_function = target_function, pack_size = pack_size, min_values = [-0.5]*n, max_values = [0.5]*n, iterations = iterations)
    ibm = loss_function(Xdata, variable_list = gwo[:-1])
    return ibm, gwo
   

if __name__ == '__main__':
    
    # Read the dataset
    df = pd.read_csv('/home/sneha/Documents/RSGWO/50x20movie_ratings.csv')
    print(df.head())
    
    # Discard first column as it contains movie names
    X = df.iloc[:,1:]
    
    # Set the first column as index of row names
    #X = X.set_index(df.iloc[:,0]) 
    
    file = open("model.pickle","wb")
    
    # Calling the model function
    u_i_bm = model(X, pack_size = 20, iterations = 100)
    
    pickle.dump((u_i_bm[0]), file)
    
    pickle_u_i_bm = pickle.load(open("model.pickle","rb"))
    file.close()
    print("Predictions of pickle", pickle_u_i_bm)
    #print("Predictions", u_i_bm[0][0])