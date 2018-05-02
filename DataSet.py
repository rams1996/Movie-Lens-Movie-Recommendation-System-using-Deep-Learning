# -*- Encoding:UTF-8 -*-

import numpy as np
import sys


class DataSet(object):
    def __init__(self, fileName):
        # data (list of tuples) => each tuple of the form -> (user, item, rating, timestamp)
        # shape => [users, items]
        self.data, self.shape = self.getData(fileName)
        
        # train, test (list of tuples) => each tuple of the form -> (user, item, rating)
        self.train, self.test = self.getTrainTest()
        
        # dictionary (keys: (user, item), values: ratings)
        self.trainDict = self.getTrainDict()

    def getData(self, fileName):
        
        # by default, the program loads data with 100K ratings
        # for loading dataset with 1M ratings, use the following command: 
        # python DMF.py -dataName ml-1m
        if fileName == 'ml-1m' or fileName == 'ml-100k':
            print("Loading {} data set.....".format(fileName))
            data = []
            filePath = ''
            if fileName == 'ml-1m':
                filePath = './Data/ml-1m/ratings.dat'
            elif fileName == 'ml-100k':
                filePath = './Data/ml-100k/u.data'
            u = 0
            i = 0
            maxr = 0.0
            with open(filePath, 'r') as f:
                for line in f:
                    if line:
                        lines = []
                        if fileName == 'ml-1m':
                            lines = line[:-1].split("::")
                        elif fileName == 'ml-100k':
                            lines = line[:-1].split("\t")
                        user = int(lines[0])
                        movie = int(lines[1])
                        score = float(lines[2])
                        time = int(lines[3])
                        data.append((user, movie, score, time))
                        if user > u:
                            u = user
                        if movie > i:
                            i = movie
                        if score > maxr:
                            maxr = score
            self.maxRate = maxr
            print("Loading Success!\n"
                  "Data Info:\n"
                  "\tUser Num: {}\n"
                  "\tItem Num: {}\n"
                  "\tData Size: {}".format(u, i, len(data)))
            return data, [u, i]
        else:
            print("Current data set is not support!")
            sys.exit()

    def getTrainTest(self):
        data = self.data
        # sort data according to user id and timestamp
        data = sorted(data, key=lambda x: (x[0], x[3]))
        
        # initialize empty train and test lists
        # list contains tuple => with each tuple of the form (user, item, rate)
        train = []
        test = []
        # iterate over all the ratings
        for i in range(len(data)-1):
            user = data[i][0]-1
            item = data[i][1]-1
            rate = data[i][2]
            # append 1 rating of each user in the test list
            if data[i][0] != data[i+1][0]:
                test.append((user, item, rate))
            else:
                train.append((user, item, rate))

        # append the last rating in the test set
        test.append((data[-1][0]-1, data[-1][1]-1, data[-1][2]))
        
        return train, test

    # create training dictionary with user and item as keys and ratings as value
    def getTrainDict(self):
        dataDict = {}
        for i in self.train:
            dataDict[(i[0], i[1])] = i[2]
        return dataDict

    # initilaize the embeddings
    def getEmbedding(self):
        # create empty train matrix of size [users, items]
        train_matrix = np.zeros([self.shape[0], self.shape[1]], dtype=np.float32)
        # iterate over the training matrix
        for i in self.train:
            user = i[0]
            movie = i[1]
            rating = i[2]
            train_matrix[user][movie] = rating
        # return the full training numpy array (will be a sparse matrix)
        return np.array(train_matrix)

    def getInstances(self, data, negNum):
        
        # Training data defined before start of each iteration
        # Y = (Y+) U (Y-)
        user = []
        item = []
        rate = []
        for i in data:
            user.append(i[0])
            item.append(i[1])
            rate.append(i[2])
            
            # negNum: 7 (by default)
            # so training instance for this (user, item) tuple includes negNum items (with zero ratings)
            for t in range(negNum):
                j = np.random.randint(self.shape[1])
                while (i[0], j) in self.trainDict:
                    j = np.random.randint(self.shape[1])
                    
                # append the (user, item) with zero rating
                user.append(i[0])
                item.append(j)
                rate.append(0.0)
        return np.array(user), np.array(item), np.array(rate)

    def getTestNeg(self, testData, negNum):
        # returns the test negative instances (user, item) which have rating 0
        # i.e., which are not in the self.data
        
        user = []
        item = []
                
        # in each iteration, we create a set of test negatives for each user
        for s in testData:
            tmp_user = []
            tmp_item = []
            
            # user_id
            u = s[0]
            # item_id (movieid)
            i = s[1]
            
            tmp_user.append(u)
            tmp_item.append(i)
            
            # in each iteration, create a set of (user, item) which is not in the training data
            for t in range(negNum):
                
                # random item_id generated
                j = np.random.randint(self.shape[1])
                
                # iterate till the time such an item (j) is found, which is not in the training 
                # data, for that particular user
                # (which means that it's corresponding (user, item(j)) rating = 0)
                while (u, j) in self.trainDict:
                    j = np.random.randint(self.shape[1])
                    
                tmp_user.append(u)
                tmp_item.append(j)
                
            user.append(tmp_user)
            
            # tmp_item (list) -> all the items which are not in the training data for that corresponding user 'u'
            # (means with rating = 0, for (tmp_user, tmp_item) tuple)
            item.append(tmp_item)
            
        return [np.array(user), np.array(item)]
