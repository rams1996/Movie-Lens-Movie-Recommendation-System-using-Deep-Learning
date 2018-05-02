# -*- Encoding:UTF-8 -*-

import tensorflow as tf
import numpy as np
import argparse
from DataSet import DataSet
import sys
import os
import heapq
import math


def main():
    
    # The ArgumentParser object will hold all the information necessary to parse the 
    # command line into Python data types.
    parser = argparse.ArgumentParser(description="Options")

    # initalize default variables used in the model
    # like lr, maxEpochs, topK, batchSize, etc.
    
    # Now, Define how a each single command-line argument should be parsed.
    
    # Firstly, adding default value of the dataName to be 'ml-100k'
    # for command line argument = -dataname  
    parser.add_argument('-dataName', action='store', dest='dataName', default='ml-100k')
    
    # How many items -> negative items (with rating 0) samples to be created
    # for command line argument = -negNum
    parser.add_argument('-negNum', action='store', dest='negNum', default=7, type=int)
    
    # define the number of first and the hidden layers and their dimensions
    # (for both user and item)
    parser.add_argument('-userLayer', action='store', dest='userLayer', default=[512, 64])
    parser.add_argument('-itemLayer', action='store', dest='itemLayer', default=[1024, 64])
    
    # add learning rate for the optimization function
    # for command line argument = -lr
    parser.add_argument('-lr', action='store', dest='lr', default=0.0001)
    
    # number of maximum training iterations
    # for command line argument = -maxEpochs
    parser.add_argument('-maxEpochs', action='store', dest='maxEpochs', default=50, type=int)
    
    # number of training examples in one forward/backward pass
    # higher the batchsize, more the memory you need 
    # for command line argument = -batchSize
    parser.add_argument('-batchSize', action='store', dest='batchSize', default=256, type=int)
    
    # used for stoping the training of the model, 
    # if since certain epochs/iterations, the model's performance (ndcg, hr metrics) have not improved.
    # for command line argument = -earlyStop
    parser.add_argument('-earlyStop', action='store', dest='earlyStop', default=5)
    
    # the directory made during training, in which the training data gets saved
    # for command line argument = -checkPoint
    parser.add_argument('-checkPoint', action='store', dest='checkPoint', default='./check_Point/')
    
    # for top K number of recommendations (or predictions, after training the model) to be considered,
    # used in the evaluation of the model: for calculating NDCG@10, HR@10 metrics 
    # for command line argument = -topK
    parser.add_argument('-topK', action='store', dest='topK', default=10)

    args = parser.parse_args()

    # create the DMF object
    classifier = DMF(args)

    # run the created DMF
    classifier.run()


class DMF:
    def __init__(self, args):
        # initialize the (object) variables
        self.dataName = args.dataName
        
        # data exploration
        # create the dataset
        # data created => dataSet.shape, dataSet.data, dataSet.train, dataSet.test, dataSet.trainDict,
        #                 dataSet.maxRate (in getData() in DataSet.py)
        self.dataSet = DataSet(self.dataName)
        
        # shape => [users, items] total users and items in the dataset
        self.shape = self.dataSet.shape
        
        # maxRate => maximum ratings given by any user in the whole dataset
        # this is used for normalization of the data in the loss (binary cross entropy) function
        self.maxRate = self.dataSet.maxRate

        # create train and test data, test negatives (sample of the test data itself)
        self.train = self.dataSet.train
        self.test = self.dataSet.test
        
        # negNum(default = 7), used in creating training instances
        self.negNum = args.negNum
        
        # zero ratings are considered to be negative from the implicit data point of view
        # create testNegatives: set of negative instances, which can be all (or sampled from) zero ratings
        self.testNeg = self.dataSet.getTestNeg(self.test, 99)
        
        # create embeddings used for creating user and item latent vectors
        self.add_embedding_matrix()

        # specify the inputs, (to feed actual training examples)
        self.add_placeholders()

        # store the userLayer and the itemlayer arguments in the DMF's object itself
        # to be used during the model creation
        self.userLayer = args.userLayer
        self.itemLayer = args.itemLayer
        
        # create model
        self.add_model()

        # add the normalized cross entropy loss function
        self.add_loss()

        # learning rate
        self.lr = args.lr
        
        # add optimization and loss (already created above) function
        self.add_train_step()

        # store the checkPoint directory value in the model's object 
        self.checkPoint = args.checkPoint
        
        # initialize tensorflow session
        self.init_sess()

        # max iterations for training, default: 50
        self.maxEpochs = args.maxEpochs
        # batch size for training, default: 256
        self.batchSize = args.batchSize

        # metrics calculated over top K recommendations (predictions)
        # topK (default 10) => to calculate ndcg@10, hr@10
        self.topK = args.topK
        
        # default: 5
        self.earlyStop = args.earlyStop


    def add_placeholders(self):
        
        # tf.placeholder is used to feed actual training examples.
        # (dummy nodes that provide entry points for data to computational graph).
        
        self.user = tf.placeholder(tf.int32)
        self.item = tf.placeholder(tf.int32)
        self.rate = tf.placeholder(tf.float32)
        self.drop = tf.placeholder(tf.float32)

    def add_embedding_matrix(self):
        
        # training numpy array => (rows: user_id), (columns: itemId)
        # Convert (sparse) training numpy array to a Tensor object
        self.user_item_embedding = tf.convert_to_tensor(self.dataSet.getEmbedding())
        
        # take a transpose of the above converted tensor object
        # shape-> (row: item_id), (column: user_id)
        self.item_user_embedding = tf.transpose(self.user_item_embedding)

    def add_model(self):
        
        # tf.nn.embedding_lookup retrieves rows of the 'user_item_embedding' tensor.
        # Example: 
        # params = tf.constant([10,20,30,40])
        # ids = tf.constant([0,1,2,3])
        # print tf.nn.embedding_lookup(params,ids).eval()
        # would return [10, 20, 30, 40]
        
        # user_input -> shape: users, items
        user_input = tf.nn.embedding_lookup(self.user_item_embedding, self.user)
        
        # item_input -> shape: items, users
        item_input = tf.nn.embedding_lookup(self.item_user_embedding, self.item)

        def init_variable(shape, name):
            # use tf.Variable for trainable variables such as weights (W) and biases (B) for the model.
            return tf.Variable(tf.truncated_normal(shape=shape, dtype=tf.float32, stddev=0.01), name=name)

        with tf.name_scope("User_Layer"):
            # generate user weight (shape: [items, 512])
            user_W1 = init_variable([self.shape[1], self.userLayer[0]], "user_W1")
            
            # multiply user_weights and the user_input, (user_input * user_W1)
            # output at first layer: l(u) = Y(u) * W(u), acc. to the paper
            # user_out: (shape: [users, 512])
            user_out = tf.matmul(user_input, user_W1)
            
            # by default we have 1 hidden user layer, as default userLayer = [512, 64]
            for i in range(0, len(self.userLayer)-1):
                # again, generate user weights, (shape: [512, 64])
                W = init_variable([self.userLayer[i], self.userLayer[i+1]], "user_W"+str(i+2))
                
                # in the computation at hidden layers, also generate and then add user biases
                # shape: [64]
                b = init_variable([self.userLayer[i+1]], "user_b"+str(i+2))
                
                # output of the hidden layer: (user_out (of the previous layer) * Weights) + biases)
                # l(u'): (l(u) * W(u)) + b(u) 
                # shape: [users, 64]
                # apply the 'relu' activation function on the l(u')
                user_out = tf.nn.relu(tf.add(tf.matmul(user_out, W), b))

        with tf.name_scope("Item_Layer"):
            # generate item weight (shape: [users, 1024])
            item_W1 = init_variable([self.shape[0], self.itemLayer[0]], "item_W1")
            
            # multiply item_weights and the item_input, (item_input * item_W1)
            # output at first layer: l(i) = Y(i) * W(i), acc. to the paper
            # item_out: (shape: [items, 1024])
            item_out = tf.matmul(item_input, item_W1)
            
            # by default we have 1 hidden item layer, as default itemLayer = [1024, 64]
            for i in range(0, len(self.itemLayer)-1):
                # again, generate item weights, (shape: [1024, 64])
                W = init_variable([self.itemLayer[i], self.itemLayer[i+1]], "item_W"+str(i+2))
                
                # in the computation at hidden layers, also generate and then add item biases
                # shape: [64]
                b = init_variable([self.itemLayer[i+1]], "item_b"+str(i+2))
                
                # output of the hidden layer: (item_out (of the previous layer) * Weights) + biases)
                # l(i') = (l(i) * W(i)) + b(i) 
                # shape: [users, 64]
                # apply the 'relu' activation function on the l(i')
                item_out = tf.nn.relu(tf.add(tf.matmul(item_out, W), b))

        # calculate the length of (or normalize) the user and item latent vectors (user_out, item_out)
        norm_user_output = tf.sqrt(tf.reduce_sum(tf.square(user_out), axis=1))
        norm_item_output = tf.sqrt(tf.reduce_sum(tf.square(item_out), axis=1))
        
        # calculate cosine similarity of the latent vectors (user_out, item_out)
        self.y_ = tf.reduce_sum(tf.multiply(user_out, item_out), axis=1, keep_dims=False) / (norm_item_output * norm_user_output)
        
        # For cross entropy loss, because the predicted score of Yij can be negative, 
        # we use the below equation to transform the original predictions self.y_ (calculated above)
        self.y_ = tf.maximum(1e-6, self.y_)

    def add_loss(self):
        # normalized cross entropy loss fucntion
        # to incorporate the explicit ratings into cross entropy loss, so that explicit and implicit
        # information can be used together for optimization
        regRate = self.rate / self.maxRate
        losses = regRate * tf.log(self.y_) + (1 - regRate) * tf.log(1 - self.y_)
        loss = -tf.reduce_sum(losses)
        # regLoss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
        # self.loss = loss + self.reg * regLoss
        self.loss = loss

    def add_train_step(self):
        '''
        global_step = tf.Variable(0, name='global_step', trainable=False)
        self.lr = tf.train.exponential_decay(self.lr, global_step,
                                             self.decay_steps, self.decay_rate, staircase=True)
        '''
        # lr = 0.0001
        # create an optimizer
        optimizer = tf.train.AdamOptimizer(self.lr)
        
        # add optimization function and the loss function
        self.train_step = optimizer.minimize(self.loss)

    def init_sess(self):
        self.config = tf.ConfigProto()
        
        # option "allow_growth" used below: 
        # attempts to allocate only as much GPU memory based on runtime allocations:
        # it starts out allocating very little memory, and as Sessions get run and
        # more GPU memory is needed, we extend the GPU memory region needed by the TensorFlow process. 
        self.config.gpu_options.allow_growth = True
        
        # If you would like TensorFlow to automatically choose an existing and supported device to
        # run the operations in case the specified one doesn't exist, you can set allow_soft_placement
        # to True in the configuration option when creating the session.
        self.config.allow_soft_placement = True
        
        # A class for running TensorFlow operations.
        # A Session object encapsulates the environment in which Operation objects are executed,
        # and Tensor objects are evaluated.
        self.sess = tf.Session(config=self.config)
        
        # tf.global_variables_initializer is a shortcut to initialize all global variables
        # sess.run(): Runs operations and evaluates tensors
        # (begin the session and allocate memory to store the current value of the model variables)
        self.sess.run(tf.global_variables_initializer())

        # Saves (and restores) model variables.
        self.saver = tf.train.Saver()
        
        if os.path.exists(self.checkPoint):
            [os.remove('./checkPoint/{}'.format(f)) for f in os.listdir(self.checkPoint)]
        else:
            os.mkdir(self.checkPoint)

    def run(self):
        # initialize the performance metrics
        best_hr = -1
        best_NDCG = -1
        
        # iteration over which the best performance of the model is observed
        best_epoch = -1
        
        print("Start Training!")
        
        # iterate over each epoch/iteration for training the model
        for epoch in range(self.maxEpochs):
            print("="*20+"Epoch ", epoch, "="*20)
            
            # run the iteration (train the model)
            self.run_epoch(self.sess)
            
            print('='*50)
            
            # start evaluating the model
            print("Start Evaluation!")
            
            # calculate the metrics HR (Hit Ratio), NDCG (Normalized Discounted Cumulative Gain)
            hr, NDCG = self.evaluate(self.sess, self.topK)
            print("Epoch ", epoch, "HR: {}, NDCG: {}".format(hr, NDCG))
            
            # update the best_hr, best_hr if better model performance is recorded
            if hr > best_hr or NDCG > best_NDCG:
                best_hr = hr
                best_NDCG = NDCG
                best_epoch = epoch
                
                # save the model variables
                self.saver.save(self.sess, self.checkPoint)
                
            # stop the model, only if
            # the metrics (ndcg, hr) are not improving over some iterations (defined in earlyStop)
            if epoch - best_epoch > self.earlyStop:
                print("Normal Early stop!")
                break
            print("="*20+"Epoch ", epoch, "End"+"="*20)
            
        # print the best_hr and the best_ndcg metrics value
        print("Best hr: {}, NDCG: {}, At Epoch {}".format(best_hr, best_NDCG, best_epoch))
        
        # training completion
        print("Training complete!")

    def run_epoch(self, sess, verbose=10):
        
        # fetch the training instances (to be fed into the model)
        train_u, train_i, train_r = self.dataSet.getInstances(self.train, self.negNum)
        
        # number of training instances
        train_len = len(train_u)
        print('\n# Training Instances: {}\n'.format(train_len))
        
        # np.arange(): generates a sequence from [0, train_len - 1]
        # Randomly permute a sequence of indexes from 0 to (train_len - 1)
        index_shuffled = np.random.permutation(np.arange(train_len))
        
        # shuffled training data (according to permuted indexes above)
        train_u = train_u[index_shuffled]
        train_i = train_i[index_shuffled]
        train_r = train_r[index_shuffled]

        # As the data is to be trained in batches, so the data has to be divided accordingly, 
        # such that, it can be fed into the model.
        # And as the batchSize (default :256) is defined above, we can calculate the number
        # of batch iterations required to include all the training instances for training the model. 
        
        # number of batch iterations required for training over all the training instances.
        # (+ 1) => to include a few left training instances at the end.
        number_of_batches = (len(train_u) // self.batchSize) + 1

        # initilaize the list for keeping a record of loss during training of each bacth.
        losses = []
        
        # iterate over all the batch iterations (calculated above 'num_batches')
        for i in range(number_of_batches):
            
            # min_idx: starting point of the training instance to be included
            # in each batch (to be trained during each iteration).
            min_index = i * self.batchSize
            
            # max_idx: last training instance in the training batch.
            
            # for the last index value, when 'i' == (num_batches - 1),
            # (i+1)*self.batchSize becomes > train_len. Hence, the max_idx = train_len
            # So, for the last few left training instances,
            # the range of the taining instances to be included becomes:
            # [(num_batches - 1) * batch_size , train_len) 
            max_index = np.min([train_len, (i + 1) * self.batchSize])
            
            # form/create the training batch data (acc. to the range calculated above)
            train_u_batch = train_u[min_index: max_index]
            train_i_batch = train_i[min_index: max_index]
            train_r_batch = train_r[min_index: max_index]

            # create a dictionary object with keys: user, item, rate
            feed_dict = self.create_feed_dict(train_u_batch, train_i_batch, train_r_batch)
            
            # run computations and evaluate tensors in the session
            # self.train_step, self.loss: optimization and loss functions
            # feed_dict: input to the model
            _, tmp_loss = sess.run([self.train_step, self.loss], feed_dict=feed_dict)
            
            # append the loss calculated during each batch iteration in the list
            losses.append(tmp_loss)
            
            # print the loss after 10 batch iterations
            if verbose and i % verbose == 0:
                sys.stdout.write('\r{} / {} : loss = {}'.format(
                    i, number_of_batches, np.mean(losses[-verbose:])
                ))
                sys.stdout.flush()
                
        # total loss (mean of all the batch iteration losses)
        loss = np.mean(losses)
        
        print("\nMean loss in this epoch is: {}".format(loss))
        return loss

    # preparing the input training data for the model
    def create_feed_dict(self, u, i, r=None, drop=None):
        return {self.user: u,
                self.item: i,
                self.rate: r,
                self.drop: drop}

    # evaluation metrics (NDCG, HR)
    def evaluate(self, sess, topK):
                
        # if the predicted item matches the target(label) return 1 (as it's a hit else a miss)
        def getHitRatio(ranklist, targetItem):
            for item in ranklist:
                if item == targetItem:
                    return 1
            return 0
        
        
        def getNDCG(ranklist, targetItem):
            for i in range(len(ranklist)):
                item = ranklist[i]
                if item == targetItem:
                    return math.log(2) / math.log(i+2)
            return 0


        hr =[]
        NDCG = []
        
        # set of users in the testNeg set created above
        testUser = self.testNeg[0]
        
        # set of items in the testNeg set created above
        testItem = self.testNeg[1]
        
        for i in range(len(testUser)):
            
            # target item for the i(th) user, which has a non-zero rating.
            target = testItem[i][0]
            
            # prepare the input
            feed_dict = self.create_feed_dict(testUser[i], testItem[i])
            
            # evaluate the model by making predictions on the test negatives
            predict = sess.run(self.y_, feed_dict=feed_dict)
            # print(predict)

            # store the predictions for corresponding items
            item_score_dict = {}
            
            # iterate over all the test items present in each set
            for j in range(len(testItem[i])):
                item = testItem[i][j]
                item_score_dict[item] = predict[j]

            # returns a list with K largest items.
            ranklist = heapq.nlargest(topK, item_score_dict, key=item_score_dict.get)

            # calculate metrics over these predictions for the particular user (i).
            tmp_hr = getHitRatio(ranklist, target)
            tmp_NDCG = getNDCG(ranklist, target)
            
            # add the metric values to hr, ndcg main lists
            hr.append(tmp_hr)
            NDCG.append(tmp_NDCG)
            
        # average all the hit ratios and the ndcg values.
        return np.mean(hr), np.mean(NDCG)

# call the main() function
if __name__ == '__main__':
    main()