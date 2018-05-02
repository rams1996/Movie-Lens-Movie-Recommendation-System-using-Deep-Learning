# Movie-Lens-Movie-Recommendation-System-using-Deep-matrix-factorization-Method
This is a Deep learning model for recommendation systems with implementation in movie lens 100k and 1m database. 
This is the result of a research work carried out on Recommendation systems using deep learning. Our model was inspired from the paper "Deep matrix factorization models
for recomendation systems". 

Program specifications:
1.python DMF.py
    - default argument values will be used
    a. dataName: ml-100k
    b. negNum: 7 (sample sets with zero ratings)
    c. lr: 0.0001 (learning rate)
    d. maxEpochs: 50 (maximum iterations/epochs)
    e. batchSize: 256 (training instamces to be included during each forwards and backward pass)
    f. earlyStop: 5 (in case the model's performance dors changes after some iterations, we can stop the training of the model early.)
    g. checkPoint: ./check_Point/ (check_Point directory is created to store the model tensor variables.)
    h. topK: 10 (top K predictions chosen for evaluating the performance of the model by calculating the NDCG@10 (Normalized Discounted Cumulative Gain), HR@10 (Hit Ratio) values)

2. Example to add the command line arguments:
    a. python DMF.py -dataName ml-1m
        - for loading dataset with 1M ratings.
    b. python DMF.py -batchSize 128
        - specifying the batch size for training purpose.
    c. and many more are can be added.
