

import numpy as np
from collections import Counter


#Euclidian Distance - could be used as a global function and put in different file
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2)**2))


class k_nearest_neighbors:
    """ Class of the KNN Algorithm implementation. """
    """
    -k is classified by the majority of votes from its n number of neighbors.  
    """
    def __init__(self, k):
        #init for algorithm
        self.k = k 

    
    #fit the X_train - training samples and y_train training lables 
    # and store them so we can use later.
    def knn_fit(self, X_train, y_train): 
        """
        This method fits the data to the model
        """
        self.X_train = X_train
        self.y_train = y_train
    
    def knn_predict(self, X): #can have multiple examples with capital X.
        predicted_lables = [self._predict(x) for x in X]
        return np.array(predicted_lables) #np convert to array instead of list. 
    
    #helper method
    def _predict(self, x): #only get one sample with lower case x. 
        #compute distances
        #calculate all the distances of x to all the training samples  
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]

        #get k nearest samples, labels
        #argsort will sort distances - will return indices of how this is sorted
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        
        #majority vote, most common class label
        majority_vote = Counter(k_nearest_labels).most_common(1)  # (1) will return a list inside a tulup ([1,3])
        return majority_vote[0][0] #[0][0] to get first item and most number in list.  so we only want first item and actual item
