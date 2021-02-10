import numpy as np
import PIL

def get_k_means_plus_plus_center_indices(n, n_cluster, x, generator=np.random):
    '''

    :param n: number of samples in the data
    :param n_cluster: the number of cluster centers required
    :param x: data-  numpy array of points
    :param generator: random number generator from 0 to n for choosing the first cluster at random
            The default is np.random here but in grading, to calculate deterministic results,
            We will be using our own random number generator.


    :return: the center points array of length n_clusters with each entry being index to a sample
             which is chosen as centroid.
    '''
    # TODO:
    # implement the Kmeans++ algorithm of how to choose the centers according to the lecture and notebook
    # Choose 1st center randomly and use Euclidean distance to calculate other centers.

    #raise Exception('Implement get_k_means_plus_plus_center_indices function in Kmeans.py')

    centers=[]
    random_idx = generator.choice(n, size=1)[0]
    centers.append(random_idx)

    for c_index in range(1,n_cluster):
        x_dist_to_closest_cluster = []

        for x_idx in range(n):
            minDistYet = float('inf')
            for prev_c_idx in centers[:c_index]:
                dist = np.linalg.norm(x[x_idx] - x[prev_c_idx])
                if(dist < minDistYet):
                    minDistYet = dist
            x_dist_to_closest_cluster.append(minDistYet)
        centers.append(x_dist_to_closest_cluster.index(max(x_dist_to_closest_cluster)))

    # DO NOT CHANGE CODE BELOW THIS LINE

    print("[+] returning center for [{}, {}] points: {}".format(n, len(x), centers))
    return centers


def get_lloyd_k_means(n, n_cluster, x, generator):
    return generator.choice(n, size=n_cluster)

class KMeans():

    '''
        Class KMeans:
        Attr:
            n_cluster - Number of cluster for kmeans clustering (Int)
            max_iter - maximum updates for kmeans clustering (Int)
            e - error tolerance (Float)
            generator - random number generator from 0 to n for choosing the first cluster at random
            The default is np.random here but in grading, to calculate deterministic results,
            We will be using our own random number generator.
    '''
    def __init__(self, n_cluster, max_iter=100, e=0.0001, generator=np.random):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e
        self.generator = generator

    def fit(self, x, centroid_func=get_lloyd_k_means):

        '''
            Finds n_cluster in the data x
            params:
                x - N X D numpy array
                centroid_func - To specify which algorithm we are using to compute the centers(Lloyd(regular) or Kmeans++)
            returns:
                A tuple
                (centroids a n_cluster X D numpy array, y a length (N,) numpy array where cell i is the ith sample's assigned cluster, number_of_updates an Int)
            Note: Number of iterations is the number of time you update the assignment
        '''
        assert len(x.shape) == 2, "fit function takes 2-D numpy arrays as input"
        
        N, D = x.shape

        self.centers = centroid_func(len(x), self.n_cluster, x, self.generator)

        # TODO:
        # - comment/remove the exception.
        # - Initialize means by picking self.n_cluster from N data points
        # - Update means and membership until convergence or until you have made self.max_iter updates.
        # - return (means, membership, number_of_updates)

        # DONOT CHANGE CODE ABOVE THIS LINE
        #raise Exception('Implement fit function in KMeans class')

        # Initialize
        centroids = x[self.centers]
        J = 10**10
        y = np.empty(N)

        iter = 0
        # Repeat
        while(iter<self.max_iter):

            # Compute membership y

            distances_squared = np.sum(((x - np.expand_dims(centroids, axis=1)) ** 2), axis=2)
            y = np.argmin(distances_squared, axis=0)
            JNew = 0
            for c_idx in range(self.n_cluster):
                JNew += np.sum([np.sum((x[y == c_idx] - centroids[c_idx]) ** 2)])


            '''''''''
            for x_idx in range(len(x)):
                sample = x[x_idx]
                closestCentroidIdxYet = -1
                minDistYet = float('inf')
                for centroid_idx in range(self.n_cluster):
                    centroid = centroids[centroid_idx]
                    dist = np.linalg.norm(centroid - sample)

                    if(dist<minDistYet):
                            minDistYet = dist
                            closestCentroidIdxYet = centroid_idx
                y[x_idx]=int(closestCentroidIdxYet)
                # Compute distortion measure J new
                JNew += minDistYet

                #print(closestCentroidIdxYet)
            '''''''''

            if(abs(JNew - J) <= self.e):
                break

            J = JNew

            # compute centroids

            for c_idx in range(self.n_cluster):
                yk = np.array(y==c_idx)
                centroids[c_idx] = np.dot(yk,x)
                count = np.count_nonzero(y == c_idx)
                if count > 0:
                         centroids[c_idx] = centroids[c_idx]/count

            '''''''''
            centroid_sums = np.zeros((self.n_cluster,D) )
            centroid_count = np.zeros(self.n_cluster)

            for x_idx in range(len(x)):

                centroid_sums[int(y[x_idx])]+= x[x_idx]
                centroid_count[int(y[x_idx])]+=1

            #print(centroid_sums)

            for centroid_idx in range(self.n_cluster):
                #print(centroid_count[centroid_idx])
                if(centroid_count[centroid_idx]>0):
                    centroid_sums[centroid_idx] /= centroid_count[centroid_idx]
                else:
                    centroid_sums[centroid_idx]=centroids[centroid_idx]

            centroids = centroid_sums
            '''''''''

            iter+=1
        # DO NOT CHANGE CODE BELOW THIS LINE
        return centroids, y, iter

class KMeansClassifier():

    '''
        Class KMeansClassifier:
        Attr:
            n_cluster - Number of cluster for kmeans clustering (Int)
            max_iter - maximum updates for kmeans clustering (Int)
            e - error tolerance (Float)
            generator - random number generator from 0 to n for choosing the first cluster at random
            The default is np.random here but in grading, to calculate deterministic results,
            We will be using our own random number generator.
    '''

    def __init__(self, n_cluster, max_iter=100, e=1e-6, generator=np.random):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e
        self.generator = generator


    def fit(self, x, y, centroid_func=get_lloyd_k_means):
        '''
            Train the classifier
            params:
                x - N X D size  numpy array
                y - (N,) size numpy array of labels
                centroid_func - To specify which algorithm we are using to compute the centers(Lloyd(regular) or Kmeans++)

            returns:
                None
            Stores following attributes:
                self.centroids : centroids obtained by kmeans clustering (n_cluster X D numpy array)
                self.centroid_labels : labels of each centroid obtained by
                    majority voting (N,) numpy array)
        '''

        assert len(x.shape) == 2, "x should be a 2-D numpy array"
        assert len(y.shape) == 1, "y should be a 1-D numpy array"
        assert y.shape[0] == x.shape[0], "y and x should have same rows"

        self.generator.seed(42)
        N, D = x.shape
        # TODO:
        # - comment/remove the exception.
        # - Implement the classifier
        # - assign means to centroids
        # - assign labels to centroid_labels

        # DONOT CHANGE CODE ABOVE THIS LINE
        #raise Exception('Implement fit function in KMeansClassifier class')

        my_KMeans = KMeans( self.n_cluster, self.max_iter, self.e, self.generator)

        centroids, membership, iter = my_KMeans.fit(x,centroid_func)

        votes_counter = []
        for i in range(self.n_cluster):
            votes_counter.append({})

        for i in range(N):
            label = y[i]
            member = int(membership[i])
            if( label not in votes_counter[member].keys()):
                votes_counter[member][label]=1
            else:
                votes_counter[member][label]+=1

        centroid_labels = []

        for vote in votes_counter:
            if not vote:
                centroid_labels.append(0)
            centroid_labels.append(max(vote,key=vote.get))
        centroid_labels = np.array(centroid_labels)

        # DONOT CHANGE CODE BELOW THIS LINE

        self.centroid_labels = centroid_labels
        self.centroids = centroids

        assert self.centroid_labels.shape == (
            self.n_cluster,), 'centroid_labels should be a numpy array of shape ({},)'.format(self.n_cluster)

        assert self.centroids.shape == (
            self.n_cluster, D), 'centroid should be a numpy array of shape {} X {}'.format(self.n_cluster, D)

    def predict(self, x):
        '''
            Predict function
            params:
                x - N X D size  numpy array
            returns:
                predicted labels - numpy array of size (N,)
        '''

        assert len(x.shape) == 2, "x should be a 2-D numpy array"

        self.generator.seed(42)
        N, D = x.shape
        # TODO:
        # - comment/remove the exception.
        # - Implement the prediction algorithm
        # - return labels

        # DONOT CHANGE CODE ABOVE THIS LINE
        #raise Exception('Implement predict function in KMeansClassifier class')

        labels = []

        for x_idx in range(N):
            minDistYet = float('inf')
            closestLabelYet = -1
            sample = x[x_idx]

            for k in range(self.n_cluster):
                dist = np.linalg.norm(sample - self.centroids[k])
                if(dist<minDistYet):
                    closestLabelYet = self.centroid_labels[k]
                    minDistYet = dist
            labels.append(closestLabelYet)

        labels = np.array(labels)

        # DO NOT CHANGE CODE BELOW THIS LINE
        return np.array(labels)
        

def transform_image(image, code_vectors):

    '''
        Quantize image using the code_vectors

        Return new image from the image by replacing each RGB value in image with nearest code vectors (nearest in euclidean distance sense)

        returns:
            numpy array of shape image.shape
    '''

    assert image.shape[2] == 3 and len(image.shape) == 3, \
        'Image should be a 3-D array with size (?,?,3)'

    assert code_vectors.shape[1] == 3 and len(code_vectors.shape) == 2, \
        'code_vectors should be a 2-D array with size (?,3)'

    # TODO
    # - comment/remove the exception
    # - implement the function

    # DONOT CHANGE CODE ABOVE THIS LINE
    #raise Exception('Implement transform_image function')

    original_shape = image.shape
    N=original_shape[0]*original_shape[1]
    D=original_shape[2]
    n_clusters = code_vectors.shape[0]

    flattened_img = np.reshape(image,(N,D))

    flat_new_im_idx = np.empty(N)

    distances_squared = np.sum(((flattened_img - np.expand_dims(code_vectors, axis=1)) ** 2), axis=2)
    flat_new_im_idx = np.argmin(distances_squared, axis=0)
    new_im = []
    for idx in flat_new_im_idx:
        new_im.append(code_vectors[idx])
    #print(new_im.shape)
    new_im = np.array(new_im)
    new_im = np.reshape(new_im,original_shape)

    '''''''''
    for pixel in image:
        minDistYet = float('inf')
        minCVIdxYet = -1

        for cv_idx in range(len(code_vectors)):
            cv = code_vectors[cv_idx]
            dist = np.linalg.norm(cv-pixel)
            if(dist<minDistYet):
                minDistYet=dist
                minCVIdxYet = cv_idx
        new_im.append(code_vectors[minCVIdxYet])
    '''''''''



    # DONOT CHANGE CODE BELOW THIS LINE
    return new_im

