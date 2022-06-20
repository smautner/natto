import numpy as np
import pandas as pd
import math
import time
from natto import input
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics import pairwise_distances
from scipy.optimize import linear_sum_assignment
from sklearn import neighbors as nbrs, metrics
from scipy.sparse.csgraph import dijkstra



def __init__():
        pass

'''
def main():
        ### Calculates the k nearest neighbors for data 

        t0 = time.time()
        path = "/Users/JackBrons/Freiburg_Sem4/Research Project/nattoWork/sim_data/"
        Data = []

        k = 4

        for i in range(5):
                Data.append(pd.read_csv(f'{path}time{i}.csv', sep=',', header=None).values)
        print("Data Loaded")

        if len(Data) <= 2:
                print("More data plz")
                quit()

        kNearestNeighbors = []
        for index, dataset in enumerate(Data):
                ### Get previous and next datasets and their overall index positions
                if index == 0:
                        indexStart = 0
                        prevData = []
                        prevIndexStart = 0
                        nextData = Data[index+1]
                        nextIndexStart = indexStart + len(dataset)
                elif index == (len(Data)-1):
                        indexStart += len(Data[index-1])
                        prevData = Data[index-1]
                        prevIndexStart = indexStart - len(prevData)
                        nextData = []
                else:
                        indexStart += len(Data[index-1])
                        prevData = Data[index-1]
                        prevIndexStart = indexStart - len(prevData)
                        nextData = Data[index+1]
                        nextIndexStart = indexStart + len(dataset)

                for i, dataPoint1 in enumerate(dataset):

                        ### Calculate nearest k/2 neighbors from previous time point
                        distances = []
                        for j, dataPoint2 in enumerate(prevData):
                                distance = euclidean_distance(dataPoint1, dataPoint2)
                                distances.append([i+indexStart, distance, j+prevIndexStart])
                        #take k closeset points, and you've got k nearest neighbours!
                        #sort_distances by index 2: distance
                        distances.sort(key= lambda x: x[1])
#                        print(distances)
                        if bool(distances)==True:
                                kNearestNeighbors.extend(distances[:k//2])

                        ### Calculate nearest k/2 neighbors from next time point
                        distances = []
                        for j, dataPoint2 in enumerate(nextData):
                                distance = euclidean_distance(dataPoint1, dataPoint2)
                                distances.append([i+indexStart, distance, j+nextIndexStart])
                        #take k closeset points, and you've got k nearest neighbours!
                        #sort_distances by index 2: distance
                        distances.sort(key= lambda x: x[1])
                        if bool(distances)==True:
                                # If list not empty, add top k/2 neighbours
                                kNearestNeighbors.extend(distances[:k//2])

                print(f"dataset {index} finished")
        print(kNearestNeighbors)
        print(f"len: {kNearestNeighbors}")

        ### Write out our knn into a csv file
        outputDF = pd.DataFrame(kNearestNeighbors)
        outputDF.to_csv('KNNGraph.csv', index=False, header=False)

        t1 = time.time()
        print(f"Time Elapsed: {t1-t0}")
'''


def timeSliceNearestNeighbor(Data, 
        k_from_neighbors=6, 
        k_from_same=15, 
        sort=True, 
        metric='nn', 
        neighbor_distance='max', 
        dist_coeff=1.5,
        verbose=False):
        ### Calculates nearest neighbors of input data between time slices.

        if len(Data) <= 2: assert("More data plz")

        ### Initialize variables
        if type(Data[0]) != np.ndarray:
                Data = [data.to_df().to_numpy() for data in Data]
        prevData = None
        prevIndex = 0
        nextData = Data[1]
        nextIndex = len(Data[0])
        indexStart = 0
        adjacentData = 1 # Equals to 1 or 2 depending on whether we are dealing with an edge dataset.

##########################################################################################
##########################################################################################
##########################################################################################
        for index, dataset in enumerate(Data):

                if metric == 'nn':
                        ### Make and fit nearest neighbor object and get k nearest neighbors for the\
                        ### current dataset. 
                        nn = NearestNeighbors(n_neighbors=k_from_neighbors//adjacentData)
                        knn_dists, knn_ind = NearestNeighbors(n_neighbors=k_from_same).fit(dataset).kneighbors(dataset)
                        knn_ind[:,:] = np.add(knn_ind[:,:], np.full((knn_ind.shape), indexStart)) #Adjust indices

                        ### Checks if we are at first or last dataset
                        knn_l_ind = None
                        knn_r_ind = None
                        if type(prevData) == np.ndarray:
                                ### Gets k Nearest Neighbors on the left dataset & updates indices
                                knn_l_dists, knn_l_ind = nn.fit(prevData).kneighbors(dataset)
                                knn_l_ind[:,:] = np.add(knn_l_ind[:,:], np.full((knn_l_ind.shape), prevIndex))
                        if type(nextData) == np.ndarray:
                                ### Gets k Nearest Neighbors on the right dataset & updates indices
                                knn_r_dists, knn_r_ind = nn.fit(nextData).kneighbors(dataset) 
                                knn_r_ind[:,:] = np.add(knn_r_ind[:,:], np.full((knn_r_ind.shape), nextIndex))

                        #knn_neighbor_ind = np.column_stack([l for l in [knn_l_ind, knn_r_ind] if l is not None])
                        knn_neighbor_ind = np.column_stack([l for l in [knn_l_ind, knn_ind, knn_r_ind] if l is not None])

##########################################################################################
##########################################################################################
##########################################################################################

                elif metric == 'hungarian':
                        ### Use hungarian algorithm to find k nearest neighbors

                        hung_distances = pairwise_distances(dataset, dataset)
                        knn_ind, knn_dists = findSliceNeighbors(dataset, hung_distances, k_from_same, indexStart)
                        
                        ### Get left neighbors
                        if type(prevData) == np.ndarray:
                                l_distances = pairwise_distances(dataset, prevData)
                                knn_l_ind, knn_l_dists = findSliceNeighbors(dataset, l_distances, k_from_neighbors//adjacentData, prevIndex)
                        else:
                                knn_l_ind = np.empty((dataset.shape[0], 0), int)

                        # Get right neighbors
                        if type(nextData) == np.ndarray:
                                r_distances = pairwise_distances(dataset, nextData)
                                knn_r_ind, knn_r_dists = findSliceNeighbors(dataset, r_distances, k_from_neighbors//adjacentData, nextIndex)
                        else:
                                knn_r_ind = np.empty((dataset.shape[0], 0), int)

                        #knn_neighbor_ind = np.column_stack([l for l in [knn_l_ind, knn_r_ind] if l is not None])
                        knn_neighbor_ind = np.column_stack([l for l in [knn_l_ind, knn_ind, knn_r_ind] if l is not None])

##########################################################################################
##########################################################################################
##########################################################################################
########## TEST HOW TO MAKE HUNGARIAN TAKE EQUAL NEIGHBORS FROM ALL OTHER SLICES ##########

                elif metric == 'hungarian_all' or metric=='hungarian_all_testnewdistances':
                        ### Use hungarian algorithm to find k nearest neighbors between ALL time slices
                        
                        knn_l_ind = None
                        knn_r_ind = None

                        ### Gets 'k_from_same' nearest neighbors from within the same timeslice
                        hung_distances = pairwise_distances(dataset, dataset)
                        knn_ind, knn_dists = findSliceNeighbors(dataset, hung_distances, k_from_same, indexStart)

                        ### Get the list of neighbor cells from between timeslices
                        indList = []
                        for index2, otherData in enumerate(Data):
                                otherPrevIndex = 0
                                if index != index2:
                                        other_distances = pairwise_distances(dataset, otherData)
                                        knn_other_ind, knn_other_dists = findSliceNeighbors(dataset, other_distances, k_from_neighbors, otherPrevIndex)
                                        indList.append(knn_other_ind)
                                ### New Part
                                else:
                                        indList.append(knn_ind)
                                ### New Part End
                                        
                                otherPrevIndex += len(otherData)
                        knn_neighbor_ind = np.column_stack(indList)

##########################################################################################
##########################################################################################
##########################################################################################

                elif metric == 'mykernel':
                        ### NOT FUNCTIONAL
                        knn_l_ind = None
                        knn_r_ind = None

                        hung_distances = mykernel(X=[dataset,dataset], neighbors = k_from_same)
                        print(hung_distances)
                        knn_ind, knn_dists = findSliceNeighbors(dataset, hung_distances, k_from_same, indexStart)

                        indList = []
                        for index2, otherData in enumerate(Data):
                                otherPrevIndex = 0
                                if index != index2:
                                        other_distances = mykernel(X=[dataset, otherData], neighbors=k_from_neighbors)
                                        knn_other_ind, knn_other_dists = findSliceNeighbors(dataset, other_distances, k_from_neighbors, otherPrevIndex)
                                        indList.append(knn_other_ind)
                                        print(distances)
                                else:
                                        indList.append(knn_ind)
                                otherPrevIndex += len(otherData)
                        knn_neighbor_ind = np.column_stack(indList)

##########################################################################################
##########################################################################################
##########################################################################################


                ### Add data and indices to overall list
                ### Also adjusts distances of neighbors from alternate time slices in order to prevent
                ### large distances from skewing the projections.

                neighbor_dists = calc_neighbor_dists(dataset, knn_dists, k_from_neighbors, metric, neighbor_distance, len(Data))

                if metric == "hungarian_all":
                        '''
                        distlist = [neighbor_dists[:, 0:(k_from_neighbors)] for x in range(len(Data)-1)]
                        distlist.append(knn_dists)
                        '''
                        distlist = [neighbor_dists[:, 0:(k_from_neighbors)] if x!= index else knn_dists for x in range(len(Data))]
                        tempDistance = np.column_stack(distlist)

                elif metric == 'hungarian_all_testnewdistances':
                        distlist = [neighbor_dists[:, 0:(k_from_neighbors)] if x!= index else knn_dists for x in range(len(Data))]
                        for i, timeSlice in enumerate(distlist):
                                distlist[i] = timeSlice*(dist_coeff*(np.abs(index-i)+1))
                                #distlist[i] = timeSlice + dist_coeff*np.abs(index-i)

                        #distlist.append(knn_dists)
                        tempDistance = np.column_stack(distlist)
                elif knn_l_ind is not None and knn_r_ind is not None:
                        #tempDistance = np.column_stack([neighbor_dists[:, 0:(k_from_neighbors//2)], 
                        #        neighbor_dists[:, 0:(k_from_neighbors//2)], knn_dists])
                        ###NEW
                        tempDistance = np.column_stack([neighbor_dists[:, 0:(k_from_neighbors//2)], 
                                knn_dists,
                                neighbor_dists[:, 0:(k_from_neighbors//2)]])
                        ###NEW
                else:
                        #tempDistance = np.column_stack([neighbor_dists, knn_dists])
                        ###NEW
                        if knn_l_ind is not None:
                                tempDistance = np.column_stack([neighbor_dists, knn_dists])
                        elif knn_r_ind is not None:
                                tempDistance = np.column_stack([knn_dists, neighbor_dists])
                        ###NEW

                #tempIndices = np.column_stack([knn_neighbor_ind, knn_ind])
                tempIndices = knn_neighbor_ind
                if index == 0:
                        if metric=='hungarian_all' or metric=='hungarian_all_testnewdistances':
                                distances = np.empty((0, k_from_same+(len(Data)-1)*k_from_neighbors), int)
                                indices = np.empty((0, k_from_same+(len(Data)-1)*k_from_neighbors), int)
                        else:
                                distances = np.empty((0, k_from_same+k_from_neighbors), int)
                                indices = np.empty((0, k_from_same+k_from_neighbors), int)


                distances = np.vstack((distances, tempDistance))
                indices = np.vstack((indices, tempIndices))

                ### Updates variables for next iteration
                prevData = dataset
                prevIndex = indexStart
                indexStart = indexStart + len(dataset)
                if index >= len(Data)-2:
                        nextData = None
                        adjacentData = 1
                else:
                        nextData = Data[index+2]
                        nextIndex = indexStart + len(Data[index+1])
                        adjacentData = 2

                print(f"Dataset {index} Complete")


        ### Finally sort indices and distances so neighbors with shortest distances are first\
        ### and neighbors with furthest distances are last
        if sort:
                distances, indices = sortDistsAndInds(distances, indices)
                '''
                sortedDistances = []
                sortedIndices = []
                for d, i in zip(distances, indices):
                        listd, listi = zip(*sorted(zip(d, i)))
                        sortedDistances.append(listd)
                        sortedIndices.append(listi)
                distances = np.asarray(sortedDistances)
                indices = np.asarray(sortedIndices)
                '''
                

        if verbose:
                print((distances))
                print(f"Length: {len(distances)}")
                print(indices)
                print(f"\nShape: ({len(indices)}, {len(indices[0])})\n")

        return indices, distances



def findSliceNeighbors(data, distances, k_neighbors, indexStart):
        knn_ind = np.empty((data.shape[0], 0), int)
        knn_dists = np.empty((data.shape[0], 0), int)

        ### Gets 'k_neighbors' nearest neighbors from within the same timeslice
        i = 0
        while i < k_neighbors:
                row_ind, col_ind = linear_sum_assignment(distances)
                knn_dists = np.column_stack((knn_dists, distances[row_ind, col_ind]))
                distances, knn_ind = calc_neighbor_indices(distances, row_ind, col_ind, knn_ind, indexStart)

                i += 1   

        return knn_ind, knn_dists   



def calc_neighbor_indices(distances, row_ind, col_ind, knn_ind, indexStart):
        distances[row_ind, col_ind] = distances[row_ind, col_ind] + np.amax(distances)
        col_ind = col_ind + indexStart
        knn_ind = np.column_stack((knn_ind, col_ind))

        return distances, knn_ind



def calc_neighbor_dists(dataset, knn_dists, k_from_neighbors, metric, dist_metric, dataLen):
        if metric=='hungarian_all':
                coeff = dataLen-1
        else:
                coeff = 1

        if dist_metric == 'max':
                neighbor_dists = np.tile(np.full(coeff*k_from_neighbors, np.amax(knn_dists)), (len(dataset),1))
        elif dist_metric == 'median':
                neighbor_dists = np.tile(np.full(coeff*k_from_neighbors, np.median(knn_dists)), (len(dataset),1))
        elif dist_metric == '3quartile':
                neighbor_dists = np.tile(np.full(coeff*k_from_neighbors, np.percentile(knn_dists, 75)), (len(dataset),1))
        elif dist_metric == 'firstKMeans':
                neighbor_dists = np.tile(np.mean(knn_dists, axis=1)[1:coeff*(k_from_neighbors)+1], (len(dataset),1))
        elif dist_metric == 'lastKMeans':
                neighbor_dists = np.tile(
                        np.mean(knn_dists, axis=1)[knn_dists.shape[1]-coeff*(k_from_neighbors)-1:knn_dists.shape[1]-1], 
                        (len(dataset),1))

        return neighbor_dists
  

def sortDistsAndInds(distances, indices):
        sortedDistances = []
        sortedIndices = []
        for d, i in zip(distances, indices):
                listd, listi = zip(*sorted(zip(d, i)))
                sortedDistances.append(listd)
                sortedIndices.append(listi)
        distances = np.asarray(sortedDistances)
        indices = np.asarray(sortedIndices)

        return distances, indices


def hungmat(x1,x2):
        x= metrics.euclidean_distances(x1,x2)
        r = np.zeros_like(x)
        a,b = linear_sum_assignment(x)
        r[a,b] = 1

        r2 = np.zeros((x.shape[1], x.shape[0]))
        r2[b,a] = 1 # rorated :)
        return r,r2


def mykernel(x1len=False, neighbors = 3, X=[], _=None, return_graph = False):
    '''
    X is list containing 2 datasets
    - we do neighbors to get quadrant 2 and 4
    - we do hungarian to do quadrants 1 and 3
    - we do dijkstra to get a complete distance matrix

    '''
    x1,x2 = X[0], X[1]
    q2 = nbrs.kneighbors_graph(x1,neighbors).todense()
    q4 = nbrs.kneighbors_graph(x2,neighbors).todense()
    q1,q3 = hungmat(x1,x2)

    graph = np.hstack((np.vstack((q2,q3)),np.vstack((q1,q4))))


    connect = dijkstra(graph,unweighted = True, directed = False)

    if return_graph:
        return graph, connect
    distances = -connect # invert
    distances -= distances.min() # longest = 4
    distances /= distances.max() # between 0 and 1 :)
    distances[distances < np.median(np.unique(distances))] = 0

    return np.power(distances,2)


'''
run_btnn_test = False
dataset = 'Waterston' #sim or Waterston
if run_btnn_test:
        Data = []
        Labels = []
        if dataset=='sim':
                path = "/Users/JackBrons/Freiburg_Sem4/Research Project/nattoWork/sim_data/"
                for i in range(5):
                        Data.append(pd.read_csv(f'{path}time{i}.csv', sep=',', header=None).values)
#                Labels.append(pd.read_csv(f'{inputDirectory}time{i}_labels.csv', sep=',', header=None)[0].values.astype(int)) 
        elif dataset=='Waterston':
                path = "/Users/JackBrons/Freiburg_Sem4/Research Project/nattoWork/GSE126954_data/Waterston_"
                for item in ['300', '400', '500_1']:
                        anndata = input.loadGSM(f'{path}{item}/', subsample=3000, cellLabels=True,)

                        data = anndata.to_df().to_numpy()
                        label = anndata.obs['labels'].to_numpy()

#                        data = data[~np.isnan(label)]
#                        label = label[~np.isnan(label)]

                        Data.append(data)
                        Labels.append(label)

        print("Beginning NN Calculation")
        (indices, distances) = timeSliceNearestNeighbor(Data, sort=True, metric= 'hungarian', neighbor_distance='3quartile')
        print(indices)
        print(list(distances[5]))
        print(list(distances[3300]))
        print(list(indices[5]))
        print(list(indices[3300]))
'''

