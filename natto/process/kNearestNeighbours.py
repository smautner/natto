import numpy as np
import pandas as pd
import math
import time
from natto import input
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import kneighbors_graph


def __init__():
        pass


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



def between_time_nearest_neighbors(Data, k_from_neighbors=6, k_from_same=15, sort=True, metric='nn', neighbor_distance='max', verbose=False):
        ### Calculates nearest neighbors of input data between time slices.

        if len(Data) <= 2: assert("More data plz")

        ### Initialize variables
        prevData = None
        prevIndex = 0
        nextData = Data[1]
        nextIndex = len(Data[0])
        indexStart = 0
        adjacentData = 1 # Equals to 1 or 2 depending on whether we are dealing with an edge dataset.
        distances = np.empty((0, k_from_same+k_from_neighbors), int)
        indices = np.empty((0, k_from_neighbors+k_from_same), int)

        for index, dataset in enumerate(Data):

                ### Make and fit nearest neighbor object and get k nearest neighbors for the\
                ### current dataset. 
                if metric == 'nn':
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

                elif metric == 'hungarian':
                        ### Use hungarian algorithm to find k nearest neighbors
                        from sklearn.metrics import pairwise_distances
                        from scipy.optimize import linear_sum_assignment

                        hung_distances = pairwise_distances(dataset, dataset)

                        i = 0
                        knn_ind = np.empty((dataset.shape[0], 0), int)
                        knn_dists = np.empty((dataset.shape[0], 0), int)
                        ### Gets 'k_from_same' nearest neighbors from within the same timeslice
                        while i < k_from_same:
                                row_ind, col_ind = linear_sum_assignment(hung_distances)
                                knn_dists = np.column_stack((knn_dists, hung_distances[row_ind, col_ind]))
                                hung_distances[row_ind, col_ind] = hung_distances[row_ind, col_ind] + np.amax(hung_distances)
                                col_ind = col_ind + indexStart
                                knn_ind = np.column_stack((knn_ind, col_ind))
                                i += 1

                        i = 0
                        knn_l_ind = np.empty((dataset.shape[0], 0), int)
                        knn_r_ind = np.empty((dataset.shape[0], 0), int)
                        #print(dataset.shape)

                        if type(prevData) == np.ndarray:
                                l_distances = pairwise_distances(dataset, prevData)
                        if type(nextData) == np.ndarray:
                                r_distances = pairwise_distances(dataset, nextData)

                        while i < k_from_neighbors//adjacentData:
                                if type(prevData) == np.ndarray:
                                        row_ind, col_ind = linear_sum_assignment(l_distances)
                                        l_distances[row_ind, col_ind] = l_distances[row_ind, col_ind] + np.amax(l_distances)
                                        col_ind = col_ind + prevIndex
                                        knn_l_ind = np.column_stack((knn_l_ind, col_ind))
                                if type(nextData) == np.ndarray:
                                        row_ind, col_ind = linear_sum_assignment(r_distances)
                                        r_distances[row_ind, col_ind] = r_distances[row_ind, col_ind] + np.amax(r_distances)
                                        col_ind = col_ind + nextIndex
                                        knn_r_ind = np.column_stack((knn_r_ind, col_ind))
                                i+=1



                ### Add data and indices to overall list
                ### Also adjusts distances of neighbors from alternate time slices in order to prevent
                ### large distances from skewing the projections.
                if neighbor_distance == 'max':
                        neighbor_dists = np.tile(np.full(k_from_neighbors, np.amax(knn_dists)), (len(dataset),1))
                elif neighbor_distance == 'median':
                        neighbor_dists = np.tile(np.full(k_from_neighbors, np.median(knn_dists)), (len(dataset),1))
                elif neighbor_distance == '3quartile':
                        neighbor_dists = np.tile(np.full(k_from_neighbors, np.percentile(knn_dists, 75)), (len(dataset),1))
                elif neighbor_distance == 'firstKMeans':
                        neighbor_dists = np.tile(np.mean(knn_dists, axis=1)[1:k_from_neighbors+1], (len(dataset),1))
                elif neighbor_distance == 'lastKMeans':
                        neighbor_dists = np.tile(
                                np.mean(knn_dists, axis=1)[knn_dists.shape[1]-k_from_neighbors-1:knn_dists.shape[1]-1], 
                                (len(dataset),1))


                if knn_l_ind is not None and knn_r_ind is not None:
                        tempDistance = np.column_stack([neighbor_dists[:, 0:(k_from_neighbors//2)], 
                                neighbor_dists[:, 0:(k_from_neighbors//2)], knn_dists])
                else:
                        tempDistance = np.column_stack([neighbor_dists, knn_dists])

                tempIndices = np.column_stack([l for l in [knn_l_ind, knn_r_ind, knn_ind] if l is not None])
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
                        #nextData = Data[index+1]
                        #nextIndex = indexStart + len(dataset)
                        nextData = Data[index+2]
                        nextIndex = indexStart + len(Data[index+1])
                        adjacentData = 2

                print(f"Dataset {index} Complete")


        ### Finally sort indices and distances so neighbors with shortest distances are first\
        ### and neighbors with furthest distances are last
        if sort:
                sortedDistances = []
                sortedIndices = []
                for d, i in zip(distances, indices):
                        listd, listi = zip(*sorted(zip(d, i)))
                        sortedDistances.append(listd)
                        sortedIndices.append(listi)
                distances = np.asarray(sortedDistances)
                indices = np.asarray(sortedIndices)

        if verbose:
                print((distances))
                print(f"Length: {len(distances)}")
                print(indices)
                print(f"\nShape: ({len(indices)}, {len(indices[0])})\n")

        return indices, distances




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
        (indices, distances) = between_time_nearest_neighbors(Data, sort=True, metric= 'hungarian', neighbor_distance='3quartile')
        print(indices)
        print(list(distances[5]))
        print(list(distances[3300]))
        print(list(indices[5]))
        print(list(indices[3300]))

