import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics import pairwise_distances
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix, hstack, vstack


def __init__():
        pass

def timeSliceNearestNeighbor(Data, 
        kFromNeighbors=6, 
        kFromSame=16, 
        sort=True,
        intraSliceNeighbors="sklearnNN",
        interSliceNeighbors=None,
        distanceMetric='max', 
        distCoeff=1,
        returnSparse=False,
        silent=False):

        ### Initialize variables
        Data, intraSliceNeighbors, interSliceNeighbors, sparseSlices = initialize(Data, intraSliceNeighbors, interSliceNeighbors)
        edgeCaseMultiplier = 2 # Equals to 1 or 2 depending on whether we are dealing with an 'edge' dataset.

        for index, timeSlice in enumerate(Data):
                ### Find indices and distances for all Neighbors at each 'timeSlice'
                sparseSlice = globalNN2(Data,
                        index, 
                        kFromSame, 
                        kFromNeighbors, 
                        intraSliceNeighbors, 
                        interSliceNeighbors, 
                        distanceMetric,
                        distCoeff,
                        edgeCaseMultiplier)
                sparseSlices = vstack((sparseSlices, sparseSlice))

                ### Updates variables for next iteration
                if (index >= len(Data)-2):
                        edgeCaseMultiplier = 2
                else:
                        edgeCaseMultiplier = 1

                if not silent:
                        print(f"Dataset {index} Complete")

        ### Prepare for Return
        sparseMatrix = sparseSlices.tocsr()
        if returnSparse:
                return sparseMatrix
        else:
                precomputedKNNIndices = []
                precomputedKNNDistances = []
                for ip in range(len(sparseMatrix.indptr)-1):
                        start = sparseMatrix.indptr[ip]
                        end = sparseMatrix.indptr[ip+1]
                        precomputedKNNIndices.append(sparseMatrix.indices[start:end])
                        precomputedKNNDistances.append(sparseMatrix.data[start:end])
                return sortDistsAndInds(np.vstack(precomputedKNNIndices), np.vstack(precomputedKNNDistances))


def initialize(Data, intraSliceNeighbors, interSliceNeighbors):
        if type(Data[0]) != np.ndarray:
                Data = [data.to_df().to_numpy() for data in Data]
        if type(intraSliceNeighbors)==str:
                intraSliceNeighbors = eval(intraSliceNeighbors)
        if interSliceNeighbors is None:
                interSliceNeighbors = intraSliceNeighbors
        elif type(interSliceNeighbors)==str:
                interSliceNeighbors = eval(interSliceNeighbors)
        sparseSlices = csr_matrix((0, Data[0].shape[0]*len(Data)))

        return (Data, intraSliceNeighbors, interSliceNeighbors, sparseSlices)


def sklearnNN(kNeighbors, timeSlice1, timeSlice2, index1, index2, edgeCase):
        ### Get Nearest Neighbors between two time-slices
        if np.abs(index1-index2)<=1:
                knnDists, knnInd = NearestNeighbors(n_neighbors=kNeighbors*edgeCase).fit(timeSlice2).kneighbors(timeSlice1)
                return (knnInd, knnDists)
        else:
                return ([], [])


def hungarian(kNeighbors, timeSlice1, timeSlice2, index1, index2, edgeCase):
        ### Use hungarian algorithm to find k nearest neighbors
        if np.abs(index1-index2)<=1:
                hungDistances = pairwise_distances(timeSlice1, timeSlice2)
                knnDists, knnInd = findSliceNeighbors(timeSlice1, hungDistances, kNeighbors*edgeCase) 
                return (knnDists, knnInd)               
        else:
                return ([], [])


def hungarianAll(kNeighbors, timeSlice1, timeSlice2, index1, index2, edgeCase):
        ### Use hungarian algorithm to find k nearest neighbors between ALL time slices      
        hungDistances = pairwise_distances(timeSlice1, timeSlice2)
        knnInd, knnDists = findSliceNeighbors(timeSlice1, hungDistances, kNeighbors)
        return (knnInd, knnDists)


def findSliceNeighbors(data, distances, kNeighbors):
        def hungarianLoop(distances):
                rowInd, colInd  = linear_sum_assignment(distances)
                colDist = distances[rowInd, colInd]
                distances[rowInd, colInd] = np.inf
                return colInd, colDist

        colInds, colDists = zip(*[hungarianLoop(distances) for i in range(kNeighbors)])
        return (np.column_stack(colInds), np.column_stack(colDists))


def globalNN2(Data, index, kFromSame, kFromNeighbors, intraSliceNeighbors, interSliceNeighbors, distanceMetric, distCoeff, edgeCase):
        intersectList = []
        neighborSliceIndex = 0
        currentSlice = Data[index]
        for index2, otherSlice in enumerate(Data):
                intersect = csr_matrix((currentSlice.shape[0], otherSlice.shape[0]))
                if index == index2:
                        indicesTemp, distancesTemp = intraSliceNeighbors(kFromSame, currentSlice, currentSlice, index, index2, 1)
                        intraDists = distancesTemp.copy()
                        sameIntersect = intersect
                else:
                        indicesTemp, distancesTemp = interSliceNeighbors(kFromNeighbors, currentSlice, otherSlice, index, index2, edgeCase)
                        intersectList.append(intersect)

                for i in range(len(indicesTemp)): intersect[i, indicesTemp[i,:]] = distancesTemp[i,:]
                        
        slices = adjustSparseDists(intraDists, intersectList, kFromNeighbors, distanceMetric, distCoeff, edgeCase)
        slices.insert(index, sameIntersect)

        sliceSparseMatrix = hstack(slices)

        return sliceSparseMatrix

def adjustSparseDists(intraDists, interDistances, kFromNeighbors, distMetric, distCoeff, edgeCase):
        ### Calculate Distances for the metrics which are in relation to intra-distances
                ### 'euclidean' and 'normMeans' calculate inter-slice neighbor distances by manipulating their real euclidean distances
                ###  the other methods estimate inter-slice neighbor distances by using the intra-slice neighbor distances

        if distMetric == 'euclidean':
                for d in interDistances: d.multiply(distCoeff)
        elif distMetric == 'normMeans':
                meanFactor=np.array([np.divide(np.mean(intraDists[:,1:], axis=1), (d.sum(axis=1).reshape(-1,1)[0]/d.getnnz(axis=1)) ) 
                        if d.nnz!=0 else np.ones(d.shape[0]) for d in interDistances])
                for k,di in zip(meanFactor,range(len(interDistances))): interDistances[di] = interDistances[di].multiply(distCoeff*k.reshape(-1,1))
        elif distMetric == 'max':
                for d in interDistances: d.data=np.array([distCoeff*np.amax(intraDists)]*len(d.data))
        elif distMetric == 'median':
                for d in interDistances: d.data=np.array([distCoeff*np.median(intraDists)]*len(d.data))
        elif distMetric == '3quartile':
                for d in interDistances: d.data=np.array([distCoeff*np.percentile(intraDists, 75)]*len(d.data))
        elif 'KMeans' in distMetric:
                if distMetric == 'firstKMeans':
                        means = distCoeff*np.mean(intraDists, axis=0)[1:edgeCase*kFromNeighbors+1]
                elif distMetric == 'lastKMeans':
                        means = distCoeff*np.mean(intraDists, axis=0)[-(edgeCase*kFromNeighbors):]
                for i, d in enumerate(interDistances):
                        if d.nnz!=0:
                                for ip in range(len(d.indptr)-1):
                                        start, end = d.indptr[ip:ip+2]
                                        sortData = sorted(d.data[start:end])
                                        for i in range(end-start): 
                                                d.data[start:end][np.where(d.data[start:end] == sortData[i])] = means[i]

        return interDistances


def sortDistsAndInds(indices, distances):
        sortedDistances = []
        sortedIndices = []
        for d, i in zip(distances, indices):
                listd, listi = zip(*sorted(zip(d, i)))
                sortedDistances.append(listd)
                sortedIndices.append(listi)
        return np.asarray(sortedIndices), np.asarray(sortedDistances)


##########################################################################################################
########################################## OLD STUFF #####################################################
##########################################################################################################

def globalNN(Data, index, kFromSame, kFromNeighbors, intraSliceNeighbors, interSliceNeighbors, distanceMetric, indexStart, distCoeff, edgeCase):
        indList = []
        distList = []
        neighborSliceIndex = 0
        currentSlice = Data[index]
        for index2, otherSlice in enumerate(Data):
                if index == index2:
                        knnIndices, distancesTemp = intraSliceNeighbors(kFromSame, currentSlice, currentSlice, index, index2, indexStart, 1)
                        knnDists = distancesTemp.copy()
                else:
                        knnIndices, distancesTemp = interSliceNeighbors(kFromNeighbors, currentSlice, otherSlice, index, index2, neighborSliceIndex, edgeCase)
                indList.append(knnIndices)
                distList.append(distancesTemp)
                neighborSliceIndex+=len(otherSlice)

        sliceIndices = np.column_stack(([l for l in indList if len(l)!=0]))
        sliceDistances = adjustInterDists(currentSlice, index, knnDists, distList, kFromNeighbors, distanceMetric, len(Data), distCoeff, edgeCase)

        return (sliceIndices, sliceDistances)

def adjustInterDists(timeSlice, sliceIndex, knnDists, interDistances, kFromNeighbors, distMetric, dataLen, distCoeff, edgeCase):
        ### Calculate Distances for the metrics which are in relation to intra-distances
                ### 'euclidean' and 'normMeans' calculate inter-slice neighbor distances by manipulating their real euclidean distances
                ###  the other methods estimate inter-slice neighbor distances by using the intra-slice neighbor distances

        if distMetric == 'euclidean':
                adjustedDists = [distCoeff*d if len(d) !=0 else [] for d in interDistances]
        elif distMetric == 'normMeans':
                meanFactor=[np.divide(np.mean(knnDists[:,1:], axis=1), np.mean(d, axis=1)) if len(d)!=0 else 0 for d in interDistances]
                adjustedDists = [(distCoeff)*np.multiply(k.reshape(-1,1),d) if len(d)!=0 else d for k,d in zip(meanFactor, interDistances)]
        elif distMetric == 'max':
                adjustedDists = [np.array([np.full(edgeCase*kFromNeighbors, distCoeff*np.amax(knnDists))]*len(timeSlice))]*dataLen
        elif distMetric == 'median':
                adjustedDists = [np.array([np.full(edgeCase*kFromNeighbors, distCoeff*np.median(knnDists))]*len(timeSlice))]*dataLen
        elif distMetric == '3quartile':
                adjustedDists = [np.array([np.full(edgeCase*kFromNeighbors, distCoeff*np.percentile(knnDists, 75))]*len(timeSlice))]*dataLen
        elif distMetric == 'firstKMeans':
                adjustedDists = [np.array([distCoeff*np.mean(knnDists, axis=1)[1:edgeCase*kFromNeighbors+1]]*len(timeSlice))]*dataLen
        elif distMetric == 'lastKMeans':
                adjustedDists = [np.array([distCoeff*np.mean(knnDists, axis=1)[-(edgeCase*kFromNeighbors):]]*len(timeSlice))]*dataLen

        ### Apply distances to the correct neighbors
        adjustedDists = [n if len(d)!=0 else [] for n, d in zip(adjustedDists, interDistances)]
        sliceDistance = concatIntraInterDistances(adjustedDists, knnDists, sliceIndex)

        return sliceDistance

def concatIntraInterDistances(neighborDists, knnDists, sliceIndex):
        neighborDists[sliceIndex] = knnDists
        distlist = [d for d in neighborDists if (len(d)!=0)]
        sliceDistance = np.column_stack(distlist)
        return sliceDistance

##########################################################################################################
##########################################################################################################

