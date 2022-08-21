import numpy as np
import pandas as pd
import math
import time
from natto import input
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
        Data, intraSliceNeighbors, interSliceNeighbors, indices, distances = initialize(Data, intraSliceNeighbors, interSliceNeighbors, kFromSame, kFromNeighbors)
        prevData = None
        prevIndex = 0
        nextData = Data[1]
        nextIndex = len(Data[0])
        indexStart = 0
        edgeCaseMultiplier = 2 # Equals to 1 or 2 depending on whether we are dealing with an edge dataset.
        sparseSlices = csr_matrix((0, Data[0].shape[0]*len(Data)))

        for index, timeSlice in enumerate(Data):
                ### Find indices and distances for all Neighbors at each 'timeSlice'
                #sliceIndices, sliceDistances = globalNN(Data, 
                sparseSlice = global2NN(Data,
                        index, 
                        kFromSame, 
                        kFromNeighbors, 
                        intraSliceNeighbors, 
                        interSliceNeighbors, 
                        distanceMetric, 
                        indexStart, 
                        distCoeff,
                        edgeCaseMultiplier)
                #indices = np.vstack((indices, sliceIndices))
                #distances = np.vstack((distances, sliceDistances))
                sparseSlices = vstack((sparseSlices, sparseSlice))

                ### Updates variables for next iteration
                prevData = timeSlice
                prevIndex = indexStart
                indexStart = indexStart + len(timeSlice)
                if (index >= len(Data)-2):
                        nextData = None
                        edgeCaseMultiplier = 2

                else:
                        nextData = Data[index+2]
                        nextIndex = indexStart + len(Data[index+1])
                        edgeCaseMultiplier = 1

                if not silent:
                        print(f"Dataset {index} Complete")

        if sort:
                ### Sort neighbors from shortest to longest distance
                distances, indices = sortDistsAndInds(distances, indices)          

        #return indices, distances
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


def initialize(Data, intraSliceNeighbors, interSliceNeighbors, kFromSame, kFromNeighbors):
        if type(Data[0]) != np.ndarray:
                Data = [data.to_df().to_numpy() for data in Data]
        if type(intraSliceNeighbors)==str:
                intraSliceNeighbors = eval(intraSliceNeighbors)
        if interSliceNeighbors is None:
                interSliceNeighbors = intraSliceNeighbors
        elif type(interSliceNeighbors)==str:
                interSliceNeighbors = eval(interSliceNeighbors)
        if interSliceNeighbors.__name__=='hungarianAll':
                distances = np.empty((0, kFromSame+(len(Data)-1)*kFromNeighbors), int)
                indices = np.empty((0, kFromSame+(len(Data)-1)*kFromNeighbors), int)
        else:
                distances = np.empty((0, kFromSame+(kFromNeighbors*2)), int)
                indices = np.empty((0, kFromSame+(kFromNeighbors*2)), int)

        return (Data, intraSliceNeighbors, interSliceNeighbors, indices, distances)


def sklearnNN(kNeighbors, timeSlice1, timeSlice2, index1, index2, indexStart, edgeCase):
        ### Make and fit nearest neighbor object and get k nearest neighbors for the\
        ### current dataset. 
        if np.abs(index1-index2)<=1:
                knnDists, knnInd = NearestNeighbors(n_neighbors=kNeighbors*edgeCase).fit(timeSlice2).kneighbors(timeSlice1)
                knnInd[:,:] = np.add(knnInd[:,:], np.full((knnInd.shape), indexStart)) #Adjust indices
                return (knnInd, knnDists)
        else:
                return (None, None)

def hungarian(kNeighbors, timeSlice1, timeSlice2, index1, index2, indexStart, edgeCase):
        ### Use hungarian algorithm to find k nearest neighbors
        if np.abs(index1-index2)<=1:
                hungDistances = pairwise_distances(timeSlice1, timeSlice2)
                knnInd, knnDists = findSliceNeighbors(timeSlice1, hungDistances, kNeighbors*edgeCase, indexStart)                
        else:
                knnInd, knnDists = (None, None)
        return (knnInd, knnDists)

def hungarianAll(kNeighbors, timeSlice1, timeSlice2, index1, index2, indexStart, adjacentData):
        ### Use hungarian algorithm to find k nearest neighbors between ALL time slices      
        hungDistances = pairwise_distances(timeSlice1, timeSlice2)
        knnInd, knnDists = findSliceNeighbors(timeSlice1, hungDistances, kNeighbors, indexStart)
        return (knnInd, knnDists)


def findSliceNeighbors(data, distances, kNeighbors, indexStart):
        def hungarianLoop(distances):
                rowInd, colInd  = linear_sum_assignment(distances)
                colDist = distances[rowInd, colInd]
                distances[rowInd, colInd] = np.inf
                return colInd, colDist

        colInds, colDists = zip(*[hungarianLoop(distances) for i in range(kNeighbors)])
        return (np.column_stack(colInds)+indexStart, np.column_stack(colDists))

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

        sliceIndices = np.column_stack(([l for l in indList if l is not None]))
        sliceDistances = adjustInterDists(currentSlice, index, knnDists, distList, kFromNeighbors, distanceMetric, len(Data), distCoeff, edgeCase)

        return (sliceIndices, sliceDistances)


def global2NN(Data, index, kFromSame, kFromNeighbors, intraSliceNeighbors, interSliceNeighbors, distanceMetric, indexStart, distCoeff, edgeCase):
        intersectList = []
        neighborSliceIndex = 0
        currentSlice = Data[index]
        for index2, otherSlice in enumerate(Data):
                intersect = csr_matrix((currentSlice.shape[0], otherSlice.shape[0]))
                if index == index2:
                        indicesTemp, distancesTemp = intraSliceNeighbors(kFromSame, currentSlice, currentSlice, index, index2, 0, 1)
                        intraDists = distancesTemp.copy()
                        #for i in range(len(indicesTemp)):
                                #intersect[i, indicesTemp[i,:]] = distancesTemp[i,:]
                        sameIntersect = intersect
                else:
                        ## return entire sparse matrix for 
                        indicesTemp, distancesTemp = interSliceNeighbors(kFromNeighbors, currentSlice, otherSlice, index, index2, 0, edgeCase)
                        intersectList.append(intersect)

                if (indicesTemp is not None):
                        for i in range(len(indicesTemp)):
                                intersect[i, indicesTemp[i,:]] = distancesTemp[i,:]
                #if sameIntersect != intersect:
                        

        slices = adjustSparseDists(intraDists, intersectList, kFromNeighbors, distanceMetric, distCoeff, edgeCase)
        slices.insert(index, sameIntersect)
        print(slices)

        sliceSparseMatrix = hstack(slices)

        return sliceSparseMatrix

def adjustSparseDists(intraDists, interDistances, kFromNeighbors, distMetric, distCoeff, edgeCase):
        ### Calculate Distances for the metrics which are in relation to intra-distances
                ### 'euclidean' and 'normMeans' calculate inter-slice neighbor distances by manipulating their real euclidean distances
                ###  the other methods estimate inter-slice neighbor distances by using the intra-slice neighbor distances

        if distMetric == 'euclidean':
                for d in interDistances: d.multiply(distCoeff)
        elif distMetric == 'normMeans':
                meanFactor=np.array([np.divide(np.mean(intraDists, axis=1), (d.sum(axis=1).reshape(-1,1)[0]/d.getnnz(axis=1))) if d.nnz!=0 else 0 for d in interDistances])
                for k,d in zip(meanFactor,interDistances): d.multiply(distCoeff*k)
        elif distMetric == 'max':
                for d in interDistances: d.data=np.array([distCoeff*np.amax(intraDists)]*len(d.data))
        elif distMetric == 'median':
                for d in interDistances: d.data=np.array([distCoeff*np.median(intraDists)]*len(d.data))
        elif distMetric == '3quartile':
                for d in interDistances: d.data=np.array([distCoeff*np.percentile(intraDists, 75)]*len(d.data))
        elif 'KMeans' in distMetric:
                if distMetric == 'firstKMeans':
                        means = distCoeff*np.mean(intraDists, axis=1)[1:edgeCase*kFromNeighbors+1]
                elif distMetric == 'lastKMeans':
                        means = distCoeff*np.mean(intraDists, axis=1)[-(edgeCase*kFromNeighbors):]
                for d in interDistances:
                        if d.nnz!=0:
                                for ip in range(len(d.indptr)-1):
                                        start, end = d.indptr[ip:ip+2]
                                        sortData = sorted(d.data[start:end])
                                        tempData = d.data[start:end]
                                        for i in range(len(tempData)): 
                                                d.data[start:end][np.where(d.data[start:end] == sortData[i])] = means[i]
                                        #d.data[start:end] = tempData
        '''
        elif distMetric == 'lastKMeans':
                means = distCoeff*np.mean(intraDists, axis=1)[-(edgeCase*kFromNeighbors):]
                for d in interDistances:
                        if d.nnz!=0:
                                for ip in range(len(d.indptr)-1):
                                        start, end = d.indptr[ip:ip+2]
                                        sortData = sorted(d.data[start:end])
                                        tempData = d.data[start:end]
                                        for i in range(len(tempData)):
                                                #tempData[np.where(d.data[start:end] == sortData[i])] = means[i]
                                                d.data[start:end][np.where(d.data[start:end] == sortData[i])] = means[i]
                                        d.data[start:end] = tempData
        '''

        return interDistances

def adjustInterDists(timeSlice, sliceIndex, knnDists, interDistances, kFromNeighbors, distMetric, dataLen, distCoeff, edgeCase):
        ### Calculate Distances for the metrics which are in relation to intra-distances
                ### 'euclidean' and 'normMeans' calculate inter-slice neighbor distances by manipulating their real euclidean distances
                ###  the other methods estimate inter-slice neighbor distances by using the intra-slice neighbor distances

        if distMetric == 'euclidean':
                adjustedDists = [distCoeff*d if d is not None else None for d in interDistances]
        elif distMetric == 'normMeans':
                meanFactor=[np.divide(np.mean(knnDists[:,1:], axis=1), np.mean(d, axis=1)) if d is not None else 0 for d in interDistances]
                adjustedDists = [(distCoeff)*np.multiply(k.reshape(-1,1),d) if d is not None else d for k,d in zip(meanFactor, interDistances)]
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
        adjustedDists = [n if d is not None else None for n, d in zip(adjustedDists, interDistances)]
        sliceDistance = concatIntraInterDistances(adjustedDists, knnDists, sliceIndex)

        return sliceDistance
  

def concatIntraInterDistances(neighborDists, knnDists, sliceIndex):
        neighborDists[sliceIndex] = knnDists
        distlist = [d for d in neighborDists if (d is not None)]
        sliceDistance = np.column_stack(distlist)
        return sliceDistance


def sortDistsAndInds(distances, indices):
        sortedDistances = []
        sortedIndices = []
        for d, i in zip(distances, indices):
                listd, listi = zip(*sorted(zip(d, i)))
                sortedDistances.append(listd)
                sortedIndices.append(listi)
        return np.asarray(sortedDistances), np.asarray(sortedIndices)

