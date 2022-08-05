import numpy as np
import pandas as pd
import math
import time
from natto import input
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics import pairwise_distances
from scipy.optimize import linear_sum_assignment
#from sklearn import neighbors as nbrs, metrics


def __init__():
        pass

def timeSliceNearestNeighbor(Data, 
        kFromNeighbors=6, 
        kFromSame=16, 
        sort=True,
        intraSliceNeighbors="sklearnNN",
        interSliceNeighbors=None,
        distanceMetric='max', 
        distCoeff=0,
        verbose=False):

        ### Initialize variables
        Data, intraSliceNeighbors, interSliceNeighbors, indices, distances = initialize(Data, intraSliceNeighbors, interSliceNeighbors, kFromSame, kFromNeighbors)
        prevData = None
        prevIndex = 0
        nextData = Data[1]
        nextIndex = len(Data[0])
        indexStart = 0
        adjacentData = 1 # Equals to 1 or 2 depending on whether we are dealing with an edge dataset.

        
        for index, timeSlice in enumerate(Data):
                ### Find indices and distances for all Neighbors at each 'timeSlice'
                sliceIndices, sliceDistances = globalNN(Data, 
                        index, 
                        kFromSame, 
                        kFromNeighbors, 
                        intraSliceNeighbors, 
                        interSliceNeighbors, 
                        distanceMetric, 
                        indexStart, 
                        distCoeff, 
                        adjacentData)
                indices = np.vstack((indices, sliceIndices))
                distances = np.vstack((distances, sliceDistances))

                ### Updates variables for next iteration
                prevData = timeSlice
                prevIndex = indexStart
                indexStart = indexStart + len(timeSlice)
                if index >= len(Data)-2:
                        nextData = None
                        adjacentData = 1
                else:
                        nextData = Data[index+2]
                        nextIndex = indexStart + len(Data[index+1])
                        adjacentData = 2

                print(f"Dataset {index} Complete")

        if sort:
                ### Sort neighbors from shortest to longest distance
                distances, indices = sortDistsAndInds(distances, indices)          

        if verbose:
                print((distances))
                print(f"Length: {len(distances)}")
                print(indices)
                print(f"\nShape: ({len(indices)}, {len(indices[0])})\n")

        return indices, distances


def sklearnNN(kNeighbors, timeSlice1, index1, index2, indexStart, adjacentData, timeSlice2=None):
        ### Make and fit nearest neighbor object and get k nearest neighbors for the\
        ### current dataset. 
        
        if timeSlice2 is None:
                knnDists, knnInd = NearestNeighbors(n_neighbors=kNeighbors).fit(timeSlice1).kneighbors(timeSlice1)
                knnInd[:,:] = np.add(knnInd[:,:], np.full((knnInd.shape), indexStart)) #Adjust indices
                return (knnInd, knnDists)
        elif np.abs(index1-index2)<=1:
                knnDists, knnInd = NearestNeighbors(n_neighbors=kNeighbors//adjacentData).fit(timeSlice2).kneighbors(timeSlice1)
                knnInd[:,:] = np.add(knnInd[:,:], np.full((knnInd.shape), indexStart)) #Adjust indices
                return knnInd
        else:
                return None

def hungarian(kNeighbors, timeSlice1, index1, index2, indexStart, adjacentData, timeSlice2=None):
        ### Use hungarian algorithm to find k nearest neighbors

        if timeSlice2 is None:
                hungDistances = pairwise_distances(timeSlice1, timeSlice2)
                knnInd, knnDists = findSliceNeighbors(timeSlice1, hungDistances, kNeighbors, indexStart)
                return (knnInd, knnDists)
                        
        elif np.abs(index1-index2)<=1:
                hungDistances = pairwise_distances(timeSlice1, timeSlice2)
                knnInd, knnDists = findSliceNeighbors(timeSlice1, hungDistances, kNeighbors//adjacentData, indexStart)
                return knnInd
        else:
                return None


def hungarianAll(kNeighbors, timeSlice1, index1, index2, indexStart, adjacentData, timeSlice2=None):
        ### Use hungarian algorithm to find k nearest neighbors between ALL time slices      

        if timeSlice2 is None:
                hungDistances = pairwise_distances(timeSlice1, timeSlice1)
                knnInd, knnDists = findSliceNeighbors(timeSlice1, hungDistances, kNeighbors, indexStart)
                return (knnInd, knnDists)
        else:
                hungDistances = pairwise_distances(timeSlice1, timeSlice2)
                knnInd, knnDists = findSliceNeighbors(timeSlice1, hungDistances, kNeighbors, indexStart)
                return knnInd


def globalNN(Data, index, kFromSame, kFromNeighbors, intraSliceNeighbors, interSliceNeighbors, distanceMetric, indexStart, distCoeff, adjacentData):
        indList = []
        neighborSliceIndex = 0
        for index2, timeSlice2 in enumerate(Data):
                if index == index2:
                        knnIndices, knnDists = intraSliceNeighbors(kFromSame, Data[index], index, index2, indexStart, adjacentData)
                        indList.append(knnIndices)
                else:
                        knnIndices = interSliceNeighbors(kFromNeighbors, Data[index], index, index2, neighborSliceIndex, adjacentData, timeSlice2=timeSlice2)
                        if knnIndices is not None:
                                        indList.append(knnIndices)
                neighborSliceIndex+=len(timeSlice2)

        sliceIndices = np.column_stack(([l for l in indList]))
        sliceDistances = calcInterNeighborDists(Data[index], index, knnDists, kFromNeighbors, interSliceNeighbors.__name__, distanceMetric, len(Data), distCoeff)

        return (sliceIndices, sliceDistances)


def findSliceNeighbors(data, distances, kNeighbors, indexStart):
        knnInd = np.empty((data.shape[0], 0), int)
        knnDists = np.empty((data.shape[0], 0), int)

        ### Gets 'k_neighbors' nearest neighbors from within the same timeslice
        i = 0
        while i < kNeighbors:
                rowInd, colInd = linear_sum_assignment(distances)
                knnDists = np.column_stack((knnDists, distances[rowInd, colInd]))
                distances, knnInd = calcNeighborIndices(distances, rowInd, colInd, knnInd, indexStart)

                i += 1   

        return knnInd, knnDists   



def calcNeighborIndices(distances, rowInd, colInd, knnInd, indexStart):
        distances[rowInd, colInd] = distances[rowInd, colInd] + np.inf
        colInd = colInd + indexStart
        knnInd = np.column_stack((knnInd, colInd))

        return distances, knnInd


def calcInterNeighborDists(timeSlice, sliceIndex, knnDists, kFromNeighbors, nnMetric, distMetric, dataLen, distCoeff):
        if nnMetric=='hungarianAll':
                coeff = dataLen-1
        else:
                coeff = 1

        ### Calculate Distances
        if distMetric == 'max':
                neighborDists = np.tile(np.full(coeff*kFromNeighbors, np.amax(knnDists)), (len(timeSlice),1))
        elif distMetric == 'median':
                neighborDists = np.tile(np.full(coeff*kFromNeighbors, np.median(knnDists)), (len(timeSlice),1))
        elif distMetric == '3quartile':
                neighborDists = np.tile(np.full(coeff*kFromNeighbors, np.percentile(knnDists, 75)), (len(timeSlice),1))
        elif distMetric == 'firstKMeans':
                neighborDists = np.tile(np.mean(knnDists, axis=1)[1:coeff*(kFromNeighbors)+1], (len(timeSlice),1))
        elif distMetric == 'lastKMeans':
                neighborDists = np.tile(
                        np.mean(knnDists, axis=1)[knnDists.shape[1]-coeff*(kFromNeighbors)-1:knnDists.shape[1]-1], 
                        (len(timeSlice),1))

        ### Apply distances to the correct neighbors
        sliceDistance = applyDistances(timeSlice, neighborDists, nnMetric, kFromNeighbors, sliceIndex, knnDists, dataLen, distCoeff)

        return sliceDistance
  

def applyDistances(timeSlice, neighborDists, nnMetric, kFromNeighbors, sliceIndex, knnDists, dataLen, distCoeff):
        if nnMetric == "hungarianAll":
                distlist = [neighborDists[:, 0:(kFromNeighbors)] if x!= sliceIndex else knnDists for x in range(dataLen)]
                if distCoeff!=0:
                        for i, Slice in enumerate(distlist):
                                distlist[i] = Slice * (distCoeff*(np.abs(sliceIndex-i))+1)
                sliceDistance = np.column_stack(distlist)
        elif sliceIndex!=0 and sliceIndex!=dataLen-1:
                sliceDistance = np.column_stack([neighborDists[:, 0:(kFromNeighbors//2)], 
                        knnDists,
                        neighborDists[:, 0:(kFromNeighbors//2)]])
        else:
                if sliceIndex == dataLen-1:
                        sliceDistance = np.column_stack([neighborDists, knnDists])
                elif sliceIndex == 0:
                        sliceDistance = np.column_stack([knnDists, neighborDists])
        return sliceDistance


def initialize(Data, intraSliceNeighbors, interSliceNeighbors, kFromSame, kFromNeighbors):

        if len(Data) <= 2: assert("More data plz")
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
                distances = np.empty((0, kFromSame+kFromNeighbors), int)
                indices = np.empty((0, kFromSame+kFromNeighbors), int)
        return (Data, intraSliceNeighbors, interSliceNeighbors, indices, distances)


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

