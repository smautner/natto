
import numpy as np
from natto.out import draw
from lmz import Map, Zip
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics import pairwise_distances
from scipy.optimize import linear_sum_assignment
import warnings
from scipy.sparse import csr_matrix, lil_matrix, hstack, vstack, SparseEfficiencyWarning

import warnings
warnings.filterwarnings("ignore", category=SparseEfficiencyWarning)

def __init__():
        pass


def timeSliceNearestNeighbor(Data, #list of numpy arrays
        kFromNeighbors=2, 
        kFromSame=10, 
        sort=True,
        intraSliceNeighbors="sklearnNN",
        interSliceNeighbors=None, #Suggest using 'hungarian'
        distanceMetric='normMeans', 
        dijkstra=False,
        distCoeff=1,
        farPenalty=1,
        returnSparse=False, # False returns UMAP compatible indicess and distances
        returnSortedBySlice=False,
        silent=False):

        ### Initialize variables
        Data, intraSliceNeighbors, interSliceNeighbors, sparseSlices = initialize(Data, intraSliceNeighbors, interSliceNeighbors)
        edgeCaseMultiplier = 2 # Equals to 1 or 2 depending on whether we are dealing with an 'edge' dataset.

        for index, timeSlice in enumerate(Data):
                ### Find indices and distances for all Neighbors at each 'timeSlice'
                sparseSlice = globalNN(Data,
                        index, 
                        kFromSame, 
                        kFromNeighbors, 
                        intraSliceNeighbors, 
                        interSliceNeighbors, 
                        distanceMetric,
                        distCoeff,
                        farPenalty,
                        edgeCaseMultiplier)
                sparseSlices = vstack((sparseSlices, sparseSlice))

                ### Updates variables for next iteration
                if (index >= len(Data)-2):
                        edgeCaseMultiplier = 2
                else:
                        edgeCaseMultiplier = 1

                if not silent:
                        print(f"Dataset {index} Complete")
        #print(sparseSlices)

        ### Prepare for Return
        if dijkstra==True:
                from scipy.sparse.csgraph import dijkstra
                sparseSlices = dijkstra(sparseSlices, directed=False)

        #sparseMatrix = sparseSlices.tocsr()
        sparseMatrix = csr_matrix(sparseSlices)
        #print(sparseMatrix)

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

                numNeighbors = max(sparseMatrix.getnnz(axis=1))-kFromSame
                if (min(sparseMatrix.getnnz(axis=1))-kFromSame != numNeighbors):
                        precomputedKNNIndices, precomputedKNNDistances = adjustUnevenMatrices(precomputedKNNIndices, precomputedKNNDistances, numNeighbors, kFromSame)

                if sort:
                        return sortDistsAndInds(np.vstack(precomputedKNNIndices), np.vstack(precomputedKNNDistances))
                else:
                        return (np.vstack(precomputedKNNIndices), np.vstack(precomputedKNNDistances))


def initialize(Data, intraSliceNeighbors, interSliceNeighbors):
        if type(Data[0]) != np.ndarray:
                Data = [data.to_df().to_numpy() for data in Data]
        if type(intraSliceNeighbors)==str:
                intraSliceNeighbors = eval(intraSliceNeighbors)
        if interSliceNeighbors is None:
                interSliceNeighbors = intraSliceNeighbors
        elif type(interSliceNeighbors)==str:
                interSliceNeighbors = eval(interSliceNeighbors)
        sparseSlices = csr_matrix((0, sum([d.shape[0] for d in Data])))

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


def globalNN(Data, index, kFromSame, kFromNeighbors, intraSliceNeighbors, interSliceNeighbors, distanceMetric, distCoeff, farPenalty, edgeCase):
        intersectList = []
        neighborSliceIndex = 0
        currentSlice = Data[index]
        for index2, otherSlice in enumerate(Data):  

                if currentSlice.shape[0]<=otherSlice.shape[0]: 
                        slice1=currentSlice
                        slice2=otherSlice
                else:
                        slice1=otherSlice
                        slice2=currentSlice
                intersect = csr_matrix((slice1.shape[0], slice2.shape[0]))

                if index == index2:
                        indicesTemp, distancesTemp = intraSliceNeighbors(kFromSame, currentSlice, currentSlice, index, index2, 1)
                        intraDists = distancesTemp.copy()
                        sameIntersect = intersect
                else:
                        indicesTemp, distancesTemp = interSliceNeighbors(kFromNeighbors, slice1, slice2, index, index2, edgeCase)
                        intersectList.append(intersect)

                for i in range(len(indicesTemp)): intersect[i, indicesTemp[i,:]] = distancesTemp[i,:]

                if currentSlice.shape[0]>otherSlice.shape[0]:
                        ### If slices were uneven shape
                        intersectList[-1]=intersectList[-1].T

        slices = adjustSparseDists(intraDists, intersectList, kFromNeighbors, distanceMetric, distCoeff, edgeCase)
        #slices = [s * (farPenalty**np.abs(index-i)) for i,s in enumerate(slices)] #Make further time-slices more distant
        slices.insert(index, sameIntersect)

        for i in range(0,index): slices[i]=slices[i].multiply(farPenalty**np.abs(index-i-1))
        for i in range(index+1,len(slices)): slices[i]=slices[i].multiply(farPenalty**np.abs(index-i+1))
        
        sliceSparseMatrix = hstack(slices)

        return sliceSparseMatrix

def adjustSparseDists(intraDists, interDistances, kFromNeighbors, distMetric, distCoeff, edgeCase):
        ### Calculate Distances for the metrics which are in relation to intra-distances
                ### 'euclidean' and 'normMeans' calculate inter-slice neighbor distances by manipulating their real euclidean distances
                ###  the other methods estimate inter-slice neighbor distances by using the intra-slice neighbor distances

        if distMetric == 'euclidean':
                for i,d in enumerate(interDistances): interDistances[i].multiply(distCoeff)
        elif distMetric == 'normMeans':
                meanFactor=np.array( [np.divide( np.mean(intraDists[:,1:], axis=1), (d.sum(axis=1).getA1()/d.getnnz(axis=1) ) ) 
                        if d.nnz!=0 else np.ones(d.shape[0]) for d in interDistances])
                for k,di in zip(meanFactor,range(len(interDistances))): 
                        interDistances[di] = interDistances[di].multiply(distCoeff*k.reshape(-1,1))
                        #print(interDistances[di])
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
                for i, d in enumerate(interDistances):
                        if d.nnz!=0:
                                for ip in range(len(d.indptr)-1):
                                        start, end = d.indptr[ip:ip+2]
                                        sortData = sorted(d.data[start:end])
                                        for j in range((end-start)):
                                                d.data[start:end][np.where(d.data[start:end] == sortData[j])] = means[j]
        elif distMetric == 'dijkstra':
                for i,d in enumerate(interDistances):
                        interDistances[i].multiply(distCoeff)

        return interDistances


def sortDistsAndInds(indices, distances):
        sortedIndices = []
        sortedDistances = []
        for d, i in zip(distances, indices):
                listd, listi = zip(*sorted(zip(d, i)))
                sortedDistances.append(listd)
                sortedIndices.append(listi)
        return np.asarray(sortedIndices), np.asarray(sortedDistances)


def adjustUnevenMatrices(indices, distances, numNeighbors, kFromSame):
        evenIndices = []
        evenDistances = []
        for indRow, distRow in zip(indices, distances):
                d, i = zip(*sorted(zip(distRow, indRow)))
                for j in range(numNeighbors - len(indRow) + kFromSame):
                        indRow = np.append(indRow, i[-1])
                        distRow = np.append(distRow, d[-1])
                evenIndices.append(indRow)
                evenDistances.append(distRow)
        return evenIndices, evenDistances


def KNNFormater(Data, precomputedKNNIndices, precomputedKNNDistances, silent=False):
        from pynndescent import NNDescent

        if not silent: print("Computing NNDescent Object")
        pyNNDobject = NNDescent(np.vstack(Data), metric='euclidean', random_state=1337)
        pyNNDobject._neighbor_graph = (precomputedKNNIndices.copy(), precomputedKNNDistances.copy())
        precomputedKNN = (precomputedKNNIndices, precomputedKNNDistances, pyNNDobject)

        return precomputedKNN

def transformer(Data, precomputedKNNIndices=None, precomputedKNNDistances=None, usePrecomputed=False, proj='UMAP', n_components=2, silent=False):
        if usePrecomputed==False:
                precomputedKNN = (None, None, None)
                n_neighbors = 15
        else:
                precomputedKNN = KNNFormater(Data, precomputedKNNIndices, precomputedKNNDistances, silent)
                n_neighbors = precomputedKNN[0].shape[1]

        if not silent: print('Beginning Projection')
        if proj=='UMAP':
                from umap.umap_ import UMAP
                mymap = UMAP(n_components=n_components, #Dimensions to reduce to
                        n_neighbors=n_neighbors,
                        random_state=1337,
                        metric='euclidean',
                        precomputed_knn=precomputedKNN,
                        force_approximation_algorithm=True)
                mymap.fit(np.vstack(Data))

                if not silent: print("Transforming Data")
                transformedData = [mymap.transform(x) for x in Data]
        elif proj=='TSNE':
                from sklearn.manifold import TSNE
                mymap = TSNE(n_components=n_components,
                        random_state=1337,
                        metric='euclidean',
                        )
                transformedDataStack = mymap.fit_transform(np.vstack(Data)) #TSNE has no transform function
                transformedData = unstack(transformedDataStack, Data)
        elif proj=='LLE':
                from sklearn.manifold import LocallyLinearEmbedding as LLE
                mymap = LLE(n_components=n_components,
                        n_neighbors=8,
                        random_state=1337,
                        method='hessian',
                        eigen_solver='dense'
                        )
                mymap.fit(np.vstack(Data))
                if not silent: print("Transforming Data")
                transformedData = [mymap.transform(x) for x in Data]
        
        return transformedData


def unstack(stackedData, listOfData):
        shapes = [d.shape[0] for d in listOfData]
        for i in range(len(shapes)-1): shapes[i+1] = shapes[i+1] + shapes[i]
        return np.split(stackedData, shapes)[:-1]


def evalProjection(Data, Labels, method='KNeighbors', heatmaps=False, n_neighbors=1):
        if method=='KNeighbors':
                results = kNeighborEval(Data, Labels, n_neighbors, heatmaps)
        elif method=='tripleScore':
                results = tripleScoreEval(Data, Labels)
        elif method=='spearman':
                results = spearman(Data, Labels)
        return np.mean(results)


def tripleScoreEval(Data, Labels):
        from sklearn.metrics import adjusted_rand_score, silhouette_score, normalized_mutual_info_score
        from sklearn.cluster import KMeans
        results = []
        for d,l in zip(Data, Labels):
                kClusters  = len(np.unique(l))
                kmeans = KMeans(n_clusters=kClusters, n_init=50).fit(d)
                pred_l = kmeans.predict(d)

                ari = adjusted_rand_score(l, pred_l)
                nmi = normalized_mutual_info_score(l, pred_l)
                sil = silhouette_score(d, pred_l)
                results.append( (ari + nmi + sil)/3 )

        return results


def kNeighborEval(Data, Labels, n_neighbors=1, heatmaps=False):
        from sklearn.neighbors import KNeighborsClassifier

        results = []
        for i in range(len(Data)-1):
                model = KNeighborsClassifier(n_neighbors=n_neighbors).fit(Data[i], Labels[i])
                result = model.predict(Data[i+1])
                simMatrix = measureClusterSimilarity(result, Labels[i+1])
                diagonals = [simMatrix[j,j] for j in range(min(simMatrix.shape))]
                if heatmaps:
                        draw.simpleheatmap(simMatrix)
                results.append(np.mean(diagonals))
        return results


def measureClusterSimilarity(predLabels, trueLabels):
        matrix = np.empty((0,len(np.unique(trueLabels))))
        for v1 in np.unique(predLabels):
                row = []
                for v2 in np.unique(trueLabels):
                        v1Indices = np.where(predLabels==v1)[0]
                        v2Indices = np.where(trueLabels==v2)[0]
                        instances = len([1 for item in v1Indices if item in v2Indices])
                        success = instances / len(v2Indices)
                        row.append(success)

                matrix = np.vstack((matrix, row)) 

        #if matrix.shape[0]!=matrix.shape[1]:
                #print("Uneven Matrix?")
                #print(matrix.shape)

        return matrix

def spearman(Data, Labels):
        from scipy.stats import spearmanr
        from sklearn.metrics.pairwise import euclidean_distances as ed
        results = []
        for i in range(len(Data)-1):
                centres1 = getcenters(Data[i], Labels[i])
                centres2 = getcenters(Data[i+1], Labels[i+1])
                matrix = ed(centres1, centres2)
                spear = spearmanr(matrix)
                results.append(np.mean(spear))
        return results

def centers(arg):
    X,y = arg
    cents = []
    for i in np.unique(y):
        m = X[y==i].mean(axis=0)
        cents.append(np.hstack([i,m]))
    return np.array(cents)

def getcenters(xx,yy):
    return np.vstack(Map(centers,Zip(xx,yy)))

##########################################################################################################
########################################## OLD STUFF #####################################################
##########################################################################################################


def timeSliceNearestNeighborOG(Data, 
        kFromNeighbors=6, 
        kFromSame=16, 
        sort=True,
        intraSliceNeighbors="sklearnNN",
        interSliceNeighbors=None,
        distanceMetric='max', 
        distCoeff=1,
        silent=False):

        ### Initialize variables
        Data, intraSliceNeighbors, interSliceNeighbors, indices, distances = initializeOG(Data, intraSliceNeighbors, interSliceNeighbors, kFromSame, kFromNeighbors)
        prevData = None
        prevIndex = 0
        nextData = Data[1]
        nextIndex = len(Data[0])
        indexStart = 0
        edgeCaseMultiplier = 2 # Equals to 1 or 2 depending on whether we are dealing with an edge dataset.


        for index, timeSlice in enumerate(Data):
                ### Find indices and distances for all Neighbors at each 'timeSlice'
                sliceIndices, sliceDistances = globalNNOG(Data, 
                        index, 
                        kFromSame, 
                        kFromNeighbors, 
                        intraSliceNeighbors, 
                        interSliceNeighbors, 
                        distanceMetric, 
                        indexStart, 
                        distCoeff,
                        edgeCaseMultiplier)
                indices = np.vstack((indices, sliceIndices))
                distances = np.vstack((distances, sliceDistances))

                ### Updates variables for next iteration
                prevData = timeSlice
                prevIndex = indexStart
                indexStart = indexStart + len(timeSlice)
                if index >= len(Data)-2:
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
                indices, distances = sortDistsAndInds(indices, distances)          
        return indices, distances


def initializeOG(Data, intraSliceNeighbors, interSliceNeighbors, kFromSame, kFromNeighbors):
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

def sklearnNNOG(kNeighbors, timeSlice1, timeSlice2, index1, index2, indexStart, edgeCase):
        ### Make and fit nearest neighbor object and get k nearest neighbors for the\
        ### current dataset. 
        if np.abs(index1-index2)<=1:
                knnDists, knnInd = NearestNeighbors(n_neighbors=kNeighbors*edgeCase).fit(timeSlice2).kneighbors(timeSlice1)
                knnInd[:,:] = np.add(knnInd[:,:], np.full((knnInd.shape), indexStart)) #Adjust indices
                return (knnInd, knnDists)
        else:
                return ([], [])

def hungarianOG(kNeighbors, timeSlice1, timeSlice2, index1, index2, indexStart, edgeCase):
        ### Use hungarian algorithm to find k nearest neighbors
        if np.abs(index1-index2)<=1:
                hungDistances = pairwise_distances(timeSlice1, timeSlice2)
                knnInd, knnDists = findSliceNeighborsOG(timeSlice1, hungDistances, kNeighbors*edgeCase, indexStart)                
        else:
                knnInd, knnDists = ([], [])
        return (knnInd, knnDists)

def hungarianAllOG(kNeighbors, timeSlice1, timeSlice2, index1, index2, indexStart, adjacentData):
        ### Use hungarian algorithm to find k nearest neighbors between ALL time slices      
        hungDistances = pairwise_distances(timeSlice1, timeSlice2)
        knnInd, knnDists = findSliceNeighborsOG(timeSlice1, hungDistances, kNeighbors, indexStart)
        return (knnInd, knnDists)


def findSliceNeighborsOG(data, distances, kNeighbors, indexStart):
        def hungarianLoop(distances):
                rowInd, colInd  = linear_sum_assignment(distances)
                colDist = distances[rowInd, colInd]
                distances[rowInd, colInd] = np.inf
                return colInd, colDist

        colInds, colDists = zip(*[hungarianLoop(distances) for i in range(kNeighbors)])
        return (np.column_stack(colInds)+indexStart, np.column_stack(colDists))


def globalNNOG(Data, index, kFromSame, kFromNeighbors, intraSliceNeighbors, interSliceNeighbors, distanceMetric, indexStart, distCoeff, edgeCase):
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

