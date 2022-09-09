import numpy as np
from sklearn.mixture import GaussianMixture
from natto.process import cluster, util
from natto.process.cluster import k2means, hungutil
from natto.out import draw
import matplotlib.pyplot as plt
import copy


def cellDifferentiationPipeline(data, method='minBIC', plotOpt=False, plot=True):
	### Pass in time-series data and this function will return labels for each time series as well as
	### a dictionary for which cells split into different cell types at each time point.
	labels = optimalClusters(data, method=method, plot=plotOpt)
	labels = prelimAlign(labels, data)
	labelsMatchList, simMatrices, xyLabs = basicMatch(labels, data)
	labels, labelPairs = adjustLabels(labels, labelsMatchList, simMatrices, xyLabs)

	if plot:
		clusterCentres = getClusterCentres(labels, data)
		individualPlots(clusterCentres, labelPairs)
		togetherPlot(clusterCentres, labelPairs)
		draw.auto_tiny(data, labels, wrap='test', dim=(1,len(data)), same_limit=True)
	return labels, labelPairs

def optimalClusters(data, max_components=35, method='minBIC', plot=False):
	### Returns the number optimal number of clusters for each time-slice
	### in single-cell time-series data

	dataStacked = np.vstack(data)
	nComponents = range(1, max_components+1)
	clusterMax = max_components
	labels = [None]*len(data)

	for i in range(len(data)-1, -1, -1):
		d = data[i]
		models = [GaussianMixture(n_components=n, covariance_type='full', random_state=1337).fit(dataStacked) 
			for n in nComponents]
		bic = [ m.bic(d) for m in models ]
		bestBICIndex, clusterMax = bestBIC(data, method, nComponents, clusterMax, bic, plot)

		print(f"Best Number of Clusters for slice {i} is {nComponents[bestBICIndex]}")
		labels[i] = models[bestBICIndex].predict(d)
	labels = makeUnique(labels)

	return labels

def bestBIC(data, method, nComponents, clusterMax, bic, plot):
	if method == 'kneed':
		from kneed import KneeLocator
		kn = KneeLocator(range(1, clusterMax+1), bic ,curve='convex', direction='decreasing')
		bestBICIndex = kn.knee-1
		if plot: kn.plot_knee_normalized()
	elif method == 'uber':
		import ubergauss as ug
		bestBICIndex = ug.diag_maxdist(bic)
		if plot:
			plt.scatter(range(1, clusterMax+1), bic)
			plt.show()
	elif method == 'minBIC':
		bic=bic[:clusterMax]
		bestBICIndex = np.argmin(bic)
		clusterMax=bestBICIndex
	elif method == 'minAdjBIC':
		bic=bic[:clusterMax]
		bestBICIndex = np.argmin([b + (np.log(i+1)*(len(data))) for i,b in enumerate(bic)])
		clusterMax=bestBICIndex

	return bestBICIndex, clusterMax

def makeUnique(labels):
	### Makes all labels from different time-slices unique
	for i in range(1, len(labels)): 
		labels[i] = np.asarray([x+max(labels[i-1])+1 for x in labels[i]])
	return labels



def prelimAlign(labels, data):
	clusterCentres = getClusterCentres(labels, data)

	Y2s=[labels[0]]
	for i in range(len(clusterCentres)-1):
		d1, d2 = clusterCentres[i], clusterCentres[i+1]
		y2 = labels[i+1]
		(r,c),distances,_=getHung(d1,d2)
		d1Keys = list(d1.keys())
		d2Keys = list(d2.keys())
		D1Clusters=[d1Keys[r[index]] for index in range(len(r))]
		D2Clusters=[d2Keys[c[index]] for index in range(len(c))]

		transformationDict = {k2:D1Clusters[i] for i, k2 in enumerate(D2Clusters)}
		for k2 in d2.keys():
			if k2 not in transformationDict.keys() and k2 not in D1Clusters:
				transformationDict[k2]=k2
			elif k2 not in transformationDict.keys() and k2 in D1Clusters:
				transformationDict[k2]=max(D2Clusters)+2

		#Actually replaces the values
		newLabels = np.asarray([transformationDict[y] for y in y2])
		Y2s.append(newLabels)

	return Y2s

def getClusterCentres(labels, data):
	### Find centres of each cluster
	clusterCentres = [{} for i in range(len(labels))]
	for i, (l,d) in enumerate(zip(labels, data)):
		for n in np.unique(l):
			dMean = np.mean(d[np.where(l==n)[0]], axis=0)
			clusterCentres[i][n]=list(dMean)
	return clusterCentres

def getHung(d1,d2):
	### Calculates linear sum assignment (Hungarian) for two dictionaries
	X=[np.vstack(list(d.values())) for d in [d1,d2]]
	indices,distances=util.hungarian(X[0],X[1])
	return indices, distances, X



def basicMatch(labels, data, threshold = 0.4):
	### This prevents some clusters from not being assigned
	### to any clusters in the next time-slice
	labelsMatchList=[]
	simMatrices=[]
	xyLabs=[]
	for i in range(len(data)-1):
		hungmatch, distances = util.hungarian(data[i], data[i+1])
		nonZeros=[0]
		while 0 in nonZeros and threshold>0:
			y1map, y2map, canvas = hungutil.make_canvas_and_spacemaps(labels[i], labels[i+1], hungmatch, normalize=True, dist = False)
			row_ind, col_ind = hungutil.solve_dense(canvas)
			canvas, canvasbackup = hungutil.clean_matrix(canvas, threshold=threshold)
			nonZeros=np.count_nonzero(canvas, axis=1)
			threshold-=0.01
		draw.doubleheatmap(canvasbackup, canvas, y1map, y2map, row_ind, col_ind)
	    
		### Get corresponding labels
		sorting = sorted(zip(col_ind, row_ind))
		col_ind, row_ind= list(zip(*sorting))

		# some rows are unused by the matching, but we still want to show them:
		order= list(row_ind) + list(set(y1map.integerlist)-set(row_ind) )
		canvas = canvas[order]
		x0labels = list(y2map.itemlist)
		y0labels = [y1map.getitem[r]for r in order]	    
		x,y=np.nonzero(-canvas)

		simMatrices.append(canvas)
		xyLabs.append((x0labels, y0labels))
	    
		labelsMatch = {y0labels[x[j]]:[] for j in range(len(x))}
		for j in range(len(x)):
			labelsMatch[y0labels[x[j]]].append(x0labels[y[j]])
		labelsMatchList.append(labelsMatch)
	    
		draw.auto_tiny([data[i], data[i+1]], [labels[i], labels[i+1]], wrap='test', dim=(1,2), same_limit=True)
    
	return labelsMatchList, simMatrices, xyLabs


def adjustLabels(lll, labelPairs, simMatrices, xyLabs):
	numTS = len(labelPairs)
	newDict =  [copy.deepcopy(d) for d in labelPairs]
    
	xs = [e[0] for e in xyLabs]
	ys=[e[1] for e in xyLabs]

	### Now iterate
	for i in range(numTS): #for maching between slice t and t+1
		#print(f"On Slice {i} working on slice {i+1}")
		canvas=simMatrices[i]
		x = xs[i]
		y = ys[i]
            
		for key,val in labelPairs[i].items():
			if len(val)>=2:
				if key not in val:
					simScores = canvas[y.index(key),:]
					valToReplace = x[np.argmin(simScores)]
					if isInValues(newDict[i], valToReplace)<2:
						#print(f"replace valToReplace: {valToReplace} w key:{key}")
						newDict[i][key][val.index(valToReplace)]=key
						x = [key if el==valToReplace else el for el in x]
						if i!=numTS-1:
							ys[i+1] = [key if el==valToReplace else el for el in ys[i+1]]
							newDict[i+1] = replace(key, valToReplace, newDict[i+1])
							labelsPairs[i+1] = replace(key, valToReplace, labelPairs[i+1])
						lll[i+1] = [key if j==valToReplace else j for j in lll[i+1]]
                    
			elif len(val)==1 and key!=val[0]:
				### Want to deal with circumstances where cluster goes to one cluster
				### But they have different labels
				if isInValues(newDict[i], key)==0 and isInValues(newDict[i], val[0])==1:
					#print(f"replace val[0]: {val[0]} w key:{key}")
					newDict[i][key]=[key]
					x = [key if el==val[0] else el for el in x]
					lll[i+1] = [key if v==val[0] else v for v in lll[i+1]]
					if i!=numTS-1: #Update next slice labels
						ys[i+1] = [key if el==val[0] else el for el in ys[i+1]]
						newDict[i+1] = replace(key, val[0], newDict[i+1])
						labelsPairs = replace(key, val[0], labelPairs[i+1])

				elif isInValues(newDict[i], key)>0 and isInValues(newDict[i], val[0])==1:
					#print(f"replace key: {key} w val[0]:{val[0]}")
					lll[i] = [val[0] if v==key else v for v in lll[i]]
					y = [val[0] if el==key else el for el in y]
					newDict[i][val[0]] = val
					del newDict[i][key]

	return lll, newDict

def replace(key, toChange, dictObject):
	dictObject[key] = toChange
	del dictObject[toChange]
	return dictObject

def isInValues(dictObject, item):
    appearances=0
    for v in dictObject.values():
        if item in v:
            appearances+=1
    return appearances



def individualPlots(clusterCentres, labelPairs):
	for i in range(len(labelPairs)):
	    d1 = clusterCentres[i]
	    d2 = clusterCentres[i+1]

	    scatter = scatterPoints([d1,d2])
	    for key,val in labelPairs[i].items():
	        for v in val:
	            plt.arrow(d1[key][0], d1[key][1], d2[v][0]-d1[key][0], d2[v][1]-d1[key][1], length_includes_head=True,
	          head_width=0.1, head_length=0.2, color='black')
	    plt.legend(handles=scatter.legend_elements()[0], 
	            labels=[i, i+1],
	           title="Time Point")
	    plt.show()

def togetherPlot(clusterCentres, labelPairs):
	scatter = scatterPoints(clusterCentres)
	for i in range(len(labelPairs)):
	    #print(labelPairs[i])
	    d1=clusterCentres[i]
	    d2=clusterCentres[i+1]
	    for key,val in labelPairs[i].items():
	        for v in val:
	            plt.arrow(d1[key][0], d1[key][1], d2[v][0]-d1[key][0], d2[v][1]-d1[key][1], length_includes_head=True,
	          head_width=0.1, head_length=0.2, color='black')
	plt.legend(handles=scatter.legend_elements()[0], 
	           labels=[i for i in range(len(clusterCentres))],
	           title="Time Point")
	plt.show()

def scatterPoints(clusterCentres):
	col = []
	toShow = np.empty((0,2), dtype=float)
	for i,d in enumerate(clusterCentres):
	    v = np.vstack(list(d.values()))
	    col = col+[i for x in range(len(v))]
	    toShow=np.vstack((toShow,v))
	    scatter = plt.scatter(toShow[:,0], toShow[:,1], c=col, s=60, cmap='spring')
	return scatter


