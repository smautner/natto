from sklearn.metrics import pairwise_distances
from lapsolver import solve_dense
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances as ed

def hungarian(X1, X2, debug = False,metric='euclidean'):
    # get the matches:
    distances = pairwise_distances(X1,X2, metric=metric)
    #distances = ed(X1, X2)

    #if solver != 'scipy':
    #    row_ind, col_ind = linear_sum_assignment(distances)
    row_ind,col_ind = solve_dense(distances)

    if debug:
        x = distances[row_ind, col_ind]
        num_bins = 100
        print("hungarian: debug hist")
        plt.hist(x, num_bins, facecolor='blue', alpha=0.5)
        plt.show()

    return (row_ind, col_ind), distances



def hungsort(X1,X2): 
    (a,b),_ = hungarian(X1,X2) 
    return X2[b]
