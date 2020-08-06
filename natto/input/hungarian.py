def hungarian(X1, X2, solver='scipy', debug = False):
    # get the matches:
    distances = ed(X1,X2)
    if solver != 'scipy':
        row_ind, col_ind = linear_sum_assignment(distances)
    else:
        row_ind,col_ind = solve_dense(distances)

    if debug:
        x = distances[row_ind, col_ind]
        num_bins = 100
        print("hungarian: debug hist")
        plt.hist(x, num_bins, facecolor='blue', alpha=0.5)
        plt.show()
    return row_ind, col_ind, distances