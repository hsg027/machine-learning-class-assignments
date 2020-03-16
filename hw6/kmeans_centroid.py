def kmeans_centroid(data, center, k=2, metric='manhattan', max_iter=100):
    np.random.seed(99)

    if center is None:
        center = data[np.random.choice(len(data), size=k, replace=False)]
    center_old = np.ones(center.shape)

    clusters = np.zeros(len(data))

    e = get_distance(center, center_old, metric, None)
    e = np.array(e)
    count = 1
    prev_sse = 0
    curr_sse = 0

    while e.any() != 0:

        for i in range(len(data)):
            distances = get_distance(data[i], center, metric)
            cluster = np.argmin([distances])
            clusters[i] = cluster

        center_old = deepcopy(center)
        curr_sse = sse(data, clusters, center)
        print('Iteration: {}'.format(count))
        print('Current SSE: {}'.format(curr_sse))
        print('Previous SSE: {}'.format(prev_sse))
        for i in range(k):
            points = [data[j] for j in range(len(data)) if clusters[j] == i]
            center[i] = np.mean(points, axis=0)

        error_old = deepcopy(e)
        e = get_distance(center, center_old, metric, None)
        # if count > 0:
        #     if np.sum(error_old) == np.sum(e):
        #         break
        #
        # if count > max_iter + 1:
        #     break

        count = count + 1
        prev_sse = curr_sse
    return clusters, count, center, curr_sse
