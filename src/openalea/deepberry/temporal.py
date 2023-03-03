""" Time-lapse tracking of berry center coordinates """

import numpy as np

# https://github.com/siavashk/pycpd
from pycpd import AffineRegistration, RigidRegistration, DeformableRegistration


def point_sets_registration(set1, set2, transformation='affine'):
    """
    Registration of two 2D point sets using the Coherent Point Drift algorithm, and coordinates scaling.

    Parameters
    ----------
    set1 : (n, 2) array
        contains the x,y coordinates of the reference/fixed point cloud.
    set2 : (m, 2) array
        contains the x,y coordinates of the moving point cloud.
    transformation : float
        define the type of transformation used for registration, among {'rigid', 'affine', 'deformable'}.

    Returns
    -------
    (m, 2) array
        The resulting deformation of set2.
    """

    # registration requires at least 3 points in each point-set
    if (len(set1) <= 2) or (len(set2) <= 2):
        return set2

    # scaling of X and Y
    q, m = 0.5 * np.max(np.max(set1, axis=0) - np.min(set1, axis=0)), np.mean(set1, axis=0)
    set1_scaled, set2_scaled = (set1 - m) / q, (set2 - m) / q

    # point set registration (Coherent Point Drift)
    reg_functions = {'rigid': RigidRegistration, 'affine': AffineRegistration, 'deformable': DeformableRegistration}
    set2_scaled_reg = reg_functions[transformation](**{'X': set1_scaled, 'Y': set2_scaled}).register()[0]

    # return set2 after reversing scaling
    return (set2_scaled_reg * q) + m


def point_sets_distance(set1, set2):
    """ Metric quantifying the distance between two 2D point sets. """
    m = np.zeros((len(set1), len(set2)))
    for k, c1 in enumerate(set1):
        m[k] = np.sqrt(np.sum((c1 - set2) ** 2, axis=1))
    return 0.5 * (np.median(np.min(m, axis=0)) + np.median(np.min(m, axis=1)))


def point_sets_matching(set1, set2, threshold=float('inf')):
    """
    Greedy bipartite matching of two 2D point sets, with a distance threshold.

    Parameters
    ----------
    set1 : (n, 2) array
        contains the x,y coordinates of the reference/fixed point cloud.
    set2 : (m, 2) array
        contains the x,y coordinates of the moving point cloud.
    threshold : float
        two points can only be paired if their euclidean distance is under this threshold.

    Returns
    -------
    (l, 2) array
        array of index matches between both sets.
    """

    # matrix of the euclidean distances between points of both sets
    b = np.zeros((len(set1), len(set2)))
    for i, c1 in enumerate(set1):
        b[i] = np.sqrt(np.sum((c1 - set2) ** 2, axis=1))

    # greedy matching
    matches = []
    for k in range(min(b.shape)):
        i_min, j_min = np.unravel_index(b.argmin(), b.shape)
        d = b[i_min, j_min]
        if d < threshold:
            matches.append([i_min, j_min])
        b[i_min, :] = float('inf')
        b[:, j_min] = float('inf')
    return np.array(matches)


def distance_matrix(sets):
    """
    Parameters
    ----------
    sets : list
        a list of n 2D point sets.

    Returns
    -------
    (n, n, 3) array
        A distance matrix M, with M[i, j] quantifying the distance between sets[i] to sets[j]. M[i, j, 0] is the raw
        distance without registration. M[i, j, 1] is computed after a registration with reference i. M[i, j, 2] is
        computed after a registration with reference j. M[i, i] values are arbitrarily set to inf.
    """

    M = np.zeros((len(sets), len(sets), 3))
    M[np.arange(len(M)), np.arange(len(M))] = float('inf')  # diagonal

    for i1 in range(len(M) - 1):
        for i2 in np.arange(i1 + 1, len(M)):

            # point-set registration
            set1, set2 = sets[i1], sets[i2]
            set2_reg = point_sets_registration(set1, set2, transformation='affine')
            set1_reg = point_sets_registration(set2, set1, transformation='affine')

            # fill the distance matrix
            for k, (s1, s2) in enumerate([[set1, set2], [set1, set2_reg], [set2, set1_reg]]):
                M[[i1, i2], [i2, i1], k] = point_sets_distance(s1, s2)

    return M


def tree_order(mat, threshold=float('inf'), i_start=0):
    """
    Finds an optimal order, through a tree structure, to progressively align elements by pairs, with M a distance matrix
    indicating the relative distances of all these elements.

    Parameters
    ----------
    mat : (n, n, 3) array
        output from distance_matrix function.
    threshold : float
        distance threshold.
    i_start : int
        in [0, n-1], defines which element is selected first.

    Returns
    -------
    list
        list indicating in which order the elements must be aligned, by pairs.
    """

    scores = []

    mat_min = np.min(mat, axis=2)
    i_done = [i_start]
    l_previous = 0
    n_steps = [0, 0]
    pairs = []

    while len(i_done) != len(mat):

        while l_previous != len(i_done) and len(i_done) != len(mat):
            l_previous = len(i_done)

            # select all possible matches under threshold
            i_match = np.where(np.min(mat_min[i_done, :], axis=0) < threshold)[0]
            m = mat_min[i_done, :][:, i_match]
            i_match_sources = np.array(i_done)[np.argmin(m, axis=0)]

            for i_ms, i_m in zip(i_match_sources, i_match):
                pairs.append([i_ms, i_m])

            i_done = sorted(i_done + list(i_match))
            mat_min[np.ix_(i_done, i_done)] = 1e10
            n_steps[0] += 1

        if len(i_done) != len(mat):
            # if no match score under threshold, add only one match with the minimum score
            i1, i2 = np.unravel_index(mat_min[i_done, :].argmin(), mat_min[i_done, :].shape)
            i_match_source, i_match = np.array(i_done)[i1], i2
            score = mat_min[i_match_source, i_match]
            i_done += [i_match]
            pairs.append([i_match_source, i_match])
            mat_min[np.ix_(i_done, i_done)] = 1e10
            n_steps[1] += 1
            scores.append(score)

    return pairs


def tracking(points_sets, dist_mat, set_threshold=8, berry_threshold=16, t_start=None, do_tree_order=True):
    """
    Time-lapse tracking of berry center coordinates.

    Parameters
    ----------
    points_sets : list
        list of n 2D point sets.
    dist_mat : (n, n, 3) array
        matrix output from distance_matrix function, containing all the distances between couples of points sets.
    set_threshold : float
        distance threshold used when determining the alignment order (see tree_order function).
        The method was tested using a threshold around 0.125 x the median berry length in our dataset.
    berry_threshold : float
        distance threshold used during the bipartite matching (see point_sets_matching function).
        The method was tested using a threshold around 0.25 x the median berry length in our dataset.
    t_start : int
        which time-point (i.e. which point set) is used to initiate tracking. in [0, n-1].
    do_tree_order : bool
        Whether normal chronological order (False) or a computed tree order (True) is used.

    Returns
    -------
    list of list
        i-th sub-list contains the berry identifiers given to each center point of the i-th point set. Points sharing
        the same identifier (over time) are assumed to correspond TO the same berry. An identifier equal to -1 means
        that the algorithm failed to track this point.
    """

    # ===== selecting optimal value for t_start ======================================================================

    if t_start is None:
        t_best, k_max = None, float('-inf')
        for t_start in range(len(points_sets)):
            sets_pairs = tree_order(dist_mat, i_start=t_start, threshold=set_threshold)
            k_above = [k for k, (i, j) in enumerate(sets_pairs) if np.min(dist_mat[i, j]) > set_threshold]
            k_fail = len(points_sets) - 1 if not k_above else k_above[0]
            if k_fail > k_max:
                t_best, k_max = t_start, k_fail
        t_start = t_best

    # ===== order of sets alignment ==================================================================================

    if do_tree_order:
        sets_pairs = tree_order(dist_mat, i_start=t_start, threshold=set_threshold)
    else:
        sets_pairs = [[k, k + 1] for k in range(len(points_sets) - 1)]

    # ===== alignment ================================================================================================

    berry_ids = [[-1] * len(set) for set in points_sets]
    t_start = sets_pairs[0][0]
    berry_ids[t_start] = [k for k in range(len(berry_ids[t_start]))]

    for t1, t2 in sets_pairs:

        set1, set2 = points_sets[t1], points_sets[t2]

        # registration
        if np.argmin(dist_mat[t1, t2]) == 1:
            set2 = point_sets_registration(set1, set2, transformation='affine')
        elif np.argmin(dist_mat[t1, t2]) == 2:
            set1 = point_sets_registration(set2, set1, transformation='affine')

        # match pairs of berries with a distance under a threshold
        berry_pairs = point_sets_matching(set1, set2, threshold=berry_threshold)

        for b1, b2 in berry_pairs:
            berry_ids[t2][b2] = berry_ids[t1][b1]

    return berry_ids
