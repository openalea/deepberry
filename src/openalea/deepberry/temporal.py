import numpy as np

from pycpd import AffineRegistration, RigidRegistration, DeformableRegistration


def scaled_cpd(X, Y, X_add=None, Y_add=None, transformation='affine'):
    """
    Coherent Point Drift algorithm, for point set registration, with inputs scaling.

    X: array of shape (n1, 2) containing the 2d coordinates of the reference/fixed point cloud
    Y: array of shape (n2, 2) containing the 2d coordinates of the moving point cloud
    X_add: array of shape (n1) containing an additional feature describing the points in X
    Y_add: array of shape (n2) containing an additional feature describing the points in Y
    transformation: type of registration, among 'rigid', 'affine', 'deformable'
    """

    if (len(X) <= 2) or (len(Y) <= 2):
        return Y

    # scaling of X and Y
    q, m = 0.5 * np.max(np.max(X, axis=0) - np.min(X, axis=0)), np.mean(X, axis=0)
    X_scaled, Y_scaled = (X - m) / q, (Y - m) / q

    # (optional) scaling of X_add and Y_add. They are then added to X and Y as third dimensions.
    if X_add is not None and Y_add is not None:
        X_add_scaled = (X_add - np.mean(X_add)) / np.std(X_add)
        Y_add_scaled = (Y_add - np.mean(Y_add)) / np.std(Y_add)
        X_scaled = np.concatenate((X_scaled, np.array([X_add_scaled]).T), axis=1)
        Y_scaled = np.concatenate((Y_scaled, np.array([Y_add_scaled]).T), axis=1)

    # point set registration (Coherent Point Drift)
    reg_functions = {'rigid': RigidRegistration, 'affine': AffineRegistration, 'deformable': DeformableRegistration}
    Y_scaled_reg = reg_functions[transformation](**{'X': X_scaled, 'Y': Y_scaled}).register()[0]

    # reverse Y scaling
    Y_scaled_reg = Y_scaled_reg[:, :2]
    Y_reg = (Y_scaled_reg * q) + m

    return Y_reg



def distance_matrix(sets):
    """
    sets = list of points sets.
    """

    M = np.zeros((len(sets), len(sets), 3))
    M[np.arange(len(M)), np.arange(len(M))] = float('inf')  # diagonal

    for i1 in range(len(M) - 1):
        for i2 in np.arange(i1 + 1, len(M)):

            set1, set2 = sets[i1], sets[i2]

            # reg = AffineRegistration(**{'X': centers1, 'Y': centers2})
            # centers2_reg = reg.register()[0]
            # reg = AffineRegistration(**{'X': centers2, 'Y': centers1})
            # centers1_reg = reg.register()[0]
            # reg_topo = topo_registration(set1, set2)
            # centers2_reg_topo = centers2 + reg_topo

            set2_reg = scaled_cpd(set1, set2, transformation='affine')
            set1_reg = scaled_cpd(set2, set1, transformation='affine')
            # centers2_reg_d = scaled_cpd(centers1, centers2, transformation='deformable')
            # centers1_reg_d = scaled_cpd(centers2, centers1, transformation='deformable')
            # centers2_reg_a = scaled_cpd(centers1, centers2, set1['area'], set2['area'], transformation='affine')
            # centers1_reg_a = scaled_cpd(centers2, centers1, set2['area'], set1['area'], transformation='affine')

            # ===== distance matrix ==============
            mats, scores = [], []
            for s1, s2 in [[set1, set2], [set1, set2_reg], [set2, set1_reg]]:
                D = np.zeros((len(s1), len(s2)))
                for k, c1 in enumerate(s1):
                    D[k] = np.sqrt(np.sum((c1 - s2) ** 2, axis=1))
                d = 0.5 * (np.median(np.min(D, axis=0)) + np.median(np.min(D, axis=1)))
                scores.append(d)
            M[[i1, i2], [i2, i1]] = scores

    return M


def pairs_order(M, threshold=8, i_start=0):

#for i_start in range(len(M)):

    scores = []
    # for threshold in np.linspace(1, 20, 50):

    M_min = np.min(M, axis=2)
    i_done = [i_start]
    l_previous = 0
    n_steps = [0, 0]
    pairs = []

    while len(i_done) != len(M):

        while l_previous != len(i_done) and len(i_done) != len(M):
            l_previous = len(i_done)
            # i_matches = sorted(set(np.where(M[i_done, :] < threshold)[1]))
            # a, b = np.where(M[i_done, :] < threshold)

            # select all possible matches under threshold
            i_match = np.where(np.min(M_min[i_done, :], axis=0) < threshold)[0]
            m = M_min[i_done, :][:, i_match]
            i_match_sources = np.array(i_done)[np.argmin(m, axis=0)]

            for i_ms, i_m in zip(i_match_sources, i_match):
                pairs.append([i_ms, i_m])

            i_done = sorted(i_done + list(i_match))
            M_min[np.ix_(i_done, i_done)] = 99999
            n_steps[0] += 1
            # print(len(i_done))

        if len(i_done) != len(M):
            # if no match score under threshold, add only one match with the minimum score
            i1, i2 = np.unravel_index(M_min[i_done, :].argmin(), M_min[i_done, :].shape)
            i_match_source, i_match = np.array(i_done)[i1], i2
            score = M_min[i_match_source, i_match]
            i_done += [i_match]
            pairs.append([i_match_source, i_match])
            M_min[np.ix_(i_done, i_done)] = 99999
            n_steps[1] += 1
            # print(len(i_done), '(add match above threshold: s = {}, i = {})'.format(round(score, 1), i_match))
            scores.append(score)

    return pairs


# def topo_features(ellipses, nvecs=3):
#
#     centers = np.array(ellipses[['ell_x', 'ell_y']])
#
#     # distance matrix
#     D = np.zeros((len(ellipses), len(ellipses)))
#     for i, c in enumerate(centers):
#         D[i] = np.sqrt(np.sum((c - centers) ** 2, axis=1))
#         D[i, i] = float('inf')
#
#     vecs = []
#     for i_berry in range(len(centers)):
#         vec = []
#         ng = D[i_berry, :].argsort()[:nvecs]
#         x, y = centers[i_berry]
#         for i_ng in ng:
#             x_ng, y_ng = centers[i_ng]
#             vec += [x - x_ng, y - y_ng]
#         vecs.append(vec)
#
#     return np.array(vecs)
#
#
# def topo_registration(s1, s2, nvecs=3):
#     """ s1, s2 = two sets (dataframes) of ellipses """
#
#     # TODO : vec[d1, a1, d2, a2, d3, a3] for rigid transfo (consistent with rotation+translation+reflection)
#
#     # TODO vec[[d, a]i], imax = 5, 10 ? faire un matching entre les deux vec et prendre la d mediane
#
#     # topologic features extraction
#     n = min(nvecs, min(len(s1), len(s2)))
#     ft1 = topo_features(s1, nvecs=n)
#     ft2 = topo_features(s2, nvecs=n)
#
#     # distance matrix
#     F = np.zeros((len(ft1), len(ft2)))
#     for i, vec in enumerate(ft1):
#         F[i] = np.sqrt(np.sum(np.abs(vec - ft2), axis=1))
#
#     # feature matching (greedy algorithm)
#     matches, _ = matching(F)
#     good_matches = matches[:(int(len(matches) / 4))]
#
#     centers1 = np.array(s1[['ell_x', 'ell_y']])
#     centers2 = np.array(s2[['ell_x', 'ell_y']])
#
#     # plt.figure()
#     # plt.gca().invert_yaxis()
#     # plt.plot(centers1[:, 0], centers1[:, 1], 'bo', alpha=0.5)
#     # plt.plot(centers2[:, 0], centers2[:, 1], 'ro', alpha=0.5)
#     # for i1, i2 in good_matches:
#     #     (x1, y1), (x2, y2) = centers1[i1], centers2[i2]
#     #     plt.plot([x1, x2], [y1, y2], 'g-')
#
#     # translation registration
#     dx, dy = np.mean(centers1[good_matches[:, 0]] - centers2[good_matches[:, 1]], axis=0)
#
#     # plt.figure()
#     # plt.plot(centers1[:, 0], centers1[:, 1], 'bo', alpha=0.5)
#     # centers2_reg = centers2 + np.array([dx, dy])
#     # plt.plot(centers2_reg[:, 0], centers2_reg[:, 1], 'ro', alpha=0.5)
#
#     return np.array([dx, dy])


# def get_features2(ellipses):
#
#     centers = np.array(ellipses[['ell_x', 'ell_y']])
#
#     # distance matrix
#     D = np.zeros((len(ellipses), len(ellipses)))
#     for i, c in enumerate(centers):
#         D[i] = np.sqrt(np.sum((c - centers) ** 2, axis=1))
#         D[i, i] = float('inf')
#
#     vecs = []
#     for i_berry in range(len(centers)):
#         vec = []
#         ng = D[i_berry, :].argsort()[:5]
#         x, y = centers[i_berry]
#         for i_ng in ng:
#             x_ng, y_ng = centers[i_ng]
#             distance = np.sqrt((x - x_ng) ** 2 + (y - y_ng) ** 2)
#             area = ellipses.iloc[i_ng]['area']
#             vec += [[distance, area]]
#         vecs.append(vec)
#
#     vecs = np.array(vecs)
#
#     return vecs


def matching(M, threshold=float('inf')):
    """ greedy algorithm """
    M2 = M.copy()
    matches = []
    dists = []
    for k in range(min(M2.shape)):
        i_min, j_min = np.unravel_index(M2.argmin(), M2.shape)
        d = M2[i_min, j_min]
        if d < threshold:  # TODO while loop would be faster
            matches.append([i_min, j_min])
            dists.append(d)
        M2[i_min, :] = float('inf')
        M2[:, j_min] = float('inf')
    return np.array(matches), dists


def points_sets_alignment(points_sets, dist_mat, set_threshold=8, berry_threshold=16):

    # ===== selecting optimal value for t_start ======================================================================

    t_best, k_max = None, float('-inf')
    for t_start in range(len(points_sets)):
        sets_pairs = pairs_order(dist_mat, i_start=t_start, threshold=set_threshold)
        k_above = [k for k, (i, j) in enumerate(sets_pairs) if np.min(dist_mat[i, j]) > set_threshold]
        k_fail = len(points_sets) - 1 if not k_above else k_above[0]
        if k_fail > k_max:
            t_best, k_max = t_start, k_fail
    t_start = t_best

    # ===== order of sets alignment ==================================================================================

    sets_pairs = pairs_order(dist_mat, i_start=t_start, threshold=set_threshold)

    # ===== alignment ================================================================================================

    berry_ids = [[-1] * len(set) for set in points_sets]
    t_start = sets_pairs[0][0]
    berry_ids[t_start] = [k for k in range(len(berry_ids[t_start]))]

    for t1, t2 in sets_pairs:

        set1, set2 = points_sets[t1], points_sets[t2]

        if np.argmin(dist_mat[t1, t2]) == 1:
            set2 = scaled_cpd(set1, set2, transformation='affine')
        elif np.argmin(dist_mat[t1, t2]) == 2:
            set1 = scaled_cpd(set2, set1, transformation='affine')

        # berry distance matrix
        D = np.zeros((len(set1), len(set2)))
        for i, c1 in enumerate(set1):
            D[i] = np.sqrt(np.sum((c1 - set2) ** 2, axis=1))

        berry_pairs, dists = matching(D, threshold=berry_threshold)

        for b1, b2 in berry_pairs:
            berry_ids[t2][b2] = berry_ids[t1][b1]

    return berry_ids