import numpy as np

from openalea.deepberry.temporal import point_sets_registration, point_sets_distance, point_sets_matching, \
    distance_matrix, tree_order, tracking


def test_registration():

    n1, n2 = 50, 60
    set1 = np.random.random((n1, 2))
    set2 = np.random.random((n2, 2))

    reg_a = point_sets_registration(set1, set2, transformation='rigid')
    reg_b = point_sets_registration(set1, set2, transformation='affine')
    reg_c = point_sets_registration(set1, set2, transformation='deformable')

    assert reg_a.shape == (max(n1, n2), 2)
    assert reg_b.shape == (max(n1, n2), 2)
    assert reg_c.shape == (max(n1, n2), 2)


def test_distance():

    n1, n2 = 50, 60
    set1 = np.random.random((n1, 2))
    set2 = np.random.random((n2, 2))

    d0 = point_sets_distance(set1, set1)
    d = point_sets_distance(set1, set2)

    assert round(d0, 6) == 0.
    assert d >= 0


def test_matching():

    n1, n2 = 50, 60
    set1 = np.random.random((n1, 2))
    set2 = np.random.random((n2, 2))

    m_a = point_sets_matching(set1, set1, threshold=float('inf'))

    assert np.all(m_a[:, 0] == m_a[:, 1])

    m_b = point_sets_matching(set1, set2, threshold=float('inf'))
    m_c = point_sets_matching(set1, set2, threshold=0.1)
    m_d = point_sets_matching(set1, set2, threshold=0.)

    assert len(m_b) == min(n1, n2)
    assert len(m_c) <= len(m_b)
    assert len(m_d) <= len(m_c)
    assert len(m_d) == 0


def test_matrix():

    n_sets = 10
    sets = [np.random.random((n, 2)) for n in np.random.randint(10, 30, n_sets)]

    matrix = distance_matrix(sets)

    assert matrix.shape == (n_sets, n_sets, 3)
    assert np.all(np.diagonal(matrix) == float('inf'))


def test_tree():

    n_sets = 50
    matrix = np.random.random((n_sets, n_sets, 3))
    for i in range(3):
        np.fill_diagonal(matrix[:, :, i], float('inf'))

    t1 = np.array(tree_order(matrix, threshold=float('inf'), i_start=0))
    t2 = np.array(tree_order(matrix, threshold=0.1, i_start=0))
    t3 = np.array(tree_order(matrix, threshold=0., i_start=0))

    assert len(set(t1[:, 0])) <= len(set(t2[:, 0])) <= len(set(t3[:, 0]))
    assert t1.shape == (n_sets - 1, 2)
    assert t2.shape == (n_sets - 1, 2)
    assert t3.shape == (n_sets - 1, 2)

    t4 = np.array(tree_order(matrix, threshold=0., i_start=int(n_sets / 2)))

    assert t4.shape == (n_sets - 1, 2)


def test_full_tracking():

    n_sets = 10
    n_per_set = np.random.randint(10, 30, n_sets)
    sets = [np.random.random((n, 2)) for n in n_per_set]

    matrix = distance_matrix(sets)

    tr = tracking(points_sets=sets, dist_mat=matrix, set_threshold=0.1, berry_threshold=0.2)

    assert all([n == len(ids) for n, ids in zip(n_per_set, tr)])
