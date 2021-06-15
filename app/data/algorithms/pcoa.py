import numpy as np
import pandas as pd
from warnings import warn
from scipy.linalg import eigh

from skbio.stats.distance import DistanceMatrix
from skbio.stats.ordination._ordination_results import OrdinationResults


def center_matrix(D):
    if D.shape[0] == D.shape[1]:
        n = D.shape[0]
    else:
        raise ValueError("Invalid distance matrix. Distance matrix should be square.")

    # # ones = np.ones((n, n))
    # # matrix = np.diag(np.ones(n)) - (ones / n)
    # return np.dot(np.dot(matrix, D), matrix)

    row_means = D.mean(axis=1, keepdims=True)
    col_means = D.mean(axis=0, keepdims=True)
    matrix_mean = D.mean()
    return D - row_means - col_means + matrix_mean


def pcoa(distance_matrix, number_of_dimensions=0):
    r"""
    This is modified version of PCoA algorithm from skbio.stats.ordination.

    Modificaton:
    When negative eigenvalues are present in the decomposition results,
    the distance matrix D can be modified using the Lingoes procedure
    to produce results without negative eigenvalues.

    In the Lingoes (1971) procedure, a constant c1, equal to twice absolute value of
    the largest negative value of the original principal coordinate analysis,
    is added to each original squared distance in the distance matrix, except the diagonal values.
    A new principal coordinate analysis, performed on the modified distances,
    has at most (n-2) positive eigenvalues, at least 2 null eigenvalues, and no negative eigenvalue.

    Basics: https://www.rdocumentation.org/packages/ape/versions/5.4-1/topics/pcoa

    :param distance_matrix : DistanceMatrix
        A distance matrix.
    :param number_of_dimensions : int, optional
        Dimensions to reduce the distance matrix to. This number determines
        how many eigenvectors and eigenvalues will be returned.
        By default, equal to the number of dimensions of the distance matrix,
        as default eigendecomposition using SciPy's `eigh` method computes
        all eigenvectors and eigenvalues. If using fast heuristic
        eigendecomposition through `fsvd`, a desired number of dimensions
        should be specified. Note that the default eigendecomposition
        method `eigh` does not natively support a specifying number of
        dimensions to reduce a matrix to, so if this parameter is specified,
        all eigenvectors and eigenvalues will be simply be computed with no
        speed gain, and only the number specified by `number_of_dimensions`
        will be returned. Specifying a value of `0`, the default, will
        set `number_of_dimensions` equal to the number of dimensions of the
        specified `distance_matrix`.
    :return:
    OrdinationResults
        Object that stores the PCoA results, including eigenvalues, the
        proportion explained by each of them, and transformed sample
        coordinates.
    """

    distance_matrix = DistanceMatrix(distance_matrix)

    # Center distance matrix, a requirement for PCoA here
    matrix_data = center_matrix(-0.5*(distance_matrix.data*distance_matrix.data))

    # If no dimension specified, by default will compute all eigenvectors
    # and eigenvalues
    if number_of_dimensions == 0:
        number_of_dimensions = matrix_data.shape[0]
    elif number_of_dimensions < 0:
        raise ValueError('Invalid operation: cannot reduce distance matrix '
                         'to negative dimensions using PCoA. Did you intend '
                         'to specify the default value "0", which sets '
                         'the number_of_dimensions equal to the '
                         'dimensionality of the given distance matrix?')

    eigvals, eigvecs = eigh(matrix_data)
    long_method_name = "Principal Coordinate Analysis"

    negative_close_to_zero = np.isclose(eigvals, 0)
    eigvals[negative_close_to_zero] = 0

    if np.any(eigvals < 0):
        warn(
            "The result contains negative eigenvalues."
            " Please compare their magnitude with the magnitude of some"
            " of the largest positive eigenvalues. If the negative ones"
            " are smaller, it's probably safe to ignore them, but if they"
            " are large in magnitude, the results won't be useful. See the"
            " Notes section for more details. The smallest eigenvalue is"
            " {0} and the largest is {1}.".format(eigvals.min(),
                                                  eigvals.max()),
            RuntimeWarning
        )
        c1 = abs(min(eigvals)) * 2
        warn(
            "Applying correction.",
            RuntimeWarning
        )
        matrix_data = (distance_matrix.data * distance_matrix.data) + c1
        matrix_data = matrix_data - np.diag(np.full(matrix_data.shape[0], c1))
        matrix_data = np.sqrt(matrix_data)
        matrix_data = center_matrix(-0.5 * matrix_data)

        eigvals, eigvecs = eigh(matrix_data)

        negative_close_to_zero = np.isclose(eigvals, 0)
        eigvals[negative_close_to_zero] = 0

        if np.any(eigvals < 0):
            warn(
                "The result contains negative eigenvalues, despite applying correction.",
                RuntimeWarning
            )


    # eigvals might not be ordered, so we first sort them, then analogously
    # sort the eigenvectors by the ordering of the eigenvalues too
    idxs_descending = eigvals.argsort()[::-1]
    eigvals = eigvals[idxs_descending]
    eigvecs = eigvecs[:, idxs_descending]

    # If we return only the coordinates that make sense (i.e., that have a
    # corresponding positive eigenvalue), then Jackknifed Beta Diversity
    # won't work as it expects all the OrdinationResults to have the same
    # number of coordinates. In order to solve this issue, we return the
    # coordinates that have a negative eigenvalue as 0
    num_positive = (eigvals >= 0).sum()
    eigvecs[:, num_positive:] = np.zeros(eigvecs[:, num_positive:].shape)
    eigvals[num_positive:] = np.zeros(eigvals[num_positive:].shape)


    sum_eigenvalues = np.sum(eigvals)

    proportion_explained = eigvals / sum_eigenvalues

    # In case eigh is used, eigh computes all eigenvectors and -values.
    # So if number_of_dimensions was specified, we manually need to ensure
    # only the requested number of dimensions
    # (number of eigenvectors and eigenvalues, respectively) are returned.
    eigvecs = eigvecs[:, :number_of_dimensions]
    eigvals = eigvals[:number_of_dimensions]
    proportion_explained = proportion_explained[:number_of_dimensions]

    # Scale eigenvalues to have length = sqrt(eigenvalue). This
    # works because np.linalg.eigh returns normalized
    # eigenvectors. Each row contains the coordinates of the
    # objects in the space of principal coordinates. Note that at
    # least one eigenvalue is zero because only n-1 axes are
    # needed to represent n points in a euclidean space.
    coordinates = eigvecs * np.sqrt(eigvals)

    axis_labels = ["PC%d" % i for i in range(1, number_of_dimensions + 1)]
    return OrdinationResults(
        short_method_name="PCoA",
        long_method_name=long_method_name,
        eigvals=pd.Series(eigvals, index=axis_labels),
        samples=pd.DataFrame(coordinates, index=distance_matrix.ids,
                             columns=axis_labels),
        proportion_explained=pd.Series(proportion_explained,
                                       index=axis_labels))