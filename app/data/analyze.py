import sys
import re
from argparse import ArgumentParser
from warnings import warn

import pandas as pd
from skbio.stats.distance import DistanceMatrix

from sklearn.manifold import MDS
from sklearn.cluster import OPTICS

from algorithms import pcoa as modified_pcoa_lib

DEF_MODE = "mds"
DEF_N_COMPONENTS = None
DEF_MDS_N_COMPONENTS = 2
DEF_MAX_ITER = 3000
DEF_MDS_EPS = 9

DEF_MIN_SAMPLES = 2
DEF_XI = 0.05


def fix_duplicated_index(file_name):
    hdf_filename = f'/output/{file_name}.h5'

    with pd.HDFStore(hdf_filename) as store:
        matrix = store["input_data"]
        index = pd.Series(matrix.index)

        for dup in index[index.duplicated()].unique():
            index[index[index == dup].index.values.tolist()] = [f"{dup}_{i}" if i != 0 else dup
                                                                for i in range(sum(index == dup))]
        matrix.index = index

        store.remove("input_data")
        store.put("input_data", matrix)


def mds(file_name, hdf_key, n_components, max_iter, eps):
    hdf_filename = f'/output/{file_name}.h5'

    stats_hdf_key = f"{hdf_key}/statistics"
    coordinates_hdf_key = f"{hdf_key}/coordinates"

    with pd.HDFStore(hdf_filename) as store:
        matrix = store["input_data"]
        embedding = MDS(
            n_components=n_components,
            dissimilarity='precomputed',
            n_jobs=-1,
            max_iter=max_iter,
            eps=10 ** (-eps)
        )
        t = embedding.fit_transform(matrix)

        stats = pd.Series({"stress": embedding.stress_, "n_iter": embedding.n_iter_})
        df = pd.DataFrame(data=t, index=matrix.index)

        if coordinates_hdf_key in store.keys():
            store.remove(coordinates_hdf_key)
        store.put(coordinates_hdf_key, df)

        if stats_hdf_key in store.keys():
            store.remove(stats_hdf_key)
        store.put(stats_hdf_key, stats)


def pcoa(file_name, hdf_key, n_components=0):
    hdf_filename = f'/output/{file_name}.h5'

    coordinates_hdf_key = f"{hdf_key}/coordinates"
    eigvals_hdf_key = f"{hdf_key}/eigvals"
    proportion_explained_hdf_key = f"{hdf_key}/proportion_explained"

    with pd.HDFStore(hdf_filename) as store:
        matrix = store["input_data"]
        dist_matrix = DistanceMatrix(matrix.values, ids=matrix.index)
        results = modified_pcoa_lib.pcoa(dist_matrix, number_of_dimensions=n_components)

        if coordinates_hdf_key in store.keys():
            store.remove(coordinates_hdf_key)
        store.put(coordinates_hdf_key, results.samples)

        if eigvals_hdf_key in store.keys():
            store.remove(eigvals_hdf_key)
        store.put(eigvals_hdf_key, results.eigvals)

        if proportion_explained_hdf_key in store.keys():
            store.remove(proportion_explained_hdf_key)
        store.put(proportion_explained_hdf_key, results.proportion_explained)


def prepare_for_optics(file_name, components_hdf_key, optics_hdf_key):
    output_hdf_key = f"{optics_hdf_key}/output"
    hdf_filename = f'/output/{file_name}.h5'

    with pd.HDFStore(hdf_filename) as store:
        matrix = store[components_hdf_key]
        df = pd.DataFrame(
            index=matrix.index,
            columns=[
                'labels',
                'reachability',
                'ordering',
                'core_distances',
                'predecessor',
            ],
            dtype='float'
        )

        if output_hdf_key in store:
            store.remove(output_hdf_key)

        store.put(output_hdf_key, df)


def optics(file_name, components_hdf_key, optics_hdf_key, min_samples, xi):
    prepare_for_optics(file_name, components_hdf_key, optics_hdf_key)

    output_hdf_key = f"{optics_hdf_key}/output"
    hierarchy_hdf_key = f"{optics_hdf_key}/hierarchy"  # TODO:
    hdf_filename = f'/output/{file_name}.h5'

    m = re.match(r"^/(?P<procedure>(pcoa|mds|components_mds))/c(?P<n_components>\d+)", components_hdf_key)

    with pd.HDFStore(hdf_filename) as store:
        if m:
            n_components = int(m.groupdict()["n_components"])
            matrix = store[components_hdf_key].iloc[:, :n_components]
        else:
            matrix = store[components_hdf_key]

        clustering = OPTICS(min_samples=min_samples, n_jobs=-1, xi=xi).fit(matrix)

        output = store[output_hdf_key]
        output['labels'] = [f"cluster {x}" for x in clustering.labels_]
        output['reachability'] = clustering.reachability_
        output['ordering'] = clustering.ordering_
        output['core_distances'] = clustering.core_distances_
        output['predecessor'] = clustering.predecessor_

        df = pd.DataFrame(data=clustering.cluster_hierarchy_)

        store.put(output_hdf_key, output)
        if hierarchy_hdf_key in store.keys():
            store.remove(hierarchy_hdf_key)
        store.put(hierarchy_hdf_key, df)


if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument('-i', '--in', help='Dataset file name', required=True)

    #Analysis mode params
    ap.add_argument('-p', '--procedure', choices=("mds", "pcoa"), default=DEF_MODE,
                    help="Algorithm to immerse the dissimilarities. "
                         f"Default: {DEF_MODE}")

    ap.add_argument('-I', '--immerse', action='store_true',
                    help="Perform immersing dissimilarities using choosen algorithm.")

    ap.add_argument('-C', '--cluster', action='store_true',
                    help="Perform clustering using OPTICS algorithm.")

    #Arguments for analysis alogritms
    ap.add_argument('-c', '--components', nargs='?',
                    default=DEF_N_COMPONENTS, help="Number of dimensions in which to immerse the dissimilarities. "
                                                   "For the PCoA algorithm 0 means n_components equals number "
                                                   "of samples in the original dataset. "
                                                   f"Default: {DEF_N_COMPONENTS}")
    ap.add_argument('-m', '--max_iter', nargs='?',
                    default=DEF_MAX_ITER, help="Maximum number of iterations of the SMACOF algorithm for a single run. "
                                               f"Default: {DEF_MAX_ITER}. Not used for PCoA mode.")
    ap.add_argument('-e', '--eps', nargs='?',
                    default=DEF_MDS_EPS, help="10^-{eps} Relative tolerance with respect to stress at which "
                                              "to declare convergence. "
                                              f"Default: {DEF_MDS_EPS}. Not used for PCoA mode.")
    ap.add_argument('-s', '--min_samples', nargs='?',
                    default=DEF_MIN_SAMPLES,
                    help="The number of samples in a neighborhood for a point to be considered as a core point. Also, "
                         "up and down steep regions can’t have more then min_samples consecutive non-steep points. "
                         f"Default: {DEF_MIN_SAMPLES}")
    ap.add_argument('-x', '--xi', nargs='?',
                    default=DEF_XI,
                    help="Determines the minimum steepness on the reachability plot that constitutes a cluster "
                         "boundary. For example, an upwards point in the reachability plot is defined by the ratio "
                         "from one point to its successor being at most 1-{xi}. "
                         f"Default: {DEF_XI}")

    if len(sys.argv) == 1:
        ap.print_help(sys.stderr)
        sys.exit(1)

    args = vars(ap.parse_args())

    input_name = args['in']
    perform_immersing = args['immerse']
    perform_clustering = args['cluster']

    if not perform_immersing and not perform_clustering:
        warn(
            "Neither immersing nor clustering ware choosen. There is not analysis to perform. "
            "Rerun program with -I, -C or both parameters given."
        )
        exit(-1)

    procedure = args['procedure']
    max_iter = int(args['max_iter'])
    eps_pow = int(args['eps'])
    min_samples = int(args['min_samples'])
    xi = args['xi']

    if args['components'] is not None:
        n_components = int(args['components'])
    else:
        if procedure == "mds":
            n_components = DEF_MDS_N_COMPONENTS
        elif procedure == "pcoa":
            n_components = None
            if perform_immersing:
                warn(
                    "Chosen procedure is PCoA, and n_components are not set. Parameter will be set automatically to "
                    "the length of the dataset."
                )
            if perform_clustering:
                warn(
                    "Chosen procedure is PCoA, and n_components are not set. OPTICS clustering will not be optimal "
                    "for these parameters. Please consider rerunning analysis with n_components parameter given."
                )
        else:
            raise ValueError(f"Cannot match default n_components for the procedure {{{procedure}}}")

    fix_duplicated_index(input_name)

    if perform_immersing:
        if procedure == "mds":
            components_hdf_key = f"components_mds/c{n_components}/m{max_iter}/e{eps_pow}"
            mds(input_name, components_hdf_key, n_components, max_iter, eps_pow)
        elif procedure == "pcoa":
            components_hdf_key = f"components_pcoa"
            pcoa(input_name, components_hdf_key)

    if perform_clustering:
        if procedure == "mds":
            components_hdf_key = f"components_mds/c{n_components}/m{max_iter}/e{eps_pow}/coordinates"
            optics_hdf_key = f"clustering_mds/c{n_components}/m{max_iter}/e{eps_pow}/s{min_samples}/x{str(xi).replace('.', '_')}"
            optics(input_name, components_hdf_key, optics_hdf_key, min_samples, xi)

        elif procedure == "pcoa":
            components_hdf_key = f"components_pcoa/coordinates"
            optics_hdf_key = f"clustering_pcoa/c{n_components}/s{min_samples}/x{str(xi).replace('.', '_')}"
            optics(input_name, components_hdf_key, optics_hdf_key, min_samples, xi)
