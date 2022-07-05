from typing import List

import numpy as np
import persim
from ripser import ripser
from scipy.spatial import distance_matrix

import matplotlib.pyplot as plt
from tqdm import tqdm

rng = np.random.default_rng()


def subsample(data: np.ndarray, num_subsample_points: int, num_target_points: int,
              num_processes: int=100, max_dim: int=1,
              scale: bool=False):
    """Compute persistent homology via subsampling methods.

    Parameters
    ----------
    data : np.ndarray
        Input data used to compute persistent homology.
    num_subsample_points : int
        Number of points to sample per process.
    num_target_points : int
        Number of points to use for target computation.
    num_processes : int, optional
        Number of times to subsample and compute persistent homology.
    max_dim : int, optional
        The maximum dimension to compute with persistent homology.

    Returns
    -------
    subsample_diagrams : list of np.ndarray
        The combined persistence diagrams (indexed by dimension) for the bootstrapped computations.
    target_diagrams : list of np.ndarray
        The persistence diagrams (indexed by dimension) for the target computation.
    """

    random_subset = rng.integers(0, data.shape[0], size=num_target_points)
    data_subset = data[random_subset, :]

    # compute persistent homology
    target_rips = ripser(data_subset, maxdim=max_dim)
    target_diagrams = target_rips['dgms']

    for i in tqdm(range(num_processes)):
        # random sample
        random_subset = rng.integers(0, data.shape[0], size=num_subsample_points)
        data_subset = data[random_subset, :]
        
        # compute persistent homology
        if scale:
            data_subset = np.sqrt(num_subsample_points / num_target_points) * distance_matrix(data_subset, data_subset)
        else:
            data_subset = distance_matrix(data_subset, data_subset)
        subsample_rips = ripser(data_subset, maxdim=max_dim, distance_matrix=True)
        
        # combine persistence diagrams
        if i == 0:
            subsample_diagrams = subsample_rips['dgms']
            # for dim, pairs in enumerate(subsample_rips['dgms']):
            #     boot_diagrams[dim] = np.sqrt(num_target_points / num_subsample_points) * pairs
        else:
            for dim, pairs in enumerate(subsample_rips['dgms']):
                # subsample_diagrams[dim] = np.vstack([subsample_diagrams[dim], np.sqrt(num_target_points / num_subsample_points) * pairs])
                subsample_diagrams[dim] = np.vstack([subsample_diagrams[dim], pairs])

    return subsample_diagrams, target_diagrams


def compare_diagrams(A: np.ndarray, B: np.ndarray):
    """Computes a measure for how much two persistence diagrams differ.

    Parameters
    ----------
    A : np.ndarray
        An array representing points in a persistence diagram.
    B : np.ndarray
        An array representing points in a persistence diagram.

    Returns
    -------
    float
        A value representing a measure for how much A and B differ.
    """
    bottleneck_distance = persim.bottleneck(A, B)
    A_variance, B_variance = np.var(A[:,1]), np.var(B[:,1])

    return bottleneck_distance / (A_variance + B_variance)


def count_features(barcode_diagram, persistence_diagram, eps: float):
    """Counts the number of features based on the two representations of the persistence plots.
    
    Parameters
    ----------
    barcode_diagram : np.ndarray

    persistence_diagram : np.ndarray

    Returns
    -------
    barcode_count : int
        The number of features counted via the barcode diagram.
    pd_count : int
        The number of features counted via the persistence diagram.
    """
    # barcode method
    # count number of bars that pass the midpoint of the longest bar
    # can this just be length >= 0.5 since barcodes are normalized?
    # interpret as probability?
    longest_length = np.max(barcode_diagram[:,1])
    barcode_count = np.sum(barcode_diagram[:,1] >= 0.5*longest_length)

    # persistence diagram method
    # count number of clusters by grouping together balls of radius epsilon
    persistence_count = np.sum(persistence_diagram[:,1] >= eps)

    return barcode_count, persistence_count


def estimate_features(distances: List[float], diagrams: List[np.ndarray]):
    """
    Estimates the number of features by using a consensus voting method.

    Given a list of values, find the peaks and count the number of features.
    Take the `consensus` to yield an estimate of the number of total features.

    Parameters
    ----------
    distances : List[float]
        A list of floats representing distances between consecutive persistence diagrams.
    diagrams : List[np.ndarray]
        A list of persistence diagrams.
    """
    # find the peaks
    peak_idx = []
    for idx, (prev_dist, curr_dist, next_dist) in enumerate(zip(distances[:-2], distances[1:-1], distances[2:]), start=1):
        if (curr_dist >= prev_dist) and (curr_dist >= next_dist):
            peak_idx.append(idx)
    # retrieve the peaks
    print(f'{peak_idx=}\n{len(diagrams)}')
    feature_counts = np.array([count_features(diagrams[idx][0], diagrams[idx][0], eps=0.1) for idx in peak_idx])
    barcode_est, persistence_est = np.average(feature_counts[:,0], weights=np.exp(feature_counts[:,0])), np.average(feature_counts[:,1], weights=np.exp(feature_counts[:,1]))
    
    return barcode_est, persistence_est


def show_diagrams(diagrams, 
                  labels=None,
                  as_lifetimes: bool=False,
                  as_barcodes: bool=False,
                  buffer_width: int=5,
                  ax=None,
                  xy_range=None, 
                  colormap: str='default',
                  title=None,
                  legend: bool=True,
                  show: bool=False,
                  **kwargs):
    """Plot persistence diagrams given the birth/death pairs.
    
    Given a list of birth/death pairs, plot the corresponding persistence diagrams.
    Most of this is taken from the persim source code.
    
    Parameters
    ----------
    diagrams : list of Nx2 arrays
        Input list of arrays where each array is an array of birth/death pairs.
    labels : list of strings
        The labels corresponding to the input diagrams.
    as_lifetimes : bool, optional
        If this is set to True, the birth/death pairs are plotted as (birth, death-birth).
    as_barcodes : bool, optional
        If this is set to True, the birth/death pairs are plotted as (birth, death) intervals.
    buffer_width : int
        Sets the spacing in a barcode diagram between barcodes in different homology groups.
    xy_range: list of numeric [xmin, xmax, ymin, ymax]
        User provided range of axes. This is useful for comparing 
        multiple persistence diagrams.
    colormap : str, optional
        Any of the available matplotlib colormaps.
    legend : bool
        Show the legend for the persistence diagram.
    show : bool
        Call plt.show() at the end.
        If using in subplots, this should be set as False.
    """ 
    ax = ax or plt.gca()
    plt.style.use(colormap)
    
    # construct copy with proper type of each diagram so we can freely edit them.
    diagrams = [diagram.astype(np.float32, copy=True) for diagram in diagrams]
     
    # find min and max of all visible diagrams
    concat_dgms = np.concatenate(diagrams).flatten()
    has_inf = np.any(np.isinf(concat_dgms))
    finite_dgms = concat_dgms[np.isfinite(concat_dgms)]
            
    # clever bounding boxes of the diagram
    if not xy_range:
        # define bounds of diagram
        ax_min, ax_max = np.min(finite_dgms), np.max(finite_dgms)
        x_r = ax_max - ax_min

        # give plot a nice buffer on all sides
        # ax_range=0 when only one point
        buffer = 1 if xy_range == 0 else x_r / 5

        x_down = ax_min - buffer / 2
        x_up = ax_max + buffer

        y_down, y_up = x_down, x_up
    else:
        x_down, x_up, y_down, y_up = xy_range

    yr = y_up - y_down
    
    # plotting as barcode diagram
    if as_barcodes:
        xlabel = '$\varepsilon$'
        bars_per_diagram = [diagram.shape[0] for diagram in diagrams]
        total_bars = sum(bars_per_diagram)
        num_lines = len(diagrams) - 1
        
        # keep track of where higher dimension homology groups start/end in array
        homology_idx = np.cumsum(bars_per_diagram)
        
        # count total number of horizontal slices needed and assign y values
        num_slices = total_bars + num_lines + 2 * len(diagrams) * buffer_width
        all_ys = np.linspace(1, 0, num=num_slices, endpoint=True)
        
        num_lines_drawn = 0
        for dim, diagram in enumerate(diagrams):
            # sort by (birth, death)
            # https://stackoverflow.com/a/38194077
            diagram = diagram[diagram[:, 1].argsort()]
            diagram = diagram[diagram[:, 0].argsort(kind='mergesort')]
            if dim == 0:
                offset = buffer_width
                end_idx = 2 * offset + homology_idx[dim]
                ys = all_ys[:end_idx]
                
                # barcodes
                ax.hlines(ys[buffer_width:-buffer_width], diagram[:, 0], diagram[:, 1], **kwargs)
                
                # buffer at top
                ax.hlines(ys[:buffer_width], ax_min, ax_max, linewidths=0)
            elif 0 < dim < len(diagrams) - 1:
                offset = 2 * dim * buffer_width + num_lines_drawn
                start_idx, end_idx = offset + homology_idx[dim - 1], offset + homology_idx[dim] + 2 * buffer_width + 1

                sep_line_y = all_ys[start_idx]
                ys = all_ys[start_idx + buffer_width + 1:end_idx - buffer_width]
                
                # line between previous H_k and current H_k
                # ax.hlines(sep_line_y, ax_min, ax_max, linestyle='--', color='k')
                ax.arrow(0, sep_line_y, ax_max, 0, linestyle='--', color='k')
                num_lines_drawn += 1
                
                # barcodes
                ax.hlines(ys, diagram[:, 0], diagram[:, 1], **kwargs)
            elif dim == len(diagrams) - 1:
                offset = 2 * dim * buffer_width + num_lines_drawn
                start_idx = offset + homology_idx[dim - 1]
                
                sep_line_y = all_ys[start_idx]
                ys = all_ys[start_idx + buffer_width + 1:]
                
                # line between previous H_k and current H_k
                # ax.hlines(sep_line_y, ax_min, ax_max, linestyle='--', color='k')
                ax.arrow(0, sep_line_y, ax_max, 0, linestyle='--', color='k')
                num_lines_drawn += 1
                
                # barcodes
                ax.hlines(ys[:-buffer_width], diagram[:, 0], diagram[:, 1], **kwargs)
                
                # buffer at bottom
                ax.hlines(ys[-buffer_width:], ax_min, ax_max, linewidths=0)
            else:
                raise ValueError(f'dimension is greater than the number of diagrams -> {dim} >= {len(diagrams)}')
        
        # draw xy axes
        ax.hlines(0, 0, ax_max, color='k')
        ax.vlines(0, 0, 1, color='k')
        
        # get rid of y axis labels
        ax.set(yticklabels=[])  # remove the tick labels
        ax.tick_params(left=False)  # remove the ticks
        
        ax.set_xlabel(r'$\varepsilon$')
                
        
    # plotting as a persistence diagram
    else:
        # plot inf line
        if has_inf:
            # put inf line slightly below top
            b_inf = y_down + yr * 0.95
            ax.plot([x_down, x_up], [b_inf, b_inf], '--', c='k', label=r'$\infty$')

            # convert each inf in each diagram with b_inf
            for dgm in diagrams:
                dgm[np.isinf(dgm)] = b_inf
        
        xlabel, ylabel = 'Birth', 'Death'
        if as_lifetimes:
            for diagram in diagrams:
                # set y value to be death - birth
                diagram[:, 1] = diagram[:, 1] - diagram[:, 0]
            ylabel = 'Lifetime'

        # plotting as a standard persistence diagram so include diagonal
        else:
            ax.plot([x_down, x_up], [x_down, x_up], '--', c='k')

        # labels and data points
        if labels is None:
            labels = [str(k) for k, diagram in enumerate(diagrams)]

        # plot each persistence diagram
        for diagram, label in zip(diagrams, labels):
            ax.scatter(diagram[:, 0], diagram[:, 1], 
                       label=label, edgecolor='none',
                      **kwargs)

            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)

        ax.set_xlim([x_down, x_up])
        ax.set_ylim([y_down, y_up])
        # ax.set_aspect('equal', 'box')
    
    if title is not None:
        ax.set_title(title)

    if legend and not as_barcodes:
        # https://stackoverflow.com/a/42403471
        leg = ax.legend(loc='lower right')
        for lh in leg.legendHandles: 
            lh.set_alpha(1)

    if show is True:
        plt.show()
        
    return ax