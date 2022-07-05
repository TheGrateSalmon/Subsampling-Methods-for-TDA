from pathlib import Path
import random
import sys

# computation
import numpy as np
from scipy.spatial import distance
from sklearn.cluster import OPTICS
from ripser import ripser

# visualization
mpl_style = 'fivethirtyeight'
ax_size = (8,4)
import matplotlib.pyplot as plt
plt.style.use(mpl_style)
from tqdm import tqdm

sys.path.append('../')
from utils import data_utils, mc_utils, visuals
from utils import subsample

rng = np.random.default_rng()


def modelnet():
    # load ModelNet files
    data_dir = Path('/home/seangrate/Projects/data/modelnet40_normal_resampled/')
    modelnet_files = data_utils.load_modelnet_files(data_dir=data_dir)

    # compute PD
    num_subsample_points = 100
    num_target_points = 1000
    num_processes = 100
    max_dim = 1

    model_file = random.choice(modelnet_files)
    model = data_utils.load_modelnet_model(model_file)
    visuals.vis_modelnet(model[:,:3],data_normals=model[:,3:])
    subsample_diagrams, target_diagrams = subsample.subsample(model[:, :3], num_subsample_points, num_target_points,
                                                              num_processes=num_processes, max_dim=max_dim)

    subsample_hist, target_hist, kl_div = subsample.compare_diagrams(subsample_diagrams[1], target_diagrams[1], dim=1, bins=100)

    print(f'{subsample_hist=}\n\n{target_hist=}')
    print(f'KL(target || subsample): {kl_div}')

    # inf_idx = np.where(np.isinf(kl_div))[0]
    # print(f'x * log(x/y) = {subsample_hist[inf_idx]} * log({subsample_hist[inf_idx]} / {target_hist[inf_idx]}) = {kl_div[inf_idx]}')

    # plot PDs
    for dim, pairs in enumerate(subsample_diagrams):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(2*ax_size[0], 2*ax_size[1]))
        subsample.show_diagrams([subsample_diagrams[dim], target_diagrams[dim]], 
                                labels=['Subsample', 'Target'], title=f'$H_{dim}$ Diagrams', colormap=mpl_style, ax=ax1, alpha=0.25)
        subsample.show_diagrams([subsample_diagrams[dim], target_diagrams[dim]],
                                labels=['Subsample', 'Target'], title=f'$H_{dim}$ Lifetimes', as_lifetimes=True, colormap=mpl_style, ax=ax2, alpha=0.25)

    # visualize model
    visuals.vis_modelnet(model[:, :3], data_normals=model[:, 3:], window_name=model_file.stem)

    num_epochs = 10
    prev_subsample_diagram = subsample_diagrams
    for epoch in tqdm(range(num_epochs)):
        if epoch == 0:
            current_subsample_diagram = ripser(prev_subsample_diagram[1])['dgms']
        else:
            # keep only points without inf value
            inf_mask = np.isinf(prev_subsample_diagram[0][:,1])
            current_subsample_diagram = ripser(prev_subsample_diagram[0][~inf_mask], maxdim=0)['dgms']
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(2*ax_size[0], 2*ax_size[1]))
        subsample.show_diagrams([current_subsample_diagram[0]],
                                labels=['Subsample'], title=f'$H_{0}$ Diagrams', as_barcodes=True, colormap=mpl_style, ax=ax1, alpha=0.5)
        subsample.show_diagrams([current_subsample_diagram[0]],
                                labels=['Subsample'], title=f'$H_{0}$ Diagrams', as_lifetimes=True, colormap=mpl_style, ax=ax2, alpha=0.5)
        prev_subsample_diagram = current_subsample_diagram

    # plot PDs
    for dim, pairs in enumerate(subsample_diagrams):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(2*ax_size[0], 2*ax_size[1]))
        subsample.show_diagrams([subsample_diagrams[dim]], 
                                labels=['Subsample', 'Target'], title=f'$H_{dim}$ Diagrams', colormap=mpl_style, ax=ax1, alpha=0.25)
        subsample.show_diagrams([subsample_diagrams[dim]],
                                labels=['Subsample', 'Target'], title=f'$H_{dim}$ Lifetimes', as_lifetimes=True, colormap=mpl_style, ax=ax2, alpha=0.25)
    # plot PDs
    for dim, pairs in enumerate(target_diagrams):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(2*ax_size[0], 2*ax_size[1]))
        subsample.show_diagrams([target_diagrams[dim]], 
                                labels=['Target'], title=f'$H_{dim}$ Diagrams', colormap=mpl_style, ax=ax1, alpha=0.25)
        subsample.show_diagrams([target_diagrams[dim]],
                                labels=['Target'], title=f'$H_{dim}$ Lifetimes', as_lifetimes=True, colormap=mpl_style, ax=ax2, alpha=0.25)


def minecraft():
    # full region has around 115 million points
    # keep subset for computational/visualization limitations
    data_dir = Path('/home/seangrate/Projects/data/minecraft/raw/')
    region = data_utils.load_minecraft_region(data_dir, bound_region=False)

    for bounds in [((0,32), (0,32)), ((-32,0), (0,32)), ((-32,0), (-32,0)), ((0,32), (-32,0))]:
        # compute PD
        num_subsample_points = 100
        num_target_points = 1000
        num_processes = 100
        max_dim = 1

        data = mc_utils.bound_region(region, x_bounds=bounds[0], y_bounds=bounds[1])
        data = data / np.linalg.norm(data, axis=0, ord=2)

        subsample_diagrams, target_diagrams = subsample.subsample(data, num_subsample_points, num_target_points,
                                                                num_processes=num_processes, max_dim=max_dim)

        # plot PDs
        for dim, pairs in enumerate(subsample_diagrams):
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(2*ax_size[0], 2*ax_size[1]))
            subsample.show_diagrams([subsample_diagrams[dim], target_diagrams[dim]], 
                                    labels=['Subsample', 'Target'], title=f'$H_{dim}$ Diagrams', colormap=mpl_style, ax=ax1, alpha=0.25)
            subsample.show_diagrams([subsample_diagrams[dim], target_diagrams[dim]],
                                    labels=['Subsample', 'Target'], title=f'$H_{dim}$ Lifetimes', as_lifetimes=True, colormap=mpl_style, ax=ax2, alpha=0.25)


def mobius_band():
    # load data
    num_subsample_points = 100
    num_target_points = 1000
    num_processes = 100
    max_dim = 1

    data = data_utils.load_mobius_band(num_points=num_target_points)
    data = data / np.linalg.norm(data, axis=0, ord=2)

    # compute PD
    subsample_diagrams, target_diagrams = subsample.subsample(data, num_subsample_points, num_target_points,
                                                              num_processes=num_processes, max_dim=max_dim)

    # plot PDs
    for dim, pairs in enumerate(subsample_diagrams):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(2*ax_size[0], 2*ax_size[1]))
        subsample.show_diagrams([subsample_diagrams[dim], target_diagrams[dim]], 
                                labels=['Subsample', 'Target'], title=f'$H_{dim}$ Diagrams', colormap=mpl_style, ax=ax1, alpha=0.25)
        subsample.show_diagrams([subsample_diagrams[dim], target_diagrams[dim]],
                                labels=['Subsample', 'Target'], title=f'$H_{dim}$ Lifetimes', as_lifetimes=True, colormap=mpl_style, ax=ax2, alpha=0.25)

    num_epochs = 10
    prev_subsample_diagram = subsample_diagrams
    for epoch in tqdm(range(num_epochs)):
        if epoch == 0:
            current_subsample_diagram = ripser(prev_subsample_diagram[1])['dgms']
        else:
            # keep only points without inf value
            inf_mask = np.isinf(prev_subsample_diagram[0][:,1])
            current_subsample_diagram = ripser(prev_subsample_diagram[0][~inf_mask], maxdim=0)['dgms']
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(2*ax_size[0], 2*ax_size[1]))
        subsample.show_diagrams([current_subsample_diagram[0]],
                                labels=['Subsample'], title=f'$H_{0}$ Diagrams', as_barcodes=True, colormap=mpl_style, ax=ax1, alpha=0.25)
        subsample.show_diagrams([current_subsample_diagram[0]],
                                labels=['Subsample'], title=f'$H_{0}$ Diagrams', as_lifetimes=True, colormap=mpl_style, ax=ax2, alpha=0.25)
        prev_subsample_diagram = current_subsample_diagram

    # kmeans
    clustering = OPTICS().fit(subsample_diagrams[1])
    ordering, labels = clustering.ordering_, clustering.labels_

    print(clustering.labels_)
    print(sum(1 for i in clustering.labels_ if i == -1))
    print(clustering.labels_.shape[0])

    fig, ax = plt.subplots(1, 1, figsize=(2*ax_size[0], 2*ax_size[1]))
    subsample.show_diagrams([subsample_diagrams[1][clustering.ordering_]], 
                            labels=['Subsample', 'Target'], title=f'$H_{1}$ Diagrams', colormap=mpl_style, c=clustering.labels_, ax=ax, alpha=0.25)


def klein_bottle():
    # load data
    num_subsample_points = 100
    num_target_points = 1000
    num_processes = 100
    max_dim = 1

    data = data_utils.load_klein_bottle(num_points=num_target_points, immersion='figure eight')
    data = data / np.linalg.norm(data, axis=0, ord=2)

    # visuals.vis_klein_bottle(data)

    # compute PD
    subsample_diagrams, target_diagrams = subsample.subsample(data, num_subsample_points, num_target_points,
                                                              num_processes=num_processes, max_dim=max_dim)

    # plot PDs
    for dim, pairs in enumerate(subsample_diagrams):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(2*ax_size[0], 2*ax_size[1]))
        subsample.show_diagrams([subsample_diagrams[dim], target_diagrams[dim]], 
                                labels=['Subsample', 'Target'], title=f'$H_{dim}$ Diagrams', colormap=mpl_style, ax=ax1, alpha=0.25)
        subsample.show_diagrams([subsample_diagrams[dim], target_diagrams[dim]],
                                labels=['Subsample', 'Target'], title=f'$H_{dim}$ Lifetimes', as_lifetimes=True, colormap=mpl_style, ax=ax2, alpha=0.25)


def torus():
    # load data
    num_subsample_points = 100
    num_target_points = 1000
    num_processes = 100
    max_dim = 1

    data = data_utils.load_torus(num_points=num_target_points, R=2, r=1)
    data = data / np.linalg.norm(data, axis=0, ord=2)

    visuals.vis_torus(data)

    # compute PD
    subsample_diagrams, target_diagrams = subsample.subsample(data, num_subsample_points, num_target_points,
                                                            num_processes=num_processes, max_dim=max_dim)

    # plot PDs
    for dim, pairs in enumerate(subsample_diagrams):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(2*ax_size[0], 2*ax_size[1]))
        subsample.show_diagrams([subsample_diagrams[dim], target_diagrams[dim]], 
                                labels=['Subsample', 'Target'], title=f'$H_{dim}$ Diagrams', colormap=mpl_style, ax=ax1, alpha=0.25)
        subsample.show_diagrams([subsample_diagrams[dim], target_diagrams[dim]],
                                labels=['Subsample', 'Target'], title=f'$H_{dim}$ Lifetimes', as_lifetimes=True, colormap=mpl_style, ax=ax2, alpha=0.25)

    num_epochs = 10
    prev_subsample_diagram = subsample_diagrams
    for epoch in tqdm(range(num_epochs)):
        if epoch == 0:
            current_subsample_diagram = ripser(prev_subsample_diagram[1])['dgms']
        else:
            # keep only points without inf value
            inf_mask = np.isinf(prev_subsample_diagram[0][:,1])
            current_subsample_diagram = ripser(prev_subsample_diagram[0][~inf_mask], maxdim=0)['dgms']
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(2*ax_size[0], 2*ax_size[1]))
        subsample.show_diagrams([current_subsample_diagram[0]],
                                labels=['Subsample'], title=f'$H_{0}$ Diagrams', as_barcodes=True, colormap=mpl_style, ax=ax1, alpha=0.5)
        subsample.show_diagrams([current_subsample_diagram[0]],
                                labels=['Subsample'], title=f'$H_{0}$ Diagrams', as_lifetimes=True, colormap=mpl_style, ax=ax2, alpha=0.5)
        prev_subsample_diagram = current_subsample_diagram


def stanford_bunny():
    # load data
    data_dir = Path('/home/seangrate/Projects/data/bunny/data/')
    data = data_utils.load_bunny(data_dir)

    # resize to fit unit cube
    data = (data - np.amin(data, axis=0).T) / (np.amax(data, axis=0) - np.amin(data, axis=0)).reshape((-1,3))
    visuals.vis_bunny(data)

    # compute PD
    num_subsample_points = 100
    num_target_points = 1000
    num_processes = 100
    max_dim = 1

    subsample_diagrams, target_diagrams = subsample.subsample(data, num_subsample_points, num_target_points,
                                                            num_processes=num_processes, max_dim=max_dim)
    # plot PDs
    for dim, pairs in enumerate(subsample_diagrams):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(2*ax_size[0], 2*ax_size[1]))
        subsample.show_diagrams([subsample_diagrams[dim], target_diagrams[dim]], 
                                labels=['Subsample', 'Target'], title=f'$H_{dim}$ Diagrams', colormap=mpl_style, ax=ax1, alpha=0.25)
        subsample.show_diagrams([subsample_diagrams[dim], target_diagrams[dim]],
                                labels=['Subsample', 'Target'], title=f'$H_{dim}$ Lifetimes', as_lifetimes=True, colormap=mpl_style, ax=ax2, alpha=0.25)

    num_epochs = 10
    prev_subsample_diagram = subsample_diagrams
    for epoch in tqdm(range(num_epochs)):
        if epoch == 0:
            current_subsample_diagram = ripser(prev_subsample_diagram[1])['dgms']
        else:
            # keep only points without inf value
            inf_mask = np.isinf(prev_subsample_diagram[0][:,1])
            current_subsample_diagram = ripser(prev_subsample_diagram[0][~inf_mask], maxdim=0)['dgms']
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(2*ax_size[0], 2*ax_size[1]))
        subsample.show_diagrams([current_subsample_diagram[0]],
                                labels=['Subsample'], title=f'$H_{0}$ Diagrams', as_barcodes=True, colormap=mpl_style, ax=ax1, alpha=0.25)
        subsample.show_diagrams([current_subsample_diagram[0]],
                                labels=['Subsample'], title=f'$H_{0}$ Diagrams', as_lifetimes=True, colormap=mpl_style, ax=ax2, alpha=0.25)
        prev_subsample_diagram = current_subsample_diagram


def anova():
    clusters = np.vstack([cluster for cluster in [rng.uniform(0,4,size=(5,2)),
                                                  rng.uniform(5,8,size=(5,2)),
                                                  rng.uniform(10,15,size=(5,2))]])

    # plt.scatter(clusters[:,0], clusters[:,1])
    # plt.show()
    # plt.clf()

    diagrams = ripser(clusters, maxdim=0)['dgms']
    diagrams[0][np.isinf(diagrams[0])] = np.amax(diagrams[0][np.isfinite(diagrams[0])]) + 10*np.finfo(np.float32).eps

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(2*ax_size[0], 2*ax_size[1]))
    subsample.show_diagrams(diagrams, 
                            labels=['Target'], title=f'$H_{0}$ Diagrams', as_barcodes=True, ax=ax1, colormap=mpl_style, alpha=0.75)
    subsample.show_diagrams(diagrams, 
                            labels=['Target'], title=f'$H_{0}$ Diagrams', ax=ax2, colormap=mpl_style, alpha=0.75)

    d_anovas = []
    num_epochs = 20
    prev_diagram = diagrams
    prev_diagrams = []
    curr_diagrams = []
    for epoch in tqdm(range(1, num_epochs+1)):
        if epoch == 1:
            curr_diagram = ripser(prev_diagram[0], maxdim=0)['dgms']
            curr_diagram[0][np.isinf(curr_diagram[0])] = np.amax(curr_diagram[0][np.isfinite(curr_diagram[0])]) + 10*np.finfo(np.float32).eps
        else:
            # keep only points without inf value
            curr_diagram = ripser(prev_diagram[0], maxdim=0)['dgms']
            curr_diagram[0][np.isinf(curr_diagram[0])] = np.amax(curr_diagram[0][np.isfinite(curr_diagram[0])]) + 10*np.finfo(np.float32).eps
            # if prev_diagram[0].shape[0] != curr_diagram[0].shape[0]:
                # print(f'Changed at epoch {epoch}') 
        # normalize diagrams to unit square
        prev_diagram[0] = prev_diagram[0] / np.amax(prev_diagram[0])
        curr_diagram[0] = curr_diagram[0] / np.amax(curr_diagram[0])
        # keep track of diagrams
        prev_diagrams.append(prev_diagram)
        curr_diagrams.append(curr_diagram)
        
        # plot diagrams
        fig, ax = plt.subplots(2, 2, figsize=(2*ax_size[0], 2*ax_size[1]))
        for i, diagram in enumerate([prev_diagram, curr_diagram]):
            subsample.show_diagrams([diagram[0]],
                                    labels=['Target'], title=f'$H_{0}$ Diagrams (Epoch {epoch - (1-i)})', as_barcodes=True, ax=ax[i][0], colormap=mpl_style, alpha=0.75)
            subsample.show_diagrams([diagram[0]],
                                    labels=['Target'], title=f'$H_{0}$ Diagrams', as_lifetimes=True, ax=ax[i][1], colormap=mpl_style, alpha=0.75)
        fig.subplots_adjust(hspace=0.5)
        plt.savefig(f'./plots/epoch_{epoch}.png')
        plt.close()

        # compute d_anova measure
        d_anova = subsample.compare_diagrams(prev_diagram[0], curr_diagram[0])
        d_anovas.append(d_anova)
        prev_diagram = curr_diagram
        epoch += 1
    
    print()
    # IMPLEMENT: consensus voting with number of peaks to determine average number of features predicted
    # IDEA: weight higher peaks more
    peak_idx = np.argmax(d_anovas)
    barcode_diagram, persistence_diagram = prev_diagrams[peak_idx], prev_diagrams[peak_idx]
    print(f'Number of features: {subsample.count_features(barcode_diagram[0], persistence_diagram[0], 0.1)}')
    if (peak_idx != 0) and (peak_idx != num_epochs):
        with np.printoptions(precision=4, suppress=True):
            for idx in [peak_idx-1, peak_idx, peak_idx+1]:
                print(f'{prev_diagrams[idx][0][:,1]}')
            print()
            print(f'{prev_diagrams[peak_idx][0][:,1] - prev_diagrams[peak_idx-1][0][:,1]}')
            print(f'{prev_diagrams[peak_idx+1][0][:,1] - prev_diagrams[peak_idx][0][:,1]}')
    # plot d_anova over time
    plt.clf()
    plt.plot([k for k, _ in enumerate(d_anovas, start=1)], d_anovas)
    plt.ylabel('$d_{anova}$')
    plt.xlabel('(New) Epoch')
    plt.xticks([k for k, _ in enumerate(d_anovas, start=1)])
    plt.tight_layout()
    plt.show()

    visuals.vis_minecraft(data, as_voxels=True)


if __name__ == '__main__':
    anova()