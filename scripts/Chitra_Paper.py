import os
import hydra
from omegaconf import DictConfig
import numpy as np
from src.networks import instantiate_network
from src.utils.utils import create_pi_list, degree_matrix, gh_rank, hg_rank
import scipy
# Chitra_Paper.py
import matplotlib.pyplot as plt
"""
Script for analysis and experiments related to the Chitra Paper.
This script is part of the HyperDynamics project.
"""

#Run simulations
sigma_list=[1]
p_list = [0.03, 0.05, 0.07]
num_trials = 10 #number of trials for each (sigma, p) pair
n = 100 #number of elements to collectively rank

def ranking(n, rank_agg_fns, names, sigma_list, p_list, num_trials):
    sl = len(sigma_list)
    pl = len(p_list)

    wkt_means_hg = np.zeros([sl, pl])
    wkt_stdevs_hg = np.zeros([sl, pl])
    
    wkt_means_gh = np.zeros([sl, pl])
    wkt_stdevs_gh = np.zeros([sl, pl])

    for ind1 in range(len(sigma_list)):
        for ind2 in range(len(p_list)):
            sigma = sigma_list[ind1]
            p = p_list[ind2]
            # store results of trials
            hg_wkt_list = np.zeros([num_trials])
            gh_wkt_list = np.zeros([num_trials])
            dwork_wkt_list = np.zeros([num_trials])

            for k in range(num_trials):
                # give each element a mean score from [0, 10]
                means = np.zeros([n])
                for i in range(n):
                    means[i] = i

                # create partial rankings
                # for each elt, create 2 partial rankings with that elt?
                # ill start with 1
                universe, pi_list = create_pi_list(n, n, p, means, sigma) #middle is s, currently using s=n
                # true ranking
                # lol the -1 bc i still want to use 1-indexing
                true_ranking = [x for _, x in sorted(zip(means[universe - 1], universe), key=lambda pair: pair[0])]

                # compute ranking for HG
                hg_ranking = hg_rank(universe, pi_list)
                hg_wkt_dist = scipy.stats.weightedtau(hg_ranking, true_ranking).correlation
                hg_wkt_list[k] = hg_wkt_dist
                # compute ranking for G^H
                gh_ranking = gh_rank(universe, pi_list)
                gh_wkt_dist = scipy.stats.weightedtau(gh_ranking, true_ranking).correlation
                gh_wkt_list[k] = gh_wkt_dist
                
            
            # compute mean, stdev for each fn and add to kt_means, kt_stdevs [and wkt variants]
            wkt_means_hg[ind1, ind2] = np.mean(hg_wkt_list)
            wkt_stdevs_hg[ind1, ind2] = np.std(hg_wkt_list)
            
            wkt_means_gh[ind1, ind2] = np.mean(gh_wkt_list)
            wkt_stdevs_gh[ind1, ind2] = np.std(gh_wkt_list)

    return wkt_means_hg, wkt_stdevs_hg, wkt_means_gh, wkt_stdevs_gh

if __name__ == "__main__":
    print('hello')
    # Example usage
    rank_agg_fns = [gh_rank, hg_rank]
    names = ["gh", "hg"]
    wkt_means_hg, wkt_stdevs_hg, wkt_means_gh, wkt_stdevs_gh = ranking(n, rank_agg_fns, names, sigma_list, p_list, num_trials)
    
    print("HG WKT Means:")
    print(wkt_means_hg)
    print("HG WKT Stdevs:")
    print(wkt_stdevs_hg)
    
    print("GH WKT Means:")
    print(wkt_means_gh)
    print("GH WKT Stdevs:")
    print(wkt_stdevs_gh)

    ind = np.arange(3)  # the x locations for the groups
    width = 0.15       # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(ind, wkt_means_hg[0,:], width, color='r', yerr=wkt_stdevs_hg[0,:])
    rects3 = ax.bar(ind + 2*width, wkt_means_gh[0,:], width, color='b', yerr=wkt_stdevs_gh[0,:])

    # add some text for labels, title and axes ticks
    ax.set_xlabel('p, Proportion of Vertices in each Partial Ranking')
    ax.set_ylabel('Weighted Kendall Tau Distance')
    ax.set_title('Hypergraph vs Dwork Performance in Rank Aggregation')
    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels(('0.03', '0.05', '0.07'))
    ax.set_ylim(bottom=0.3,top=0.95)

    ax.legend((rects1[0], rects3[0]), ('Hypergraph', 'Dwork', 'Clique Graph'), prop=dict(size=7), loc='best')
    plt.savefig('hg_dwork_clique_sigma_1.pdf')