# compute mahalanobis distance for a measurement
import numpy as np
import scipy as sp
from ipdb import set_trace as st

def mahalanobis_distance(obs, data):
    # st()
    obs_min_mu = obs - np.mean(data, axis=0)
    cov = np.cov(data.T)
    inv_cov = sp.linalg.pinv(cov)
    first_term = np.dot(obs_min_mu,inv_cov)
    mal_dist_sq = np.dot(first_term, obs_min_mu.T)
    mal_dist = np.sqrt(mal_dist_sq)

    sp_mal_dist = sp.spatial.distance.mahalanobis(obs, np.mean(data, axis=0), inv_cov)
    assert sp_mal_dist == mal_dist
    return mal_dist
