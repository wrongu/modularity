import torch
from collections import deque
from tqdm import trange, tqdm


def is_valid_adjacency_matrix(adj:torch.Tensor, enforce_sym=False, enforce_no_self=False, enforce_binary=False) -> bool:
    valid = torch.all(adj >= 0.)
    if enforce_binary:
        valid = valid and torch.all(torch.logical_or(adj == 0., adj == 1.))
    if enforce_no_self:
        valid = valid and torch.all(adj.diag() == 0.)
    if enforce_sym:
        valid = valid and torch.allclose(adj, adj.T)
    return valid


def girvan_newman(adj:torch.Tensor, clusters:torch.Tensor):
    """Compute GN modularity statistic on adjacency matrix A using cluster assignments P.

    This version follows the equations in the GN paper fairly literally. The first line computes the cluster-cluster
    connectivity matrix 'e', which is then summed along rows or columns to get expected connectivity.
    """
    e = (clusters.T @ adj @ clusters) / torch.sum(adj)
    e_sum_row = torch.sum(e, dim=0)
    e_sum_col = torch.sum(e, dim=1)
    return torch.trace(e) - torch.sum(e_sum_col * e_sum_row)


def girvan_newman_sym(adj:torch.Tensor, clusters:torch.Tensor):
    """Compute GN modularity statistic on symmetric adjacency matrix A using cluster assignments P.

    This version makes two assumptions: that A is symmetric, and that the rows of P sum to 1. These assumptions are
    exploited to reduce redundant calculations for a slight speedup over girvan_newman().
    """
    uB = adj @ clusters
    # Since P sums to 1 along rows, this is equivalent to sum(A) but on a much smaller matrix
    B = uB / torch.sum(uB)
    # The following is equivalent to Trace(P'*A*P) but without transposing and without computing off-diagonal elements
    # that would be dropped by trace anyway
    cluster_connectivity = torch.sum(clusters * B)
    # When A is symmetric, we don't need to compute the expected connectivity separately for rows and columns... just
    # compute once and multiply by itself for the trace. In the GN paper, this corresponds to computing the expected
    # connectivity statistic row- or column-wise (a_i or a_j).
    Brow = torch.sum(B, dim=0)
    expected_connectivity = torch.sum(Brow * Brow)
    # GN modularity statistic is the actual cluster connectivity minus the baseline 'expected' connectivity
    return cluster_connectivity - expected_connectivity


def spectral_modularity(adj, max_clusters=None) -> torch.Tensor:
    """Spectral algorithm to quickly find an approximate maximum for the Girvan-Newman modularity score.

    See Newman, M. E. J. (2006). Modularity and community structure in networks. PNAS, 103(23), 8577–8582.
    """
    n = adj.size(0)
    max_clusters = max_clusters or n
    # Normalize by sum of all 'edges' / AKA convert to probability assuming all A>=0
    adj = adj / adj.sum()
    # Compute degree of each 'vertex' / AKA compute marginal probability
    A1 = adj.sum(dim=1, keepdims=True)
    is_dead = A1.flatten() == 0.
    # Compute modularity matrix 'B': connectivity minus expected random connectivity
    B = adj - A1 * A1.T

    def gn_score(clusters):
        # Local helper function to compute Girvan-Newman modularity score using intermediate matrix 'B'
        return torch.sum(B * (clusters @ clusters.T))

    # Initialize everything into a single cluster, pruning out 'dead' units right away
    clusters = torch.zeros(n, max_clusters, device=adj.device)
    clusters[~is_dead, 0] = 1.
    best_score = gn_score(clusters)
    # Iteratively subdivide until modularity score is not improved. This is a greedy method that takes any high-level
    # split that improves total modularity score. Each split is a branch of a binary tree, so after 'l' splits there
    # may be as many as 2^l modules, but some branches may be pruned early. This 'tree' is traversed breadth-first.
    # Tree traversal simply means keeping track of which column to try splitting next, which we keep track of in a
    # FIFO queue.
    queue, next_avail_col = deque([0]), 1
    while len(queue) > 0 and next_avail_col < max_clusters:
        col = queue.popleft()
        # Which variables are we working with here
        mask = clusters[:, col] == 1.
        # Skip this division if only one element left
        if mask.sum() <= 1:
            continue
        # Isolate the submatrix of B just containing these variables
        Bsub = B[mask, :][:, mask]
        # Compute single top eigenvectors for this sub-matrix
        _, _, v = torch.svd(Bsub)
        # Propose a split based on the first eigenvector that contains differing signs
        has_alternating_signs = torch.logical_and(torch.any(v > 0, dim=0), torch.any(v < 0, dim=0))
        subdiv = v[:, has_alternating_signs][:, 0]
        # Try new subdivision out: use current 'col' for (+) side of v, and use 'next_avail_col' for (-) side
        clusters[mask, col] = (subdiv > 0).float()
        clusters[mask, next_avail_col] = (subdiv < 0).float()
        subdiv_score = gn_score(clusters)
        # If improved, keep it and push potential further subdivisions onto the queue
        if subdiv_score > best_score:
            queue.extend([col, next_avail_col])
            next_avail_col += 1
            best_score = subdiv_score
        # If not improved, undo change to 'clusters'
        else:
            clusters[:, col] = mask.float()
            clusters[:, next_avail_col] = 0.

    # Sanity check: only those units initially pruned as 'dead' should be missing
    assert torch.all((clusters.sum(dim=1) == 0.) == is_dead)

    return clusters


def monte_carlo_modularity(adj, clusters=None, max_clusters=None, steps=10000, temperature=1.0):
    """Optimize GN modularity by initializing using spectral method, then repeatedly shuffling values around with
    probability proportional to how good the GN score is after that shuffle. Return the best-scoring clustering
    """
    if clusters is None:
        clusters = spectral_modularity(adj, max_clusters=max_clusters)
    max_clusters = clusters.size(1)
    best_score, best_clusters = girvan_newman_sym(adj, clusters).cpu(), clusters.clone()
    scores_history = torch.zeros(steps)
    # Some elements are disconnected – don't bother moving these around
    is_dead = adj.sum(dim=1) == 0.
    # Pick a random item to shuffle around
    ishuffle = torch.multinomial((~is_dead).float(), num_samples=steps, replacement=True)
    for i, idx in tqdm(enumerate(ishuffle), desc='MC shuffles', leave=False, total=steps):
        # Consider all possible re-assignments, but only consider creating a new column at most 1 time
        scores, did_new_col = float('-inf')*torch.ones(max_clusters), False
        for j in range(max_clusters):
            is_new_col = clusters[:, j].sum() == 0.
            if is_new_col and did_new_col:
                continue
            else:
                did_new_col = did_new_col or is_new_col
                # Re-assign vertex 'idx' to cluster 'j' and see what the score would be
                clusters[idx, :] = 0.
                clusters[idx, j] = 1.
                scores[j] = girvan_newman_sym(adj, clusters).cpu()
                # Track best-scoring clusters we've seen across all proposals
                if scores[j] > best_score:
                    best_score = scores[j]
                    best_clusters = clusters.clone()
        # Pick randomly among decent-scoring options
        choice = torch.multinomial(torch.exp((scores - scores.max())/temperature), num_samples=1)
        clusters[idx, :] = 0.
        clusters[idx, choice] = 1.
        scores_history[i] = scores[choice]

    # Sanity check: only those units initially pruned as 'dead' should be missing
    assert torch.all((best_clusters.sum(dim=1) == 0.) == is_dead)

    return best_clusters, scores_history


def gradient_ascent_modularity(adj, max_k=None, steps=5000, num_init=100):
    n = adj.size(0)
    max_k = max_k or n
    best_init, best_score = None, torch.tensor(float('-inf'))
    for _ in range(num_init):
        log_clusters = torch.randn(n, max_k, device=adj.device)
        tmp_score = girvan_newman_sym(adj, torch.softmax(log_clusters, dim=1))
        if tmp_score > best_score:
            best_score = tmp_score
            best_init = log_clusters
    log_clusters = best_init
    log_clusters.requires_grad_(True)
    scores = torch.zeros(steps, device=adj.device)
    opt = torch.optim.Adam([log_clusters], lr=0.005)
    for s in trange(steps, desc='GN gradient ascent'):
        opt.zero_grad()
        gn = girvan_newman_sym(adj, torch.softmax(log_clusters, dim=1))
        (-gn).backward()
        opt.step()

        with torch.no_grad():
            scores[s] = gn.item()

    return torch.softmax(log_clusters.detach(), dim=1).cpu(), scores