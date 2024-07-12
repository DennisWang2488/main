import numpy as np

def genDataRd(num_data, num_features, num_items, seed=42, Q=100, dim=1, deg=1, noise_width=0.5, epsilon=0.1):
    """
    A function to generate synthetic data for alpha fairness problem

    Args:
        num_data (int): number of data points
        num_items (int): number of items
        num_features (int): number of features
        seed (int): random state seed
        Q (float): Total available quantity

    Returns:
       tuple: x (np.ndarray), r (np.ndarray), c (np.ndarray), Q (float)
    """
    # Set random seed
    rnd = np.random.RandomState(seed)
    n = num_data
    p = num_features
    m = num_items
    
    x = rnd.normal(0, 1, (n, m, p))
    B = rnd.binomial(1, 0.5, (m, p))

    c = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            # cost without noise
            values = (np.dot(B[j], x[i, j].reshape(p, 1)).T / np.sqrt(p) + 3) ** deg + 1
            # rescale
            values *= 5
            values /= 3.5 ** deg
            # noise
            epislon = rnd.uniform(1 - noise_width, 1 + noise_width, 1)
            values *= epislon
            # convert into int
            values = np.ceil(values)
            c[i, j] = values

    # float
    c = c.astype(np.float64)

    r = rnd.normal(0, 1, (n, m))
    r = 1 / (1 + np.exp(-r))

    return x, r, c, Q
