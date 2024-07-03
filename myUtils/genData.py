import numpy as np

def genDataAr(num_data, num_features, num_items, dim=1, deg=1, noise_width=0, seed=42, Q=1000, epsilon=0.1):
    """
    A function to generate synthetic data for alpha fairness E2E learning

    Args:
        num_data (int): number of data points
        num_features (int): dimension of features
        num_items (int): number of items
        dim (int): dimension of multi-dimensional knapsack
        noise_width (float): half width of data random noise
        seed (int): random state seed

    Returns:
       tuple: a, b, c (np.ndarray), r (np.ndarray), features (np.ndarray)
    """
    # Set random seed
    rnd = np.random.RandomState(seed)
    
    # Number of data points
    n = num_data
    # Dimension of features
    p = num_features
    # Number of items
    m = num_items

    # Generate features
    x = rnd.normal(0, 1, (n, p))

    # Generate a, b, c parameters as non-negative real numbers
    a = rnd.uniform(0.5, 1.5, (n, m))
    b = rnd.uniform(0.5, 1.5, (n, m))
    r = rnd.uniform(0, 1, (n, m))
    B = rnd.binomial(1, 0.5, (m, p))
    c = np.zeros((n, m), dtype=int)
    for i in range(n):
        # cost without noise
        values = (np.dot(B, x[i].reshape(p, 1)).T / np.sqrt(p) + 3) ** deg + 1
        # rescale
        values *= 5
        values /= 3.5 ** deg
        # noise
        epislon = rnd.uniform(1 - noise_width, 1 + noise_width, m)
        values *= epislon
        # convert into int
        values = np.ceil(values)
        c[i, :] = values
        # float
        c = c.astype(np.float64)

    return a, b, c, r, x, Q, epsilon

def genDataRd(num_data, num_features, num_items, seed=42, Q=100, dim=1, deg=1, noise_width=0, epsilon=0.1):
    """
    A function to generate synthetic data for alpha fairness problem

    Args:
        num_data (int): number of data points
        num_items (int): number of items
        seed (int): random state seed
        Q (float): Total available quantity

    Returns:
       tuple: r (np.ndarray), c (np.ndarray), Q (float)
    """
    # Set random seed
    rnd = np.random.RandomState(seed)
    n = num_data
    p = num_features
    m = num_items
    
    x = rnd.normal(0, 1, (n, p))
    B = rnd.binomial(1, 0.5, (m, p))

    c = np.zeros((n, m))
    for i in range(n):
        # cost without noise
        values = (np.dot(B, x[i].reshape(p, 1)).T / np.sqrt(p) + 3) ** deg + 1
        # rescale
        values *= 5
        values /= 3.5 ** deg
        # noise
        epislon = rnd.uniform(1 - noise_width, 1 + noise_width, m)
        values *= epislon
        # convert into int
        values = np.ceil(values)
        c[i, :] = values
        
        # float
        c = c.astype(np.float64)

    r = rnd.normal(0, 1, (n, m))
    r = 1 / (1 + np.exp(-r))

    return x, r, c, Q
