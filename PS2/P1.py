import numpy as np
import math

# ======== GIVEN =========
# Don't modify this part
n, m = 5, 3  # Default: 5 data points, 3 attributes

# Initialize simple nxm matrix
data = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
    [10, 11, 12],
    [13, 14, 15]
], dtype=float)

print("Given Data Matrix:")
print(data)
print(f"Shape: ({n}, {m})")
print()

# ======== TO IMPLEMENT =========
# Fill these functions

def euclidean_distance(a, b):
    """
    Calculate Euclidean distance between two points
    a, b: numpy arrays of length m
    Returns: float
    """
    # YOUR CODE HERE
    # Hint: sqrt(sum((a_i - b_i)^2))
    sum = 0
    for i in range(m):
        sum += ((a[i]-b[i])**2)
    
    return math.sqrt(sum)


def cosine_similarity(a, b):
    """
    Calculate cosine similarity between two points
    a, b: numpy arrays of length m
    Returns: float between 0 and 1
    """
    # YOUR CODE HERE
    # Hint: (a·b) / (||a|| * ||b||)
    a_b = 0
    mod_a = 0
    mod_b = 0
    for i in range(m):
        a_b +=  a[i]*b[i]
        mod_a += a[i]**2
        mod_b += b[i]**2
    
    return a_b/(math.sqrt(mod_a)*math.sqrt(mod_b))


def compute_distance_matrix(data):
    """
    Compute Euclidean distance matrix
    data: numpy array of shape (n, m)
    Returns: numpy array of shape (n, n)
    """
    n = data.shape[0]
    dist_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            dist_matrix[i][j] = euclidean_distance(data[i],data[j])

    return dist_matrix
        

def compute_similarity_matrix(data):
    """
    Compute cosine similarity matrix
    data: numpy array of shape (n, m)
    Returns: numpy array of shape (n, n)
    """
    n = data.shape[0]
    sim_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            sim_matrix[i][j] = cosine_similarity(data[i],data[j])
    
    return sim_matrix
    

# ======== MAIN EXECUTION =========
# Don't modify this part
if __name__ == "__main__":
    dist_matrix = compute_distance_matrix(data)
    sim_matrix = compute_similarity_matrix(data)
    
    print("=" * 50)
    print("Euclidean Distance Matrix:")
    print("=" * 50)
    np.set_printoptions(precision=4, suppress=True)
    print(dist_matrix)
    
    print("\n" + "=" * 50)
    print("Cosine Similarity Matrix:")
    print("=" * 50)
    print(sim_matrix)
    
    
    

