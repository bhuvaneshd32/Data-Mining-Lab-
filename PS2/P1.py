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
    for i in range(m):
        a_b +=  a[i]*b[i]
        


def compute_distance_matrix(data):
    """
    Compute Euclidean distance matrix
    data: numpy array of shape (n, m)
    Returns: numpy array of shape (n, n)
    """
    n = data.shape[0]
    dist_matrix = np.zeros((n, n))
    
    # YOUR CODE HERE
    # Fill the matrix using euclidean_distance()
    # Diagonal should be 0
    pass

def compute_similarity_matrix(data):
    """
    Compute cosine similarity matrix
    data: numpy array of shape (n, m)
    Returns: numpy array of shape (n, n)
    """
    n = data.shape[0]
    sim_matrix = np.zeros((n, n))
    
    # YOUR CODE HERE
    # Fill the matrix using cosine_similarity()
    # Diagonal should be 1
    pass

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
    
    # Verification
    print("\n" + "=" * 50)
    print("Verification:")
    print("=" * 50)
    print(f"1. Distance matrix symmetric? {np.allclose(dist_matrix, dist_matrix.T)}")
    print(f"2. Diagonal all zeros? {np.all(np.diag(dist_matrix) == 0)}")
    print(f"3. Similarity matrix diagonal all ones? {np.all(np.diag(sim_matrix) == 1)}")