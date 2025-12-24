import numpy as np

# ======== GIVEN =========
# Don't modify this part
n, m = 6, 4  # Default: 6 data points, 4 binary attributes

# Initialize sample binary data (1=Y, 0=N)
data = np.array([
    [1, 0, 1, 0],  # Point 0
    [0, 1, 0, 1],  # Point 1
    [1, 1, 1, 1],  # Point 2
    [0, 0, 0, 0],  # Point 3
    [1, 1, 0, 0],  # Point 4
    [0, 0, 1, 1]   # Point 5
])

print("Given Binary Data Matrix (1=Y, 0=N):")
print("Shape:", data.shape)
print("Data (n={}, m={}):".format(n, m))
print(data)
print()

# ======== TO IMPLEMENT =========
# Fill these functions

def compute_contingency_table(point_i, point_j):
    """
    Compute contingency table counts for two binary points
    point_i, point_j: arrays of length m (binary values 0/1)
    Returns: q, r, s, t (integers)
    """
    # YOUR CODE HERE
    # q = count where both are 1
    # r = count where i=1, j=0
    # s = count where i=0, j=1
    # t = count where both are 0
    q,r,s,t = 0,0,0,0
    for i in range(m):
        if point_i[i] == point_j[i] == 1 :
            q+=1
        if point_i[i] == point_j[i] == 0 :
            t+=1
        if point_i[i] == 1 and point_j[i] == 0 :
            r+=1
        if point_i[i] == 0 and point_j[i] == 1 :
            s+=1
    return q,r,s,t
        

def jaccard_distance(q, r, s, t):
    """
    Calculate Jaccard distance from contingency table counts
    Formula: (r + s) / (q + r + s)
    Ignores t (negative matches)
    Returns: float distance between 0 and 1
    """
    if q+r+s == 0 :
        return 0
    else:
        return (r+s)/(q+r+s)

def compute_distance_matrix(data, distance_type='jaccard'):
    """
    Compute distance matrix for binary data
    data: numpy array of shape (n, m) with binary values
    distance_type: 'jaccard', 'smc', or 'rogers'
    Returns: numpy array of shape (n, n) with distance values
    
    For each pair (i, j):
    1. Compute q, r, s, t
    2. Calculate distance based on distance_type
    3. Store in matrix
    """
    n, m = data.shape
    dist_matrix = np.zeros((n, n))
    
    # YOUR CODE HERE
    # For each pair of points:
    # 1. Call compute_contingency_table
    # 2. Calculate distance using appropriate formula
    # 3. Fill dist_matrix[i][j]
    
    # Note: Matrix should be symmetric and diagonal = 0
    for i in range(n):
        for j in range(n):
             q,r,s,t = compute_contingency_table(data[i],data[j])
             dist_matrix[i][j] = jaccard_distance(q,r,s,t)

    return dist_matrix

# ======== MAIN EXECUTION =========
# Don't modify this part
if __name__ == "__main__":
    # Compute distance matrix using Jaccard distance
    dist_matrix = compute_distance_matrix(data, distance_type='jaccard')
    
    print("=" * 60)
    print("Distance Matrix (Jaccard Distance):")
    print("=" * 60)
    np.set_printoptions(precision=4, suppress=True)
    print(dist_matrix)
    
    # Display contingency table for first two points
    print("\n" + "=" * 60)
    print("Contingency Table for Points 0 and 1:")
    print("=" * 60)
    q, r, s, t = compute_contingency_table(data[0], data[1])
    print(f"q (both 1) = {q}")
    print(f"r (i=1, j=0) = {r}")
    print(f"s (i=0, j=1) = {s}")
    print(f"t (both 0) = {t}")
    print(f"Total attributes m = {q + r + s + t}")
    
    # Verification and analysis
    print("\n" + "=" * 60)
    print("Analysis:")
    print("=" * 60)
    
    # Check properties
    print(f"1. Matrix shape: {dist_matrix.shape}")
    print(f"2. Is symmetric? {np.allclose(dist_matrix, dist_matrix.T)}")
    print(f"3. Diagonal all zeros? {np.all(np.diag(dist_matrix) == 0)}")
    print(f"4. All values between 0 and 1? {(dist_matrix >= 0).all() and (dist_matrix <= 1).all()}")
    
