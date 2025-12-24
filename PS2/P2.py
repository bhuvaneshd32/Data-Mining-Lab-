import numpy as np

# ======== GIVEN =========
# Don't modify this part
n, p = 5, 3  # Default: 5 data points, 3 attributes

# Initialize sample nominal data matrix
# Each row is a data point, each column is an attribute
data = np.array([
    ['Red', 'Male', 'NYC'],
    ['Blue', 'Male', 'NYC'],
    ['Blue', 'Female', 'LA'],
    ['Red', 'Female', 'Chicago'],
    ['Green', 'Male', 'NYC']
])

print("Given Nominal Data Matrix:")
print("Shape:", data.shape)
print("First 5 rows:")
for i in range(min(n, 5)):
    print(f"Point {i}: {data[i]}")
print()

# ======== TO IMPLEMENT =========
# Fill these functions

def count_matching_attributes(point1, point2):
    """
    Count number of matching attributes between two points
    point1, point2: arrays of length p (nominal values)
    Returns: integer count of matching attributes
    """
    matching_count = 0 
    for i in range(len(point1)):
        if point1[i] == point2[i] : 
            matching_count+=1
    
    return matching_count


def compute_dissimilarity_matrix(data):
    """
    Compute dissimilarity matrix for nominal data
    data: numpy array of shape (n, p) with string values
    Returns: numpy array of shape (n, n) with dissimilarity values
    
    Formula: dissimilarity(x, y) = (p - m) / p
    where: p = number of attributes
           m = number of matching attributes between x and y
    """
    n, p = data.shape
    dissim_matrix = np.zeros((n, n))
    
    # YOUR CODE HERE
    # Fill the matrix using the formula
    # For each pair (i, j):
    # 1. Count matching attributes m
    # 2. Calculate (p - m) / p
    # 3. Store in dissim_matrix[i][j]
    
    # Note: Matrix should be symmetric and diagonal = 0
    for i in range(n):
        for j in range(n):
            m = count_matching_attributes(data[i],data[j])
            dissim_matrix[i][j] = (p-m)/p

    return dissim_matrix

# ======== MAIN EXECUTION =========
# Don't modify this part
if __name__ == "__main__":
    dissim_matrix = compute_dissimilarity_matrix(data)
    
    print("=" * 60)
    print("Dissimilarity Matrix:")
    print("=" * 60)
    np.set_printoptions(precision=4, suppress=True)
    print(dissim_matrix)
    
    # Verification and analysis
    print("\n" + "=" * 60)
    print("Analysis:")
    print("=" * 60)
    
    # Check properties
    print(f"1. Matrix shape: {dissim_matrix.shape}")
    print(f"2. Is symmetric? {np.allclose(dissim_matrix, dissim_matrix.T)}")
    print(f"3. Diagonal all zeros? {np.all(np.diag(dissim_matrix) == 0)}")
    print(f"4. All values between 0 and 1? {(dissim_matrix >= 0).all() and (dissim_matrix <= 1).all()}")
    
    
# ======== EXTRA: FILE READING VERSION =========
"""
If you need to read from file, use this function:
def read_nominal_data(filename):
    with open(filename, 'r') as f:
        # First line: n p
        n, p = map(int, f.readline().split())
        
        data = []
        for _ in range(n):
            line = f.readline().strip()
            # Split and handle quoted strings if needed
            if ',' in line:
                # CSV format
                values = [x.strip().strip('"') for x in line.split(',')]
            else:
                # Space-separated
                values = line.split()
            data.append(values)
            
    return np.array(data)
"""