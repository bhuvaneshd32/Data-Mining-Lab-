import numpy as np

n = 8
data = ['HS', 'BS', 'MS', 'PhD', 'BS', 'HS', 'MS', 'PhD']
categories = ['HS', 'BS', 'MS', 'PhD']
M = len(categories)

print("Data:", data)
print("Categories (ordered):", categories)
print("M =", M)
print()

def rank_category(value, categories):
    return categories.index(value) + 1

def normalize_rank(rank, M):
    if M <= 1:
        return 0
    return (rank - 1) / (M - 1)

def compute_normalized_scores(data, categories):
    n = len(data)
    normalized = np.zeros(n)
    for i in range(n):
        rank = rank_category(data[i], categories)
        normalized[i] = normalize_rank(rank, M)
    return normalized

def compute_distance_matrix(normalized_scores):
    n = len(normalized_scores)
    return np.abs(normalized_scores[:, None] - normalized_scores[None, :])

normalized_scores = compute_normalized_scores(data, categories)
dist_matrix = compute_distance_matrix(normalized_scores)

print("Normalized scores:", normalized_scores)
print("\nDistance Matrix:")
np.set_printoptions(precision=4, suppress=True)
print(dist_matrix)