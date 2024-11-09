import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Take euclidean distance of two points
def distance(p1, p2):

    p1, p2 = np.array(p1), np.array(p2)

    dist = np.sqrt(np.sum(np.square(p1 - p2)))

    return dist


# Assign each point to a cluster
def assign_clusters(X, centroids):
    clusters = []

    for x in X.values:
        min_dist = float('inf')
        closest_centroid = -1

        for idx, mu in enumerate(centroids):
            dist = distance(x, mu)
            # Update if this centroid is closer
            if dist < min_dist:
                min_dist = dist
                closest_centroid = idx

        # Append the index of the closest centroid for this data point
        clusters.append(closest_centroid)

    return clusters


def update_centroids(X, clusters, k):
    # Initialize a list to hold the new centroids
    new_centroids = []

    # Iterate over each cluster index
    for cluster_idx in range(k):
        # Points in cluster
        pts = []
        for idx, pt in enumerate(clusters):
            if pt == cluster_idx:
                pts.append(X.iloc[idx])
        # If there are points in the cluster, calculate the mean
        if pts:
            # Convert pts to a DataFrame to calculate the mean of each feature
            pts_df = pd.DataFrame(pts)
            new_centroid = pts_df.mean().values
            new_centroids.append(new_centroid)
        else:
            new_centroids.append(np.zeros(X.shape[1]))
    return np.array(new_centroids)


# Find objective function for plotting
def calculate_objective(X, clusters, centroids):
    total_distance = 0
    for idx, point in enumerate(X.values):
        centroid = centroids[clusters[idx]]
        total_distance += distance(point, centroid) ** 2
    return total_distance


# Main procedure
def main(k, data, max_iters=999, err=1e-4):
    # Read csv
    data = pd.read_csv(data)

    # Set seed for reproducibility
    np.random.seed(123)

    # Find number of features
    # Assume the last column is the target column
    features = data.shape[1]-1

    # Get feature columns
    X = data.iloc[:, :features]

    objective_values = []

    # Select K random data points as the initial centroids
    centroids = np.array(X.loc[np.random.choice(X.shape[0], k, replace=False)])

    for _ in range(max_iters):
        clusters = assign_clusters(X, centroids)

        new_centroids = update_centroids(X, clusters, k)

        # Calculate and store the objective value
        objective_value = calculate_objective(X, clusters, new_centroids)
        objective_values.append(objective_value)

        diff = np.abs(new_centroids - centroids)
        if np.all(diff <= err):
            print('Convergence Reached!')
            # Plot the objective function over iterations
            plt.figure(figsize=(8, 6))
            plt.plot(objective_values, marker='o')
            plt.title(f"Objective Function (Sum of Squared Distances) Over Iterations, k={k}")
            plt.xlabel("Iteration")
            plt.ylabel("Objective Value")
            plt.show()
            return centroids, clusters
        centroids = new_centroids

    print("No convergence...")

    # Plot the objective function over iterations
    plt.figure(figsize=(8, 6))
    plt.plot(objective_values, marker='o')
    plt.title(f"Objective Function (Sum of Squared Distances) Over Iterations, k = {k}")
    plt.xlabel("Iteration")
    plt.ylabel("Objective Value")
    plt.show()

    return centroids, clusters


main(2, 'irisdata.csv', max_iters=100)
