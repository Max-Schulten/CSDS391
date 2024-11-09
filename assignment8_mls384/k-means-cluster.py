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


# Plotting function for data points and centroids
def plot_clusters(X, centroids, clusters, title):
    plt.figure(figsize=(8, 6))
    for cluster_idx in range(len(centroids)):
        cluster_points = X.values[np.array(clusters) == cluster_idx]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {cluster_idx + 1}')
    centroids = np.array(centroids)
    plt.scatter(centroids[:, 0], centroids[:, 1], s=50, c='black', marker='x', label='Centroids')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(title)
    plt.legend()
    plt.show()


def plot_decision_boundaries_approx(X, centroids, clusters, k, title):
    # Define the range of the grid based on the first two features
    x_min, x_max = X.iloc[:, 0].min() - 1, X.iloc[:, 0].max() + 1
    y_min, y_max = X.iloc[:, 1].min() - 1, X.iloc[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05),
                         np.arange(y_min, y_max, 0.05))

    # Prepare the grid points to assign clusters
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    grid_df = pd.DataFrame(grid_points, columns=[X.columns[0], X.columns[1]])

    # Assign each grid point to the nearest centroid
    grid_clusters = assign_clusters(grid_df, centroids)
    grid_clusters = np.array(grid_clusters).reshape(xx.shape)

    # Plot decision boundaries by coloring each region
    plt.figure(figsize=(8, 6))
    plt.imshow(grid_clusters, extent=(x_min, x_max, y_min, y_max),
               origin='lower', cmap='viridis', alpha=0.3, interpolation='nearest')

    # Plot the actual data points and centroids
    for cluster_idx in range(k):
        cluster_points = X.values[np.array(clusters) == cluster_idx]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {cluster_idx + 1}')

    centroids = np.array(centroids)
    plt.scatter(centroids[:, 0], centroids[:, 1], s=50, c='black', marker='X', label='Centroids')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(title)
    plt.legend()
    plt.show()


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

    for i in range(max_iters):
        clusters = assign_clusters(X, centroids)

        new_centroids = update_centroids(X, clusters, k)

        # Calculate and store the objective value
        objective_value = calculate_objective(X, clusters, new_centroids)
        objective_values.append(objective_value)

        diff = np.abs(new_centroids - centroids)
        if np.all(diff <= err):
            print('Convergence Reached!')

            """
            plt.figure(figsize=(8, 6))
            plt.plot(objective_values, marker='o')
            plt.title(f"Objective Function (Sum of Squared Distances) Over Iterations, k={k}")
            plt.xlabel("Iteration")
            plt.ylabel("Objective Value")
            plt.show()
            """
            plot_decision_boundaries_approx(X.iloc[:, :2], centroids, clusters, k, title=f"Decision Boundaries for $k = {k}$")
            return centroids, clusters
        centroids = new_centroids


    print("No convergence...")

    """
    # Plot the objective function over iterations
    plt.figure(figsize=(8, 6))
    plt.plot(objective_values, marker='o')
    plt.title(f"Objective Function (Sum of Squared Distances) Over Iterations, k = {k}")
    plt.xlabel("Iteration")
    plt.ylabel("Objective Value")
    plt.show()
    """

    return centroids, clusters


main(2, 'irisdata2.csv', max_iters=100)
