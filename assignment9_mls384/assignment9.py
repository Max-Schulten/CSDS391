import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def sigmoid(x):
    return 1/(1+np.exp(-x))


def oneLayerNN(data, weights):
    return sigmoid(np.dot(data, weights))


def decisionBoundaryPlot(data, weights):
    class_2 = data[data['species'] == 'versicolor']
    class_3 = data[data['species'] == 'virginica']

    plt.figure(figsize=(10, 6))

    plt.scatter(class_2['petal_length'], class_2['petal_width'],
                label="Iris-versicolor", alpha=0.7)

    # plt.scatter(class_1['petal_length'], class_1['petal_width'],
    #            label="Iris-versicolor", alpha=0.7)

    plt.scatter(class_3['petal_length'], class_3['petal_width'],
                label="Iris-virginica", alpha=0.7)

    x_min, x_max = data['petal_length'].min() - 0.5, data['petal_length'].max() + 0.5

    y_min, y_max = data['petal_width'].min() - 0.5, data['petal_width'].max() + 0.5

    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))

    grid_points = np.c_[xx.ravel(), yy.ravel()]
    grid_points = np.c_[grid_points, np.ones(grid_points.shape[0])]
    outputs = oneLayerNN(grid_points, weights)

    outputs = outputs.reshape(xx.shape)
    plt.contour(xx, yy, outputs, levels=[0.5], colors='red', linewidths=2)

    # Add labels and title
    plt.xlabel('Petal Length')
    plt.ylabel('Petal Width')
    plt.title('Decision Boundary Overlaid on Iris Data')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_surface(weights, bias, x_range, y_range):
    # Generate a grid of points over the feature space
    xx, yy = np.meshgrid(np.linspace(x_range[0], x_range[1], 100),
                         np.linspace(y_range[0], y_range[1], 100))

    # Flatten the grid and compute model outputs
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    outputs = oneLayerNN(grid_points, weights, bias)

    # Reshape the output to match the grid shape
    zz = outputs.reshape(xx.shape)

    # Create a 3D surface plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(xx, yy, zz, cmap='viridis', alpha=0.8)

    # Add labels and title
    ax.set_xlabel('Feature 1 (Petal Length)')
    ax.set_ylabel('Feature 2 (Petal Width)')
    ax.set_zlabel('Network Output')
    ax.set_title('3D Surface Plot of Neural Network Output')

    # Show the plot
    plt.show()


def sampleOutput(data, weights, bias):
    data = data[(data['species'] == 'versicolor') | (data['species'] == 'virginica')]

    features = data[['petal_length', 'petal_width']].values
    outputs = oneLayerNN(features, weights, bias)

    data['output'] = outputs

    data['class'] = data['output'] > 0.5

    data['class'] = data['class'].map({True: 'virginica', False: 'versicolor'})

    unambiguous_examples = data[(outputs < 0.2) | (outputs > 0.8)]
    near_boundary_examples = data[(outputs >= 0.4) & (outputs <= 0.6)]

    # Display results
    print("Unambiguous Examples:")
    print(unambiguous_examples
          [['petal_length', 'petal_width', 'output', 'class', 'species']])

    print("\nNear-Boundary Examples:")
    print(near_boundary_examples
          [['petal_length', 'petal_width', 'output', 'class', 'species']])


def plotClasses(iris_data):
    # class_1 = iris_data[iris_data['species'] == 'setosa']
    class_2 = iris_data[iris_data['species'] == 'versicolor']
    class_3 = iris_data[iris_data['species'] == 'virginica']

    plt.figure(figsize=(10, 6))

    plt.scatter(class_2['petal_length'], class_2['petal_width'],
                label="Iris-versicolor", alpha=0.7)

    # plt.scatter(class_1['petal_length'], class_1['petal_width'],
    #            label="Iris-versicolor", alpha=0.7)

    plt.scatter(class_3['petal_length'], class_3['petal_width'],
                label="Iris-virginica", alpha=0.7)

    plt.xlabel('Petal Length')
    plt.ylabel('Petal Width')
    plt.title('Petal Dimensions of Iris Classes')
    plt.legend()
    plt.grid(True)

    plt.show()


def mse(data, weights, labels):
    data = data[(data['species'] == 'versicolor') | (data['species'] == 'virginica')]

    features = data[['petal_length', 'petal_width', 'bias']]
    outputs = oneLayerNN(features, weights)

    mse = np.mean((labels-outputs)**2)
    return mse


def gradient(data, weights, labels):
    # Filter and preprocess data
    data = data[(data['species'] == 'versicolor') | (data['species'] == 'virginica')]
    data = data[['petal_length', 'petal_width']]
    data['bias'] = 1  # Add a bias term
    labels = labels[(data.index)]

    # Convert to numpy arrays
    X = data.to_numpy()
    y = labels.to_numpy()

    # Compute z = X @ weights
    z = np.dot(X, weights)

    # Apply sigmoid function
    sigma_z = 1 / (1 + np.exp(-z))

    # Compute the gradient: (sigma_z - y) * X
    errors = sigma_z - y
    gradient = np.dot(errors, X)

    return gradient


def optimize(data, weights, labels, epsilon=0.01, maxIters=999, minDelta=1e-6):
    mse_history = []  # Store MSE for each iteration
    last_loss = float('inf')
    convergedIters = -1
    for i in range(maxIters):
        grad = gradient(data, weights, labels)
        weights -= epsilon * grad

        mse_value = mse(data, weights, labels)
        mse_history.append(mse_value)
        """
        if i % 500 == 0:
            print(i)
            decisionBoundaryPlot(data, weights)
        """
        if abs(mse_value - last_loss) < minDelta:
            print(f"Converged after {i} iterations")
            convergedIters = i
            break

        last_loss = mse_value
    print("Final Weights:", weights)
    print("Final MSE:", last_loss)
    return weights, mse_history, convergedIters


def optimize_tracked(data, weights, labels, epsilon=0.01, maxIters=999, minDelta=1e-6):
    mse_history = []  # Store MSE for each iteration
    last_loss = None
    convergedIters = -1
    initial_mse = None
    half_mse_reached = False
    half_mse_iteration = -1
    half_mse_weights = None

    for i in range(maxIters):
        grad = gradient(data, weights, labels)
        weights -= epsilon * grad

        mse_value = mse(data, weights, labels)
        mse_history.append(mse_value)

        if i == 0:
            initial_mse = mse_value
            last_loss = None  # Initialize properly

        if not half_mse_reached and mse_value < (initial_mse / 2):
            half_mse_reached = True
            half_mse_iteration = i
            half_mse_weights = weights.copy()

        if last_loss is not None and abs(mse_value - last_loss) < minDelta:
            print(f"Converged after {i} iterations")
            convergedIters = i
            break

        last_loss = mse_value  # Update after the stopping criterion check

    return weights, mse_history, half_mse_iteration, half_mse_weights, convergedIters

def main():
    weights = np.array([2, 4])
    bias = -11
    x_range = (0, 7)
    y_range = (0, 3)
    iris_data = pd.read_csv("irisdata.csv")
    data = iris_data[(iris_data['species'] == 'versicolor') | (iris_data['species'] == 'virginica')]
    labels = data['species'].map({'versicolor': 0, 'virginica': 1})
    # plotClasses(iris_data)
    # decisionBoundaryPlot(iris_data, weights, bias)
    # plot_surface(weights, bias, x_range, y_range)
    # sampleOutput(iris_data, weights, bias)
    # mse(iris_data, weights, bias, data['species'].map({'versicolor': 0, 'virginica': 1}))
    # mse(iris_data, np.array([-0.5, 2.5]), -5, data['species'].map({'versicolor': 0, 'virginica': 1}))
    # decisionBoundaryPlot(iris_data, weights, bias)
    # gradient(data, np.array([2, 4, -11]) , data['species'].map({'versicolor': 0, 'virginica': 1}))
    # decisionBoundaryPlot(iris_data, np.array([1.5, 3.8]), -11.2)
    # gradient(data, np.array([1.5, 3.8, -11.2]) , data['species'].map({'versicolor': 0, 'virginica': 1}))
    # data = data[['petal_length', 'petal_width']]
    # data['bias'] = 1
    # print(oneLayerNN(data, np.array([2,2,-13])))
    # w = optimize(data, np.array([1.5, 3.8, -11.2]), data['species'].map({'versicolor': 0, 'virginica': 1}), maxIters=2000)
    # decisionBoundaryPlot(data, w)
    # print(mse(data, w, labels))
    # final_weights, mse_history = optimize(data, np.array([1.5, 3.8, -11.2]),
    #                                       labels, maxIters=9000, epsilon=0.01)
    # decisionBoundaryPlot(data, final_weights)

    # Plot MSE over iterations
    """
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(mse_history)), mse_history, label="MSE")
    plt.xlabel("Iteration")
    plt.ylabel("Mean Squared Error")
    plt.title("MSE as a Function of Iterations")
    plt.grid(True)
    plt.legend()
    plt.show()
    """

    """
    np.random.seed(42)
    for _ in range(4):
        initial_weights = np.random.uniform(-0.25, 0.25, size=3)
        print(initial_weights)
        decisionBoundaryPlot(data, initial_weights)
        w, hmse, i = optimize(data, initial_weights, labels, maxIters=9999, epsilon=0.0001, minDelta=1e-5)
        print(f"Converged after {i} iterations, with final weights {w}")
        decisionBoundaryPlot(data, w)
    """

    iris_data = pd.read_csv("irisdata.csv")
    data = iris_data[(iris_data['species'] == 'versicolor') | (iris_data['species'] == 'virginica')]
    labels = data['species'].map({'versicolor': 0, 'virginica': 1})
    data['bias'] = 1

    # Initialize random weights and parameters
    np.random.seed(42)
    initial_weights = np.random.uniform(-1, 1, size=3)
    learning_rate = 0.001
    max_iters = 9000
    epsilon_loss = 1e-5

    print("Initial Weights:", initial_weights)

    # Plot initial decision boundary
    print("Initial Decision Boundary:")
    decisionBoundaryPlot(data, initial_weights)

    # Optimize weights and track MSE
    final_weights, mse_history, half_mse_iteration, half_mse_weights, convergedIters = optimize_tracked(
        data, initial_weights, labels, minDelta=epsilon_loss, maxIters=max_iters, epsilon=learning_rate
    )

    # Plot decision boundary when error is reduced by half
    if half_mse_iteration > -1:
        print(f"Decision Boundary after MSE reduced by half (Iteration {half_mse_iteration}):")
        decisionBoundaryPlot(data, half_mse_weights)

    # Plot final decision boundary
    print(f"Final Decision Boundary after Convergence (Iteration {convergedIters}):")
    decisionBoundaryPlot(data, final_weights)

    # Plot MSE over iterations
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(mse_history)), mse_history, label="MSE")
    plt.axvline(x=half_mse_iteration, color='orange', linestyle='--', label="MSE Reduced by Half")
    plt.xlabel("Iteration")
    plt.ylabel("Mean Squared Error")
    plt.title("MSE as a Function of Iterations")
    plt.legend()
    plt.grid(True)
    plt.show()




if __name__ == "__main__":
    main()
