import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def sigmoid(x):
    return 1/(1+np.exp(-x))


def oneLayerNN(data, weights, bias):
    return sigmoid(np.dot(data, weights) + bias)


def decisionBoundaryPlot(data, weights, bias):
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
    outputs = oneLayerNN(grid_points, weights, bias)

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


def mse(data, weights, bias, labels):
    data = data[(data['species'] == 'versicolor') | (data['species'] == 'virginica')]

    features = data[['petal_length', 'petal_width']]
    outputs = oneLayerNN(features, weights, bias)

    mse = np.mean((labels-outputs)**2)
    print(f'Weights: {weights}, Bias: {bias}, MSE: {mse}')
    return mse


def main():
    weights = np.array([1.0, 1.5])
    bias = -7.5
    x_range = (0, 7)
    y_range = (0, 3)
    iris_data = pd.read_csv("irisdata.csv")
    data = iris_data[(iris_data['species'] == 'versicolor') | (iris_data['species'] == 'virginica')]
    # plotClasses(iris_data)
    # decisionBoundaryPlot(iris_data, weights, bias)
    # plot_surface(weights, bias, x_range, y_range)
    # sampleOutput(iris_data, weights, bias)
    mse(iris_data, weights, bias, data['species'].map({'versicolor': 0, 'virginica': 1}))
    mse(iris_data, np.array([-0.5, 2.5]), -5, data['species'].map({'versicolor': 0, 'virginica': 1}))


if __name__ == "__main__":
    main()
