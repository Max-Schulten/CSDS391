import numpy as np
import matplotlib.pyplot as plt

def main():
    # Need to model the actual type of the bag
    bags = {
        "a": {"dist": [100, 0], "prob": 0.1},
        "b": {"dist": [75, 25], "prob": 0.2},
        "c": {"dist": [50, 50], "prob": 0.4},
        "d": {"dist": [25, 75], "prob": 0.2},
        "e": {"dist": [0, 100], "prob": 0.1}
    }

    # Number of observations (x-axis)
    x = np.arange(0, 11)

    # Plot
    p_h1_d = postCalc(bags.get("a").get("prob"), bags.get("a").get("dist"), x)

    plt.plot(x, p_h1_d, label=r'$P(h_1|d)$', marker='o', linestyle='-')

    # Formatting the plot
    plt.xlabel('Number of observations in d')
    plt.ylabel('Posterior Probability')
    plt.ylim(0, 1)
    plt.xlim(0, 10)
    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()

    # Show the plot
    plt.show()


def postCalc(prob, dist, x):
    return x/3

if __name__ == "__main__":
    main()
