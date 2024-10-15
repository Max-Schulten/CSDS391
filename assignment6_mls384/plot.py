import numpy as np
import matplotlib.pyplot as plt


def main(bags, n, trueBag):

    # Initialize a dictionary to keep track of the respective calculated
    # probabilities at each observation
    probs = {key: [bags.get(key).get("prob")] for key in bags}

    # True bag type
    trueBag = bags.get(trueBag)

    # Keep track of observations
    observations = []

    # Number of observations (x-axis)
    x = np.arange(0, n)

    for i in x:
        # Take a candy from trueBag and add it to the list of observations
        data = take(trueBag)
        observations.append(data)

        for key, hypothesis in bags.items():
            # Calculate the probability of each hypothesis and add it to the
            # dictionary with its corresponding array
            probs.get(key).append(postCalc(prob=hypothesis.get("prob"),
                                           dist=hypothesis.get("dist"),
                                           observation=data,
                                           observations=observations,
                                           bags=bags))

    for key, probabilities in probs.items():
        plt.plot(np.arange(0, n+1), probabilities, label=rf'${key}$',
                 marker='x', linestyle='--')

    # Formatting the plot
    plt.xlabel('Number of observations in d')
    plt.ylabel('Posterior Probability')
    plt.ylim(0, 1)
    plt.xlim(0, n)
    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()

    # Show the plot
    plt.show()


# Finds p(h_i|d)
def postCalc(prob, dist, observation, observations, bags):
    # Calculating P(d|h_i)
    d_given_h = 1

    # Take product
    for j in observations:
        d_given_h *= dist[j]

    # Multiply by the probability of the hypothesis
    h_given_d = d_given_h * prob

    # Find proportionality constant
    alpha = 0
    for h in bags.values():
        likelihood = 1
        for j in observations:
            likelihood *= h["dist"][j]
        alpha += likelihood * h["prob"]

    return h_given_d / alpha


def take(bag):
    # Select a random index, representing a candy, with given probability
    indices = np.arange(len(bag.get("dist")))

    return np.random.choice(indices, p=bag.get("dist"))


if __name__ == "__main__":
    main(
        bags={
            "P(h_1|d)": {"dist": [1, 0.0], "prob": 0.1},
            "P(h_2|d)": {"dist": [0.75, 0.25], "prob": 0.2},
            "P(h_3|d)": {"dist": [0.50, 0.50], "prob": 0.4},
            "P(h_4|d)": {"dist": [0.25, 0.75], "prob": 0.2},
            "P(h_5|d)": {"dist": [0.0, 1], "prob": 0.1}
        },
        n=50,
        trueBag="P(h_2|d)"
    )
