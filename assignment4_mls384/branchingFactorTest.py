import unittest
from puzzle import branchingFactor  # Replace 'your_file_name' with the actual filename where your Newton's method is defined


class branchingFactorTest(unittest.TestCase):

    def test_binary_tree(self):
        true_b = 2
        d = 3
        N = sum([true_b**i for i in range(d + 1)]) - 1
        estimated_b = branchingFactor(N, d)
        print(f"Estimated b*: {estimated_b}, Known b*: {true_b}")
        self.assertAlmostEqual(estimated_b, true_b, places=6, msg="Failed to estimate b* for binary tree")

    def test_ternary_tree(self):
        true_b = 3
        d = 4
        N = sum([true_b**i for i in range(d + 1)]) - 1
        estimated_b = branchingFactor(N, d)
        print(f"Estimated b*: {estimated_b}, Known b*: {true_b}")
        self.assertAlmostEqual(estimated_b, true_b, places=6, msg="Failed to estimate b* for ternary tree")

    def test_quaternary_tree(self):
        true_b = 4
        d = 2
        N = sum([true_b**i for i in range(d + 1)]) - 1
        estimated_b = branchingFactor(N, d)
        print(f"Estimated b*: {estimated_b}, Known b*: {true_b}")
        self.assertAlmostEqual(estimated_b, true_b, places=6, msg="Failed to estimate b* for quaternary tree")

    def test_large_tree(self):
        true_b = 5
        d = 10
        N = sum([true_b**i for i in range(d + 1)]) - 1
        estimated_b = branchingFactor(N, d)
        print(f"Estimated b*: {estimated_b}, Known b*: {true_b}")
        self.assertAlmostEqual(estimated_b, true_b, places=6, msg="Failed to estimate b* for large tree")


if __name__ == '__main__':
    unittest.main()
