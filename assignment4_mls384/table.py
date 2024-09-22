import puzzle  # Assuming relevant methods are available in puzzle.py

# Manually generated states progressively farther from the goal state
depths = [
    [[0, 4, 2], [1, 3, 5], [6, 7, 8]],  # Depth 4
    [[4, 2, 0], [1, 3, 5], [6, 7, 8]],  # Depth 6
    [[4, 2, 5], [1, 3, 8], [6, 7, 0]],  # Depth 8
    [[4, 2, 5], [1, 0, 8], [6, 3, 7]],  # Depth 10
    [[0, 4, 5], [1, 2, 8], [6, 3, 7]],  # Depth 12
    [[1, 4, 5], [6, 2, 8], [0, 3, 7]],  # Depth 14
    [[1, 4, 5], [6, 2, 8], [3, 7, 0]],  # Depth 16
]


def print_table():
    print(f"{'d':<5} {'DFS':<10} {'BFS':<10} {'A*(h1)':<10} {'A*(h2)':<10} {'DFS BF':<10} {'BFS BF':<10} {'A*(h1) BF':<10} {'A*(h2) BF':<10}")
    print("-" * 95)

    for d in depths:
        puzzle.current_state = d

        dfs_result = puzzle.dfs(999999, suppress=True)

        # Record execution time for BFS
        bfs_result = puzzle.bfs(999999, suppress=True)

        astar_h1_result = puzzle.astar(999999, heuristic='h1', suppress=True)

        astar_h2_result = puzzle.astar(999999, heuristic='h2', suppress=True)

        # Get the number of nodes generated and actual depth for each search method
        dfs_nodes = dfs_result["nodes"]
        bfs_nodes = bfs_result["nodes"]
        astar_h1_nodes = astar_h1_result["nodes"]
        astar_h2_nodes = astar_h2_result["nodes"]

        dfs_depth = dfs_result["depth"]
        bfs_depth = bfs_result["depth"]
        astar_h1_depth = astar_h1_result["depth"]
        astar_h2_depth = astar_h2_result["depth"]

        # Calculate branching factors using the `branchingFactor` function from puzzle.py
        dfs_bf = puzzle.branchingFactor(dfs_nodes, dfs_depth)
        bfs_bf = puzzle.branchingFactor(bfs_nodes, bfs_depth)
        astar_h1_bf = puzzle.branchingFactor(astar_h1_nodes, astar_h1_depth)
        astar_h2_bf = puzzle.branchingFactor(astar_h2_nodes, astar_h2_depth)

        # Print the row for current depth, including execution time
        print(f"{bfs_depth:<5} {dfs_nodes:<10} {bfs_nodes:<10} {astar_h1_nodes:<10} {astar_h2_nodes:<10} {dfs_bf:<10.2f} {bfs_bf:<10.2f} {astar_h1_bf:<10.2f} {astar_h2_bf:<10.2f}")


if __name__ == "__main__":
    print_table()
    print("-" * 95)
