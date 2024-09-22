import puzzle  # Assuming relevant methods are available in puzzle.py
import time  # To measure execution time

# Manually generated states progressively farther from the goal state
depths = [
    [[1, 0, 2], [3, 4, 5], [6, 7, 8]],  # Depth 2 (one move away)
    [[1, 4, 2], [3, 0, 5], [6, 7, 8]],  # Depth 3
    [[1, 4, 2], [0, 3, 5], [6, 7, 8]],  # Depth 4
    [[0, 4, 2], [1, 3, 5], [6, 7, 8]],  # Depth 5
    [[4, 0, 2], [1, 3, 5], [6, 7, 8]],  # Depth 6
    [[4, 2, 0], [1, 3, 5], [6, 7, 8]],  # Depth 7
    [[4, 2, 5], [1, 3, 0], [6, 7, 8]],  # Depth 8
    [[4, 2, 5], [1, 3, 8], [6, 7, 0]],  # Depth 9
    [[4, 2, 5], [1, 3, 8], [6, 0, 7]],  # Depth 10
    [[4, 2, 5], [1, 0, 8], [6, 3, 7]],  # Depth 11
    [[4, 0, 5], [1, 2, 8], [6, 3, 7]],  # Depth 12
    [[0, 4, 5], [1, 2, 8], [6, 3, 7]],  # Depth 13
    [[1, 4, 5], [0, 2, 8], [6, 3, 7]],  # Depth 14
    [[1, 4, 5], [6, 2, 8], [0, 3, 7]],  # Depth 15
    [[1, 4, 5], [6, 2, 8], [3, 0, 7]],  # Depth 16
]

def print_table():
    print(f"{'d':<5} {'BFS':<10} {'A*(h1)':<10} {'A*(h2)':<10} {'BFS BF':<10} {'A*(h1) BF':<10} {'A*(h2) BF':<10} {'BFS Time (s)':<12} {'A*(h1) Time (s)':<15} {'A*(h2) Time (s)':<15}")
    print("-" * 120)
    
    for d in depths:
        puzzle.current_state = d

        # Record execution time for BFS
        start_time = time.time()
        bfs_result = puzzle.bfs(999999, suppress=True)
        bfs_time = time.time() - start_time

        # Record execution time for A* with h1
        start_time = time.time()
        astar_h1_result = puzzle.astar(999999, heuristic='h1', suppress=True)
        astar_h1_time = time.time() - start_time

        # Record execution time for A* with h2
        start_time = time.time()
        astar_h2_result = puzzle.astar(999999, heuristic='h2', suppress=True)
        astar_h2_time = time.time() - start_time
        
        # Get the number of nodes generated and actual depth for each search method
        bfs_nodes = bfs_result["nodes"]
        astar_h1_nodes = astar_h1_result["nodes"]
        astar_h2_nodes = astar_h2_result["nodes"]

        bfs_depth = bfs_result["depth"]
        astar_h1_depth = astar_h1_result["depth"]
        astar_h2_depth = astar_h2_result["depth"]

        # Calculate branching factors using the `branchingFactor` function from puzzle.py
        bfs_bf = puzzle.branchingFactor(bfs_nodes, bfs_depth)
        astar_h1_bf = puzzle.branchingFactor(astar_h1_nodes, astar_h1_depth)
        astar_h2_bf = puzzle.branchingFactor(astar_h2_nodes, astar_h2_depth)
        
        # Print the row for current depth, including execution time
        print(f"{bfs_depth:<5} {bfs_nodes:<10} {astar_h1_nodes:<10} {astar_h2_nodes:<10} {bfs_bf:<10.2f} {astar_h1_bf:<10.2f} {astar_h2_bf:<10.2f} {bfs_time:<12.6f} {astar_h1_time:<15.6f} {astar_h2_time:<15.6f}")

if __name__ == "__main__":
    print_table()
