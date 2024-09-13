import sys
import random
import re
import copy
import math
import queue


# Defining a class for a node to use for DFS and BFS
class Node:
    # Constructor
    def __init__(self, parent=None, direction=None, state=None, cost=0):
        # Using optional parameters because root node will not have parent or directoin but will have a state argument
        if parent and direction:
            # Trivial Constrcutor Tasks
            self.parent = parent
            self.direction = direction
            # Create current state of this node based on parents state and the direction of the move we are moving
            arr = move(direction, state=copy.deepcopy(parent.state))
            self.state = arr
            self.zero = findZero(self.state)
        # If a state is provided it will be the root node
        elif state and not parent and not direction:
            # Setting these fields for semantics
            self.parent = None
            self.direction = None
            self.state = state
            # Finding the zero in the given state
            self.zero = findZero(state=state)
        # Defining an array for each direction we move
        arr = ['left', 'right', 'up', 'down']
        # Creating the list to store children of node
        children = []
        for dir in arr:
            if checkMove(dir, self.zero[0], self.zero[1]):
                children.append(dir)
        # Storing valid directions for children
        self.children = children
        # Setting the cost function result
        self.cost = cost

    # Adding this string representation for debugging purposes
    def __str__(self):
        return (f'\n{self.state}\n')

    # Adding this function to allow comparison between nodes
    def __lt__(self, other):
        return self.cost < other.cost


# Define the goal state for the 8-puzzle as a 3x3 grid.
goal_state = [
    [0, 1, 2],
    [3, 4, 5],
    [6, 7, 8]
]

# Initialize the current state to be the same as the goal state.
current_state = [
    [0, 1, 2],
    [3, 4, 5],
    [6, 7, 8]
]

# Defining global lineNumber counter
lineNumber = 1


# Defining a global set for repeated state checking
used = {}


# Main method that runs when the script is executed.
def main():
    # Setting seed for rng
    random.seed(123)

    if len(sys.argv) != 1:
        # Check if a filename is provided in the command line arguments.
        fileName = sys.argv[1]

        # Open the provided file in read mode.
        file = open(fileName, "r")

        # Initialize line number for tracking the current line in the file.
        global lineNumber

        # Read and execute commands line by line from the file.
        for line in file:
            executeCommand(line.rstrip(), lineNumber)
            lineNumber += 1
            print(str(line))

        # Close the file after processing all lines.
        file.close()
    else:
        # If no file is provided, run the command-line interface (CLI).
        runCLI()


# Method to run the interactive command-line interface.
def runCLI():
    while True:
        # Prompt the user to enter a command.
        cmd = input("Enter a command:\n")
        # Execute the entered command.
        executeCommand(cmd, 'N/A')


# Method to execute a given command based on its input.
def executeCommand(cmd, line):
    # Split the input command into an array of tokens.
    arr = cmd.split(' ')
    cmd = arr[0]  # Extract the command.
    arr.pop(0)    # Remove the command from the array.

    # Execute the appropriate function based on the command.
    if cmd != "#" and cmd != "//" and cmd != "":  # Ignore comments and blanks.
        if cmd == "setState":
            setState(arr)
        elif cmd == "printState":
            printState()
        elif cmd == "move":
            move(arr[0])
        elif cmd == "scrambleState":
            scrambleState(arr[0])
        elif cmd == "solve" and arr[0] == "A*":
            if len(arr) == 3:
                maxnodes = int((re.search(r'\d+', arr[2])).group())
            else:
                maxnodes = -1
            astar(maxnodes, arr[1])
        elif cmd == "solve":
            if len(arr) == 2:
                maxnodes = int((re.search(r'\d+', arr[1])).group())
            else:
                maxnodes = -1
            if arr[0] == "DFS":
                dfs(maxnodes)
            elif arr[0] == "BFS":
                bfs(maxnodes)
            if arr[0] == "h1":
                print(h1())
            elif arr[0] == "h2":
                print(h2())
        else:
            # Print an error message for invalid commands.
            print(f"Error: invalid command at line {line}")


# Method to set the current state to a new state provided as input.
def setState(state):
    global current_state
    index = 0
    used = []

    # Ensure that exactly 9 numbers are provided to set the state.
    if len(state) == 9:
        for i in range(3):
            for j in range(3):
                num = int(state[index])
                if num not in used:
                    used.append(num)
                current_state[i][j] = num
                index += 1
        if len(used) != 9:
            print(f"Error: Invalid puzzle state: {lineNumber}")
            current_state = [[0, 1, 2], [4, 5, 6], [7, 8, 9]]
    else:
        print("Error: Invalid puzzle state")


# Method to print the current state of the 8-puzzle.
def printState():
    out = ""
    for x in current_state:
        out += "\n-------\n|"  # Formatting the grid with lines.
        for y in x:
            out += f"{y}|"

    out += "\n-------\n"
    print(f'{out}\n')


# Method to scramble current state by making n random moves from goal state.
def scrambleState(n):
    global current_state
    current_state = [
        [0, 1, 2],
        [3, 4, 5],
        [6, 7, 8]
    ]
    i, j = 0, 0  # Initial position of the empty space (zero).
    moves = ['up', 'down', 'left', 'right']

    # Make n random moves.
    for x in range(int(n)):
        dir = random.choice(moves)
        valid = checkMove(dir, i, j)
        while not valid:  # Find a valid move.
            dir = random.choice(moves)
            valid = checkMove(dir, i, j)
        if dir == 'up':
            i -= 1
        elif dir == 'down':
            i += 1
        elif dir == 'left':
            j -= 1
        elif dir == 'right':
            j += 1
        move(dir)


# Method to move the empty space (zero) in the specified direction.
def move(direction, state=None):
    if state is None:
        zero = findZero()  # Find the position of the empty space.
        i, j = zero[0], zero[1]

        # Check if the move is valid and execute it.
        if checkMove(direction, i, j):
            if direction == "up":
                current_state[i][j], current_state[i-1][j] = current_state[i-1][j], 0
            elif direction == "left":
                current_state[i][j], current_state[i][j-1] = current_state[i][j-1], 0
            elif direction == "down":
                current_state[i][j], current_state[i+1][j] = current_state[i+1][j], 0
            elif direction == "right":
                current_state[i][j], current_state[i][j+1] = current_state[i][j+1], 0
        else:
            # Print an error message if the move is invalid.
            print(f"Error: Invalid Move: {lineNumber}")
    else:
        zero = findZero(state)

        i, j = zero[0], zero[1]

        if direction == "up":
            state[i][j], state[i-1][j] = state[i-1][j], 0
        elif direction == "left":
            state[i][j], state[i][j-1] = state[i][j-1], 0
        elif direction == "down":
            state[i][j], state[i+1][j] = state[i+1][j], 0
        elif direction == "right":
            state[i][j], state[i][j+1] = state[i][j+1], 0

        return state


# Helper method to check if a move in a given direction is possible.
def checkMove(direction, i, j):
    # Prevent moves that would go out of bounds.
    if direction == "up" and i-1 >= 0:
        return True
    elif direction == "down" and i+1 <= 2:
        return True
    elif direction == "left" and j-1 >= 0:
        return True
    elif direction == "right" and j+1 <= 2:
        return True
    else:
        return False


# Helper method to find the position of the empty space (zero) in the current state.
def findZero(state=None):
    # Using an optional parameter to change which state we are editing
    if state is None:
        for i in range(3):
            for j in range(3):
                if current_state[i][j] == 0:
                    return [i, j]
    else:
        for i in range(3):
            for j in range(3):
                if state[i][j] == 0:
                    return [i, j]


# Method for DFS, iterative implementation
def dfs(maxnodes):
    # Checking if argument for maxnodes was provided in txt file
    if maxnodes == -1:
        maxnodes = 1000

    root = Node(state=current_state)

    # Defining counter to limit maxnodes
    counter = 1

    # Defining the queue for DFS
    stack = [root]

    # Defining an array with the solution steps as elements
    solution = None

    # Actual BFS search implementation
    while stack and counter < maxnodes:
        node = stack.pop()
        # If we've reached the goal the backtrack function will recursively find the solution steps
        if node.state == goal_state:
            solution = backtrack(node)
            break
        else:
            for child in node.children:
                if counter < maxnodes:
                    newChild = Node(parent=node, direction=child)
                    stack.append(newChild)
                    counter += 1
                else:
                    break
    # Check all remaining created but unvisited nodes
    # Doesn't count towards created nodes
    for node in stack:
        if node.state == goal_state:
            solution = backtrack(node)

    # Checking if a solution has been found
    if not solution:
        print(f"Error: maxnodes limit ({maxnodes}) reached: {lineNumber}")
    else:
        string = f"Nodes created during search: {counter}\nSolution length = {len(solution)-1}\nMove sequence:"
        for move in solution:
            if move:
                string += f"\nmove {move}\n"
        print(string)


# BFS implementation
def bfs(maxnodes):
    # Checking if argument was provided
    if maxnodes == -1:
        maxnodes = 1000

    root = Node(state=current_state)

    # Defining counter to limit maxnodes
    counter = 0

    # Defining the queue for BFS
    queue = [root]

    # Defining an array with the solution steps as elements
    solution = None

    # Actual BFS search implementation
    while queue and counter < maxnodes:
        node = queue.pop(0)
        if node.state == goal_state:
            solution = backtrack(node)
            break
        else:
            for child in node.children:
                if counter < maxnodes:
                    newChild = Node(parent=node, direction=child)
                    queue.append(newChild)
                    counter += 1
                else:
                    break
    # Iterating over all unvisitied, but previously created nodes to find solution if there
    for node in queue:
        if node.state == goal_state:
            solution = backtrack(node)

    # Checking if a solution was found
    if not solution:
        print(f"Error: maxnodes limit ({maxnodes}) reached")
    else:
        string = f"Nodes created during search: {counter}\nSolution length = {len(solution)-1}\nMove sequence:\n"
        for move in solution:
            if move:
                string += f"move {move}\n"
        print(string)


# Recursively retreats over the graph based on the direction of the nodes
def backtrack(node):
    solution = [node.direction]
    parent = node.parent
    while parent is not None:
        solution.append(parent.direction)
        parent = parent.parent
    # Reverses the order
    return solution[::-1]


# Function for h1, heuristic: number of displaced tiles from goal
def h1(state=current_state):
    # Store number of displaced tiles in counter
    counter = 0

    # For each of non-zero element check if its displacement from goal is 0
    for i in range(3):
        for j in range(3):
            if state[i][j] != 0 and displacement(state[i][j], [i, j]) > 0:
                counter += 1

    return counter


# Function for h2, heuristic: Manhattan
def h2(state=current_state):
    # Store sum of displaced tiles in counter
    sum = 0

    # For each of non-zero element check if its displacement from goal is 0
    for i in range(3):
        for j in range(3):
            distance = displacement(state[i][j], [i, j])
            if state[i][j] != 0 and distance > 0:
                sum += distance

    return sum


# Function that finds an elements distance from the its goal position
def displacement(element, coords):
    # Computing coordiantes of goal
    goal_i = math.floor(int(element)/3)
    goal_j = int(element) - (math.floor(int(element)/3)*3)

    # Computing displacement
    return abs(goal_i - coords[0]) + abs(goal_j - coords[1])


# A* implementation
def astar(maxnodes, heuristic):
    # Max nodes handling in case no arg provided
    if maxnodes == -1:
        maxnodes = 1000
    if heuristic == "h1":
        heuristic = h1
    if heuristic == "h2":
        heuristic = h2

    # Initializing Priority Queue
    q = queue.PriorityQueue()

    global used

    # Initializing the root
    root = Node(state=current_state, cost=0)

    # Insert the root
    q.put(root)

    # Using a counter to keep track of initialize nodes
    counter = 1

    # Initializing None solution var for checking if we have found one later
    solution = None

    # Main loop
    while not q.empty() and counter <= maxnodes:
        node = q.get()
        if node.state == goal_state:
            solution = backtrack(node)
            break
        else:
            for child in node.children:
                if counter < maxnodes:
                    nextState = move(child, state=copy.deepcopy(node.state))
                    if statecheck(nextState):
                        newChild = Node(parent=node, direction=child, state=nextState, cost=node.cost+1)
                        newChild.cost += heuristic(newChild.state)
                        q.put(newChild)
                        counter += 1
                else:
                    break
    if not solution:
        print(f"Error: maxnodes limit ({maxnodes}) reached")
    else:
        string = f"Nodes created during search: {counter}\nSolution length = {len(solution)-1}\nMove sequence:\n"
        for turn in solution:
            if turn:
                string += f"move {turn}\n"
        print(string)
    # Reset set since it persists otherwise (caused me a massive headache lol)
    used = {}


# Implementing repeated state checking
def statecheck(state):
    if used.get(str(state), False):
        return False
    else:
        used[str(state)] = True
        return True


# Standard Python convention to run the main method when the script is executed directly.
if __name__ == "__main__":
    main()
