import sys
import random

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
        lineNumber = 1

        # Read and execute commands line by line from the file.
        for line in file:
            executeCommand(line.rstrip(), lineNumber)
            lineNumber += 1

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
            print("Error: Invalid puzzle state")
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
def move(direction):
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
        print("Error: Invalid Move")


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
def findZero():
    for i in range(3):
        for j in range(3):
            if current_state[i][j] == 0:
                return [i, j]


# Standard Python convention to run the main method when the script is executed directly.
if __name__ == "__main__":
    main()

