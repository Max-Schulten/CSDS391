import sys
import random

# Setting goal state to compare
goal_state = [
    [0, 1, 2],
    [3, 4, 5],
    [6, 7, 8]
            ]

# 2D array that will hold the current state
current_state = [
    [0, 1, 2],
    [3, 4, 5],
    [6, 7, 8]
            ]


# Main method that will run when puzzle.py is called
def main():
    if (len(sys.argv) != 1):
        # Receives filename from CLI
        fileName = sys.argv[1]

        # Opens file to read
        file = open(fileName, "r")

        # Keeps track of what line we're on
        lineNumber = 1

        # Execute commands line by line
        for line in file:
            executeCommand(line.rstrip(), lineNumber)
            lineNumber += 1
    else:
        runCLI()


def runCLI():
    while True:
        cmd = input("Enter a command:\n")
        executeCommand(cmd, 'N/A')


# Method that executes some command
def executeCommand(cmd, line):

    # Splitting string into array to tokenize
    arr = cmd.split(' ')
    cmd = arr[0]
    arr.pop(0)

    # Bunch of if blocks to check what command is called
    if cmd != "#" and cmd != "//":
        if cmd == "setState":
            setState(arr)
        elif cmd == "printState":
            printState()
        elif cmd == "move":
            move(arr[0])
        elif cmd == "scrambleState":
            scrambleState(arr[0])
        else:
            print(f"Error: invalid command: {line}")


# Method to set current State
def setState(state):
    index = 0
    used = []
    if (len(state) == 9):
        for i in range(3):
            for j in range(3):
                num = int(state[index])
                if num in used:
                    break
                current_state[i][j] = num
                index += 1
            else:
                continue
            break


# Method that prints the current state
def printState():
    out = ""
    for x in current_state:
        out += "\n-------\n|"
        for y in x:
            out += f"{y}|"

    out += "\n-------\n"
    print(f'{out}\n')


# Method that scrambles the state by making n random moves from goal state
def scrambleState(n):
    global current_state
    current_state = goal_state
    i = 0
    j = 0
    moves = ['up', 'down', 'left', 'right']
    for x in range(int(n)):
        dir = random.choice(moves)
        valid = checkMove(dir, i, j)
        while not valid:
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


# Method that makes some move
def move(direction):
    zero = findZero()
    i = zero[0]
    j = zero[1]
    if checkMove(direction, i, j):
        if direction == "up":
            current_state[i][j] = current_state[i-1][j]
            current_state[i-1][j] = 0
        elif direction == "left":
            current_state[i][j] = current_state[i][j-1]
            current_state[i][j-1] = 0
        elif direction == "down":
            current_state[i][j] = current_state[i+1][j]
            current_state[i+1][j] = 0
        elif direction == "right":
            current_state[i][j] = current_state[i][j+1]
            current_state[i][j+1] = 0
    else:
        print("Error: Invalid Move")


# Helper method that checks if a move is possible
def checkMove(direction, i, j):

    # Essentially just prevents an index out of bounds
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


# Helper method that locates the index of the zero a.k.a. the empty table
def findZero():
    for i in range(3):
        for j in range(3):
            if current_state[i][j] == 0:
                return [i, j]


# Semantic python best practices for main method call
if __name__ == "__main__":
    main()
