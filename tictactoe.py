#assgn 8
import random

# a. Setting up environment
def new_board():
    return [' '] * 9

# b. Defining Tic-Tac-Toe game
def winner(b):
    wins = [(0,1,2),(3,4,5),(6,7,8),
            (0,3,6),(1,4,7),(2,5,8),
            (0,4,8),(2,4,6)]
    for a,c,d in wins:
        if b[a] == b[c] == b[d] != ' ':
            return b[a]
    return None

def display(b):
    for i in range(0,9,3):
        print('|'.join(b[i:i+3]))
        if i < 6:
            print('-----')

# c. Q-Learning model
Q = {}
lr = 0.1
gamma = 0.9

def choose_action(b, epsilon):
    moves = [i for i in range(9) if b[i] == ' ']
    state = tuple(b)

    if random.random() < epsilon:
        return random.choice(moves)

    return max(moves, key=lambda a: Q.get((state,a), 0))

def update(state, action, reward, next_state, done):
    old = Q.get((state,action), 0)
    future = 0 if done else max(
        [Q.get((next_state,a),0)
         for a in range(9) if next_state[a] == ' '] or [0]
    )
    Q[(state,action)] = old + lr * (
        reward + gamma * future - old
    )

# d. Training the model
for episode in range(5000):
    b = new_board()

    while True:
        # Random opponent plays X
        x_moves = [i for i in range(9) if b[i] == ' ']
        b[random.choice(x_moves)] = 'X'

        if winner(b) == 'X':
            break

        if ' ' not in b:
            break

        state = tuple(b)
        action = choose_action(b, 0.2)
        b[action] = 'O'

        if winner(b) == 'O':
            update(state, action, 1, tuple(b), True)
            break

        if ' ' not in b:
            update(state, action, 0, tuple(b), True)
            break

        # Opponent's next move
        x_moves = [i for i in range(9) if b[i] == ' ']
        b[random.choice(x_moves)] = 'X'

        if winner(b) == 'X':
            update(state, action, -1, tuple(b), True)
            break

        update(state, action, 0, tuple(b), False)

print("Training completed!")

# e. Testing the trained model
b = new_board()

while True:
    display(b)

    move = int(input("Enter position (0-8): "))
    if move not in range(9) or b[move] != ' ':
        print("Invalid move!")
        continue

    b[move] = 'X'

    if winner(b) == 'X':
        display(b)
        print("You win!")
        break

    if ' ' not in b:
        display(b)
        print("Draw!")
        break

    action = choose_action(b, 0)
    b[action] = 'O'
    print("Agent chose:", action)

    if winner(b) == 'O':
        display(b)
        print("Agent wins!")
        break

    if ' ' not in b:
        display(b)
        print("Draw!")
        break
