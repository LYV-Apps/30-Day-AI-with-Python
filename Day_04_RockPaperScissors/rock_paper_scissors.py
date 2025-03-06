import numpy as np 
import random
import time
from collections import deque
from colorama import init, Fore, Style
from sklearn.neighbors import KNeighborsClassifier

def slow_print(text, delay=0.1):
    print(text)
    time.sleep(delay)


# Global variables to track game state
move_history = deque(maxlen=3) # Stores last 2 moves plus the current
X = []
Y = []
model = KNeighborsClassifier(n_neighbors=1) # k-NN with 1 neighbor
user_score = 0
ai_score = 0
ties = 0
moves = { "rock": 0, "paper": 1, "scissors": 2 }
reverse_moves = { 0: "rock", 1: "paper", 2: "scissors" }

def get_user_move():
    while True:
        choice = input(f"\n{Fore.BLUE}Enter your move (rock, paper, or scissors): {Style.RESET_ALL}").lower()
        if choice in moves:
            move_history.append(moves[choice])
            return choice
        slow_print(f"{Fore.RED}Invalid move. Please enter rock, paper, or scissors.{Style.RESET_ALL}")

def train_model():
    if len(move_history) < 3:
        return
    
    sequence = list(move_history)[:-1]
    next_move = move_history[-1]

    X.append(sequence)
    Y.append(next_move)
    if len(X) >= 2:
        model.fit(np.array(X), np.array(Y))

def predict_user_move():
    if len(move_history) < 2 or len(X) < 2: # Not enough data to make a prediction
        return random.choice(list(moves.keys()))
    
    # Predict based on last 2 moves
    recent_moves = list(move_history)[:2]
    prediction = model.predict([recent_moves])[0]
    predicted_move = reverse_moves[prediction]

    # Counter the predicted move
    if predicted_move == "rock":
        return "paper"
    elif predicted_move == "paper":
        return "scissors"
    else:
        return "rock"
    
def determine_winner(user, computer):
    global user_score, ai_score, ties
    if user == computer:
        ties += 1
        return "Tie"
    elif (user == "rock" and computer == "scissors") or \
         (user == "paper" and computer == "rock") or \
         (user == "scissors" and computer == "paper"):
        user_score += 1
        return "You win!"
    else:
        ai_score += 1
        return "Computer wins!"
    
if __name__ == "__main__":
    rounds = 0
    slow_print(f"{Fore.CYAN}{'*' * 27}{Style.RESET_ALL}")
    slow_print(f"{Fore.GREEN}*    🤖  30 DAY AI  🤖    *{Style.RESET_ALL}")
    slow_print(f"{Fore.YELLOW}*    WELCOME TO DAY 4!    *{Style.RESET_ALL}")
    slow_print(f"{Fore.RED}* AI ROCK, PAPER, SCISSORS *{Style.RESET_ALL}")
    slow_print(f"{Fore.BLUE}*          🪨 📑 ✄          *{Style.RESET_ALL}")
    slow_print(f"{Fore.CYAN}{'*' * 27}{Style.RESET_ALL}")
    while rounds < 10:
        user_choice = get_user_move()
        train_model()
        computer_choice = predict_user_move()
        result = determine_winner(user_choice, computer_choice)

        slow_print(f"\n{Fore.YELLOW}You chose: {user_choice}{Style.RESET_ALL}")
        slow_print(f"{Fore.MAGENTA}Computer chose: {computer_choice}{Style.RESET_ALL}")
        slow_print(f"\n{Fore.CYAN}Result: {result}{Style.RESET_ALL}")
        rounds += 1

    slow_print(f"\n{Fore.CYAN}{'*' * 27}{Style.RESET_ALL}")
    slow_print(f"\nFinal Score = You: {user_score}, Computer: {ai_score}, Ties: {ties}")
    slow_print(f"\n{Fore.CYAN}{'*' * 27}{Style.RESET_ALL}")
