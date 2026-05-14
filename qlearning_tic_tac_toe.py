from typing import List, Tuple
import random
import matplotlib.pyplot as plt
import numpy as np
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import os
from vae.model import VAE
import torch.nn as nn
from sklearn.model_selection import train_test_split
import copy
from sklearn.decomposition import PCA
Board = List[int]
Turn = int

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
PLOT_PATH = "./plots"



class TicTacToe_N_K:
    def __init__(self, n: int, k: int):
        self.n = n
        self.k = k

    def reset(self) -> Board:
        return [0] * (self.n * self.n)
    
    def encode_state(self, board: Board, turn: Turn) -> Tuple[int, ...]:
        return tuple(board + [turn])
    
    def get_legal_actions(self, board:Board) -> List[int]:
        return [i for i , v in enumerate(board) if v == 0]
    
    def step(self, board: Board, turn: Turn, action: int) -> Tuple[Board, Turn, float, bool]:
        if board[action] != 0:
            raise ValueError("Invalid action: Cell is already occupied.")
        next_board = board.copy()
        next_board[action] = turn

        winner = self.check_winner(next_board)
        if winner == turn:
            return next_board, -turn, 1.0, True 
        
        if all(cell != 0 for cell in next_board):
            return next_board, -turn, 0.0, True
        
        return next_board, -turn, 0.0, False
    
    def check_winner(self, board: Board) -> int:
        for r in range(self.n):
            for c in range(self.n - self.k + 1):
                first = board[r * self.n + c]
                if first == 0:
                    continue
                if all(board[r * self.n + c + i] == first for i in range(self.k)):
                    return first
        
        for c in range(self.n):
            for r in range(self.n - self.k + 1):
                first = board[r * self.n + c]
                if first == 0:
                    continue
                if all(board[(r + i) * self.n + c] == first for i in range(self.k)):
                    return first 
        for r in range(self.n - self.k + 1):
            for c in range(self.n - self.k + 1):
                first = board[r * self.n + c]
                if first == 0:
                    continue
                if all(board[(r + t) * self.n + (c + t)] == first for t in range(self.k)):
                    return first

        for r in range(self.n - self.k + 1):
            for c in range(self.k - 1, self.n):
                first = board[r*self.n + c]
                if first == 0:
                    continue
                if all(board[(r + t)*self.n + (c - t)] == first for t in range(self.k)):
                    return first
        
        return 0
    
    def print_board(self, board: Board):
        symbols = {0: '-', 1: 'X', -1: 'O'}
        for i in range(self.n):
            row = [symbols[board[i * self.n + j]] for j in range(self.n)]
            print(" ".join(row))


def update_Q_zero(Q : dict, state: Tuple[int,...], action: int, reward: float, next_state: Tuple[int, ...], legal_moves: List[int], legal_moves_next: List[int], alpha: float, gamma:float):
    if state not in Q:
        Q[state] = {a: 0.0 for a in legal_moves}
    
    if legal_moves_next:
        if next_state not in Q:
            Q[next_state] = {a: 0.0 for a in legal_moves_next}
        max_next_Q = max(Q[next_state].values())
    else:
        max_next_Q = 0.0

    Q[state][action] += alpha * (reward + gamma * max_next_Q - Q[state][action])

    return Q
    
def update_Q_random(Q : dict, state: Tuple[int,...], action: int, reward: float, next_state: Tuple[int, ...], legal_moves: List[int], legal_moves_next: List[int], alpha: float, gamma:float):
    if state not in Q:
        Q[state] = {a: random.uniform(-0.01,0.01) for a in legal_moves}
    
    if legal_moves_next:
        if next_state not in Q:
            Q[next_state] = {a: random.uniform(-0.01,0.01) for a in legal_moves_next}
        max_next_Q = max(Q[next_state].values())
    else:
        max_next_Q = 0.0
    
    
    Q[state][action] += alpha * (reward + gamma * max_next_Q - Q[state][action])

    return Q

def update_Q_minimax(
    Q: dict,
    state: Tuple[int, ...],
    action: int,
    reward: float,
    next_state: Tuple[int, ...] | None,
    legal_moves: List[int],
    legal_moves_next: List[int],
    alpha: float,
    gamma: float
):
    """
    Minimax-style Q update for 2-player zero-sum alternating-turn games.

    Interpretation:
      Q(s,a) = value for the player whose turn it is in state s.

    Key difference vs standard Q-learning:
      next_state belongs to the OPPONENT, so their best value should be subtracted:
        target = r - gamma * max_a' Q(next_state, a')
    """
    if state not in Q:
        Q[state] = {a: 0.0 for a in legal_moves}

    # Terminal next_state: no future value
    if not legal_moves_next or next_state is None:
        max_next_Q = 0.0
    else:
        if next_state not in Q:
            Q[next_state] = {a: 0.0 for a in legal_moves_next}
        max_next_Q = max(Q[next_state].values())

    target = reward - gamma * max_next_Q  # <-- adversarial backup (minus sign!)
    Q[state][action] += alpha * (target - Q[state][action])
    return Q

def e_greedy_selection(Q: dict, state: Tuple[int,...], legal_moves: List[int], epsilon: float) -> int:
    # exploration
    if random.random() < epsilon:
        return random.choice(legal_moves)

    # exploitation
    state_Q = Q.get(state, {})
    max_Q = max(state_Q.get(a, 0.0) for a in legal_moves)
    best_actions = [a for a in legal_moves if state_Q.get(a, 0.0) == max_Q]
    return random.choice(best_actions)


def scheduled_epsilon(t: int, epsilon_start: float, epsilon_end: float, decay_steps: int) -> float:
    if t >= decay_steps:
        return epsilon_end
    return epsilon_start - (epsilon_start - epsilon_end) * (t / decay_steps)

# OPTION A – SELF-PLAY (shared Q-table for both players)
# ------------------------------------------------------
# - The same Q-table is used for both X and O.
# - The state always includes BOTH the board and the current turn:
#       state = encode_state(board, turn)
# - The agent chooses actions for whichever player’s turn it is.
# - This allows symmetric learning and produces a general policy
#   independent of playing first or second.
#
# OPTION B – FIXED OPPONENT (random or minimax)
# ---------------------------------------------
# - The Q-agent always plays as X, while O is external
#   (random policy or minimax opponent).
# - The state still includes turn for consistency with the project
#   requirements, but only states with turn = +1 (X to act) are
#   stored and updated in the Q-table.
# - This simplifies learning because only the agent’s decisions
#   are learned, while the opponent acts as part of the environment.
# - step function returns reward for the player who just moved, so when O (opponent) moves and wins, the reward for X is -1, which we need to account for in the Q update to ensure X learns from losses as well as wins.
# - In this design, reward is not equal to zero only in terminal states.
# --------------------------------*-----------------------*-----------------------*-----------------------*-----------------------*-----------------------*
def play_game_q_agent(game: TicTacToe_N_K, Q: dict, epsilon: float, games: int) -> None:
    board = game.reset()
    done = False
    count_X_wins = 0
    count_draws = 0
    count_O_wins = 0
    loss_rates = []
    episodes = []
    for i in range(games):
        board= game.reset()
        done = False
        start_turn = random.choice([1, -1])

        if start_turn == -1:
            states_visited.add(game.encode_state(board, -1)) # Track the initial state
            legal_O = game.get_legal_actions(board)
            action_O = random.choice(legal_O)
            board, _, _, done = game.step(board, -1, action_O)
 
        epsilon = scheduled_epsilon(i, epsilon_start=1.0, epsilon_end=0.1, decay_steps=games//2)

        while not done:
            state_X = game.encode_state(board, 1)
            legal_actions_X = game.get_legal_actions(board)
            action_X = e_greedy_selection(Q, state_X, legal_actions_X, epsilon)
            board_after_X, _, reward_X, done = game.step(board, 1, action_X)

            states_visited.add(state_X)  # Track unique states visited during training
            
            if done:
                states_visited.add(game.encode_state(board_after_X, -1))  # Track the final state where X wins
                legal_moves_next = []
                next_state = None
                update_Q_zero(Q, state_X, action_X, reward_X, next_state, legal_actions_X, legal_moves_next, alpha=0.1, gamma=0.9)
                winner = game.check_winner(board_after_X)
                if winner == 1.0:
                    count_X_wins += 1
                elif winner == -1:
                    count_O_wins += 1
                else:
                    count_draws += 1
                
                break
            
            states_visited.add(game.encode_state(board_after_X, -1)) # Track the state after X's move as well, even though it's O's turn, for analysis of visited states
            legal_O = game.get_legal_actions(board_after_X)
            action_O = random.choice(legal_O)
            board_after_O, _, reward_O, done = game.step(board_after_X, -1, action_O)
            if done:
                states_visited.add(game.encode_state(board_after_O, 1))  # Track the final state where O wins
            if done and reward_O == 1.0:
                # step function returns reward for the player who moves. Since O just moved and won, reward_O is +1 for O, which means it's -1 for X. 
                # We add manually the reward for X after O's winning move to ensure the Q update for X reflects the loss.
                r_for_X = -1.0  
                legal_next_for_X = []
                count_O_wins += 1
            elif done:
                r_for_X = 0.0
                legal_next_for_X = []
                count_draws += 1
            else:
                    r_for_X = 0.0
                    legal_next_for_X = game.get_legal_actions(board_after_O)
            
            next_state_X = game.encode_state(board_after_O, 1) if legal_next_for_X else None
            
            update_Q_zero(Q, state_X, action_X, r_for_X, next_state_X, legal_actions_X, legal_next_for_X, alpha=0.1, gamma=0.9)

            
            board = board_after_O

        if (i+1) % 1000 == 0 and i > 0:
            loss_rates.append(count_O_wins / (count_X_wins + count_O_wins + count_draws) if (count_X_wins + count_O_wins + count_draws) > 0 else 0)
            episodes.append(i+1)
            count_O_wins = 0
            count_X_wins = 0
            count_draws = 0
        

    
    plt.figure()
    plt.plot(episodes, loss_rates, label='Loss Rate (O wins)')
    plt.xlabel('Episodes')
    plt.ylabel('Loss Rate')
    plt.title('Loss Rate of Q-Learning Agent Over Time')
    plt.savefig(PLOT_PATH + '/q_learning_random_loss_rate.png')
    plt.close()

def evaluate_q_agent(game: TicTacToe_N_K, Q: dict, games: int) -> None:
    count_X_wins = 0
    count_draws = 0
    count_O_wins = 0

    for _ in range(games):
        board = game.reset()
        done = False
        start_turn = random.choice([1, -1])

        if start_turn == -1:
            legal_O = game.get_legal_actions(board)
            action_O = random.choice(legal_O)
            board, _, _, done = game.step(board, -1, action_O)
            if done:
                winner = game.check_winner(board)
                if winner == 1.0:
                    count_X_wins += 1
                elif winner == -1:
                    count_O_wins += 1
                else:
                    count_draws += 1
                continue
            
        while not done:
            state_X = game.encode_state(board, 1)
            legal_actions_X = game.get_legal_actions(board)
            action_X = e_greedy_selection(Q, state_X, legal_actions_X, epsilon=0.0)  # No exploration during evaluation
            board_after_X, _, reward_X, done = game.step(board, 1, action_X)

            if done:
                winner = game.check_winner(board_after_X)
                if winner == 1.0:
                    count_X_wins += 1
                elif winner == -1:
                    count_O_wins += 1
                else:
                    count_draws += 1
                break

            legal_O = game.get_legal_actions(board_after_X)
            action_O = random.choice(legal_O)
            board_after_O, _, reward_O, done = game.step(board_after_X, -1, action_O)

            if done:
                winner = game.check_winner(board_after_O)
                if winner == 1:
                    count_X_wins += 1
                elif winner == -1:
                    count_O_wins += 1
                else:
                    count_draws += 1
                break
            
            board = board_after_O

    print(f"Evaluation after {games} episodes: X wins: {count_X_wins}, O wins: {count_O_wins}, Draws: {count_draws}")
# --------------------------------*-----------------------*-----------------------*-----------------------*-----------------------*-----------------------*

def terminal_score(game, board):
    winner = game.check_winner(board)
    if winner == -1:
        return 1.0
    if winner == 1:
        return -1.0
    if not game.get_legal_actions(board):
        return 0.0
    return None  # Game is not terminal

def alphabeta(game, board, turn, alpha, beta):
    score = terminal_score(game, board)
    if score is not None:
        return score
    
    legal = game.get_legal_actions(board)
    if turn == -1:
        value = -float('inf')
        for action in legal:
            nb = board.copy()
            nb[action] = turn
            value = max(value, alphabeta(game, nb, 1, alpha, beta))
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return value
    
    else:
        value = float('inf')
        for action in legal:
            nb = board.copy()
            nb[action] = turn
            value = min(value, alphabeta(game, nb, -1, alpha, beta))
            beta = min(beta, value)
            if alpha >= beta:
                break
        return value

def minimax_move_alphabeta(game, board, turn=-1):
    legal = game.get_legal_actions(board)
    best_score = float('-inf')
    best_actions = []
    alpha, beta = -float('inf'), float('inf')

    for action in legal:
        nb = board.copy()
        nb[action] = turn   #player O plays
        score = alphabeta(game, nb, -turn, alpha, beta)
        if score > best_score:
            best_score = score
            best_actions = [action]
            alpha = max(alpha, best_score)
        elif score == best_score:
            best_actions.append(action)

        alpha = max(alpha, best_score)

    return random.choice(best_actions)


def play_game_q_minimax_opponent(game: TicTacToe_N_K, Q: dict, epsilon: float, games: int) -> None:
    board = game.reset()
    done = False
    count_X_wins = 0
    count_draws = 0
    count_O_wins = 0
    loss_rates = []
    episodes = []
    for i in range(games):
        board= game.reset()
        done = False
        start_turn = random.choice([1, -1])

        if start_turn == -1:
            states_visited.add(game.encode_state(board, -1)) # Track the initial state
            action_O = minimax_move_alphabeta(game, board, turn=-1)
            board, _, _, done = game.step(board, -1, action_O)
            
    
        epsilon = scheduled_epsilon(i, epsilon_start=1.0, epsilon_end=0.1, decay_steps=games//2)

        while not done:
            state_X = game.encode_state(board, 1)
            states_visited.add(state_X)  
            legal_actions_X = game.get_legal_actions(board)
            action_X = e_greedy_selection(Q, state_X, legal_actions_X, epsilon)
            board_after_X, _, reward_X, done = game.step(board, 1, action_X)

            if done:
                states_visited.add(game.encode_state(board_after_X, -1))  # Track the final state where X wins
                legal_moves_next = []
                next_state = None
                update_Q_zero(Q, state_X, action_X, reward_X, next_state, legal_actions_X, legal_moves_next, alpha=0.1, gamma=0.9)
                winner = game.check_winner(board_after_X)
                if winner == 1.0:
                    count_X_wins += 1
                elif winner == -1:
                    count_O_wins += 1
                else:
                    count_draws += 1
                
                break
            
            states_visited.add(game.encode_state(board_after_X, -1)) 
            action_O = minimax_move_alphabeta(game, board_after_X, turn=-1)
            board_after_O, _, reward_O, done = game.step(board_after_X, -1, action_O)

            if done:
                states_visited.add(game.encode_state(board_after_O, 1))  # Track the final state where O wins
            if done and reward_O == 1.0:
                #r_for_X is the reward for X after O's move, which is -1 if O wins, 0 if draw, and 0 if game continues
                r_for_X = -1.0  
                legal_next_for_X = []
                count_O_wins += 1
            elif done:
                r_for_X = 0.0
                legal_next_for_X = []
                count_draws += 1
            else:
                    r_for_X = 0.0
                    legal_next_for_X = game.get_legal_actions(board_after_O)
            
            next_state_X = game.encode_state(board_after_O, 1) if legal_next_for_X else None
            update_Q_zero(Q, state_X, action_X, r_for_X, next_state_X, legal_actions_X, legal_next_for_X, alpha=0.1, gamma=0.9)

            
            board = board_after_O
        if (i+1) % 1000 == 0 and i > 0:
            loss_rates.append(count_O_wins / (count_X_wins + count_O_wins + count_draws) if (count_X_wins + count_O_wins + count_draws) > 0 else 0)
            episodes.append(i+1)
            count_O_wins = 0
            count_X_wins = 0
            count_draws = 0
        

    print(f"After {games} games: X wins: {count_X_wins}, O wins: {count_O_wins}, Draws: {count_draws}")
    plt.figure()
    plt.plot(episodes, loss_rates, label='Loss Rate (O wins)')
    plt.xlabel('Episodes')
    plt.ylabel('Loss Rate')
    plt.title('Loss Rate of Q-Learning Agent Over Time')
    plt.savefig(PLOT_PATH + '/q_learning_minimax_loss_rate.png')
    plt.close()



def evaluate_q_agent_minimax_opponent(game: TicTacToe_N_K, Q: dict, games: int) -> None:
    count_X_wins = 0
    count_draws = 0
    count_O_wins = 0

    for _ in range(games):
        board = game.reset()
        done = False
        start_turn = random.choice([1, -1])

        if start_turn == -1:
            action_O = minimax_move_alphabeta(game, board, turn=-1)
            board, _, _, done = game.step(board, -1, action_O)
            if done:
                winner = game.check_winner(board)
                if winner == 1.0:
                    count_X_wins += 1
                elif winner == -1:
                    count_O_wins += 1
                else:
                    count_draws += 1
                continue
            
        while not done:
            state_X = game.encode_state(board, 1)
            legal_actions_X = game.get_legal_actions(board)
            action_X = e_greedy_selection(Q, state_X, legal_actions_X, epsilon=0.0)  # No exploration during evaluation
            board_after_X, _, _, done = game.step(board, 1, action_X)

            if done:
                winner = game.check_winner(board_after_X)
                if winner == 1.0:
                    count_X_wins += 1
                elif winner == -1:
                    count_O_wins += 1
                else:
                    count_draws += 1
                break

            action_O = minimax_move_alphabeta(game, board_after_X, turn=-1)
            board_after_O, _, _, done = game.step(board_after_X, -1, action_O)


            if done:
                winner = game.check_winner(board_after_O)
                if winner == 1:
                    count_X_wins += 1
                elif winner == -1:
                    count_O_wins += 1
                else:
                    count_draws += 1
                break
            
            board = board_after_O

    print(f"Evaluation after {games} episodes: X wins: {count_X_wins}, O wins: {count_O_wins}, Draws: {count_draws}")


# --------------------------------*-----------------------*-----------------------*-----------------------*-----------------------*-----------------------*

def play_game_q_self_play(game: TicTacToe_N_K, Q: dict, games: int) -> None:
    count_X_wins = 0
    count_O_wins = 0
    count_draws = 0

    draw_rates = []
    episodes = []

    for i in range(games):
        board = game.reset()
        done = False
        turn = random.choice([1, -1])  # random start player

        epsilon_i = scheduled_epsilon(i, epsilon_start=1.0, epsilon_end=0.1, decay_steps=games // 2)

        # Track the previous move so we can assign -1 to the loser when the opponent wins
        prev_state = None
        prev_action = None
        prev_legal = None
        prev_turn = None  # whose move created prev_state/action

        while not done:
            state = game.encode_state(board, turn)
            states_visited.add(state)
            legal = game.get_legal_actions(board)

            action = e_greedy_selection(Q, state, legal, epsilon_i)
            next_board, _, reward, done = game.step(board, turn, action)

            if done:
                states_visited.add(game.encode_state(next_board, -turn))  # Track the final state as well
                # Update winner's last move normally (reward from environment is +1 for winner, 0 for draw)
                update_Q_minimax(Q, state, action, reward, None, legal, [], alpha=0.1, gamma=0.9)

                winner = game.check_winner(next_board)

                
                if winner != 0 and prev_state is not None and prev_action is not None:
                    # The player who made the previous move is the loser (opponent just won)
                    # Terminal update: no next_state, so legal_moves_next=[]
                    update_Q_minimax(Q, prev_state, prev_action, reward=-1.0, next_state=None,
                                     legal_moves=prev_legal, legal_moves_next=[],
                                     alpha=0.1, gamma=0.9)

                if winner == 1:
                    count_X_wins += 1
                elif winner == -1:
                    count_O_wins += 1
                else:
                    count_draws += 1
                break

            # non-terminal: next player to act is -turn
            next_turn = -turn
            next_state = game.encode_state(next_board, next_turn)
            states_visited.add(next_state)
            legal_next = game.get_legal_actions(next_board)

            # Update current move with minimax backup
            update_Q_minimax(Q, state, action, reward, next_state, legal, legal_next, alpha=0.1, gamma=0.9)

            # Save this move as "previous" (for possible loss penalty on next terminal)
            prev_state = state
            prev_action = action
            prev_legal = legal
            prev_turn = turn

            board = next_board
            turn = next_turn

        if (i + 1) % 1000 == 0 and i > 0:
            total = count_X_wins + count_O_wins + count_draws
            draw_rates.append(count_draws / total if total > 0 else 0)
            episodes.append(i+1)
            count_O_wins = 0
            count_X_wins = 0
            count_draws = 0

    print(f"After {games} games (self-play): X wins: {count_X_wins}, O wins: {count_O_wins}, Draws: {count_draws}")
    plt.figure()
    plt.plot(episodes, draw_rates, label='Draw Rate')
    plt.xlabel('Episodes')
    plt.ylabel('Draw Rate')
    plt.title('Self-Play Draw Rate Over Time')
    plt.savefig(PLOT_PATH + '/self_play_draw_rate.png')
    plt.close()



def evaluate_self_play_q_agent(game: TicTacToe_N_K, Q: dict, games: int) -> None:
    count_X_wins = 0
    count_O_wins = 0
    count_draws = 0

    for _ in range(games):
        board = game.reset()
        done = False
        turn = random.choice([1, -1])

        while not done:
            state = game.encode_state(board, turn)
            legal = game.get_legal_actions(board)

            action = e_greedy_selection(Q, state, legal, epsilon=0.0)
            board, _, _, done = game.step(board, turn, action)

            if done:
                winner = game.check_winner(board)
                if winner == 1:
                    count_X_wins += 1
                elif winner == -1:
                    count_O_wins += 1
                else:
                    count_draws += 1
                break

            turn = -turn

    print(f"Self-play eval ({games} games): X wins: {count_X_wins}, O wins: {count_O_wins}, Draws: {count_draws}")

def preprocess(states_tensor):
    board_states = states_tensor[:, :-1] #Extract all columns expect of the last one
    turns = states_tensor[:, -1]    #Extract last column

    board_mapped = board_states.clone()
    #Transform {-1,1,0} encoding of cells to 0,1,2 to perform one hot encoding
    board_mapped[board_states == 1] = 0
    board_mapped[board_states == -1] = 1
    board_mapped[board_states == 0] = 2

    board_onehot = torch.nn.functional.one_hot(board_mapped, num_classes=3).float()
    board_flat = board_onehot.view(board_onehot.size(0), -1)

    turns_01 = (turns + 1) / 2  # Transform {-1,1} turn to {0,1} for consistency with one-hot encoding and to be used as an additional feature
    X = torch.cat([board_flat, turns_01.unsqueeze(1)], dim=1) # final input: one-hot encoding of board (3 features per cell) + 1 feature for turn 

    return X, board_mapped, turns_01.unsqueeze(1)    

def run_epoch(loader, training: bool, optimizer, beta):

    if training:
        vae.train()
    else:
        vae.eval()
    
    total_loss = 0.0
    total_re = 0.0
    total_kl = 0.0
    n_samples = 0

    for x_enc, y_cells, y_turn in loader:
        x_enc, y_cells, y_turn = x_enc.to(device), y_cells.to(device), y_turn.to(device)

        if training:
            optimizer.zero_grad()
        
        with torch.set_grad_enabled(training):
            loss, re, kl = vae(x_enc, y_cells, y_turn, beta=beta, reduction='avg') #average loss, re, kl per batch 
            if training:
                loss.backward() 
                optimizer.step()
        
        bs = x_enc.size(0)
        total_loss += loss.item() * bs
        total_re += re.item() * bs
        total_kl += kl.item() * bs
        n_samples += bs

        

    return total_loss / n_samples, total_re / n_samples, total_kl / n_samples #return average loss, re, kl per state over the whole dataset

def beta_schedule(epoch, warmup_epochs=10, max_beta=0.5):
    return min(max_beta, max_beta * (epoch / warmup_epochs)) 


# -----------------------* -----------------------*-----------------------*-----------------------*-----------------------*-----------------------*
            # RUN A DEMO FOR OPPONENT RANDOM, AGENT LEARNING WITH Q-LEARNING UPDATE
# -----------------------*-----------------------*-----------------------*-----------------------*-----------------------*-----------------------*

# ------------------------*-----------------------*-----------------------*-----------------------*-----------------------*-----------------------*
            # SET PARAMETERS FOR TIC TAC TOE
n = 3
k = 3
train_games = 50000
test_games = 200
# -----------------------*-----------------------*-----------------------*-----------------------*-----------------------*-----------------------*

if "train_states.pt" not in os.listdir() or "test_states.pt" not in os.listdir() or "val_states.pt" not in os.listdir():
    game = TicTacToe_N_K(n, k)
    states_visited = set()  # For tracking unique states visited during training

    # #Play and evaluate Q-learning agent against random opponent
    Q = {}
    play_game_q_agent(game, Q, epsilon=1.0, games=train_games)
    evaluate_q_agent(game, Q, games=test_games)

    #Play and evaluate Q-learning agent against minimax opponent
    Q={}
    play_game_q_minimax_opponent(game, Q, epsilon=1.0, games=train_games)
    evaluate_q_agent_minimax_opponent(game, Q, games=test_games)

    #Play and evaluate Q-learning agent against itself (self-play)
    Q = {}
    play_game_q_self_play(game, Q, games=train_games)
    evaluate_self_play_q_agent(game, Q, games=test_games)

    print(f"Total unique states visited during training: {len(states_visited)}")

    states_visited = list(states_visited)
        
    random.Random(42).shuffle(states_visited)  # Shuffle the states to ensure random order
    n = len(states_visited)

    train_ratio, val_ratio, test_ratio = 0.7, 0.15, 0.15
    train_size = int(train_ratio * n)
    val_size = int(val_ratio * n)
    test_size = n - train_size - val_size

    #Split dataset list to train, val, test
    train_states = states_visited[:train_size]
    test_states = states_visited[train_size:train_size + test_size]
    val_states = states_visited[train_size + test_size:]

    train_states_visited = np.array(train_states, dtype = np.int64)
    test_states_visited = np.array(test_states, dtype = np.int64)
    val_states_visited = np.array(val_states, dtype = np.int64)

    train_states_tensor = torch.tensor(train_states_visited, dtype=torch.int64)
    test_states_tensor = torch.tensor(test_states_visited, dtype=torch.int64)
    val_states_tensor = torch.tensor(val_states_visited, dtype=torch.int64)

    #Save tensors for future reference
    torch.save(train_states_tensor, "train_states.pt")
    torch.save(test_states_tensor, "test_states.pt")
    torch.save(val_states_tensor, "val_states.pt")
    print("States tensor saved to train_states.pt, test_states.pt, val_states.pt")

else:
    #Load pre-existed tensors
    train_states_tensor = torch.load("train_states.pt")
    test_states_tensor = torch.load("test_states.pt")
    val_states_tensor = torch.load("val_states.pt")
    print("States tensors loaded from train_states.pt, test_states.pt, val_states.pt")


X_train, y_cells_train, y_turn_train = preprocess(train_states_tensor)
X_val,   y_cells_val,   y_turn_val   = preprocess(val_states_tensor)
X_test,  y_cells_test,  y_turn_test  = preprocess(test_states_tensor)


train_dataset = TensorDataset(X_train, y_cells_train, y_turn_train)
val_dataset = TensorDataset(X_val, y_cells_val, y_turn_val)
test_dataset = TensorDataset(X_test, y_cells_test, y_turn_test)

#Load dataset in dataloader in order to pass data into batches.
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# ------------------------*-----------------------*-----------------------*-----------------------*-----------------------*-----------------------*
            #DEFINE VAE ARCHITECTURE
# -----------------------*-----------------------*-----------------------*-----------------------*-----------------------*-----------------------*

input_dim = X_train.shape[1]
latent_dim = 8 #will multiplied by 2
hidden_dim = 64
D = 9   #number of cells on the board for classic 3x3 Tic-Tac-Toe
num_moves = 3 #empty, X, O


#Define structure of encoder, decoder of VAE
encoder_net = nn.Sequential(
    nn.Linear(input_dim, hidden_dim),
    nn.ReLU(),
    nn.Linear(hidden_dim, hidden_dim),
    nn.ReLU(),
    nn.Linear(hidden_dim, latent_dim * 2)
)

decoder_net = nn.Sequential(
    nn.Linear(latent_dim, hidden_dim),
    nn.ReLU(),
    nn.Linear(hidden_dim, hidden_dim),
    nn.ReLU(),
    nn.Linear(hidden_dim, D*num_moves + 1)
)

vae = VAE(encoder_net, decoder_net, D = D, L = latent_dim, num_vals=num_moves)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Move model to GPU if available for faster training
vae.to(device)

optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)

# Define number of epochs and number of warm-up epochs. The warm-up epochs are those during which beta is gradually increased from 0 to 1, allowing the model to first focus on reconstruction before regularizing with KL divergence. After warm-up, beta remains at 1 for true ELBO optimization.
epochs = 30
warmup_epochs = 20

train_loss_list = []
train_re_list = []
train_kl_list = []
epoch_list = []
val_loss_list = []
val_re_list = []
val_kl_list = []

best_state = None
best_val_loss = float("inf")
for epoch in range(epochs): #each epoch pass through the whole training dataset once
    beta = beta_schedule(epoch, warmup_epochs, max_beta=1.0) # Linearly increase beta from 0 to max_beta over warmup_epochs, then keep it at max_beta
    train_loss, train_re, train_kl = run_epoch(
        train_dataloader, training=True, optimizer=optimizer, beta=beta
    )

    # evaluate with beta=1.0 (true ELBO scale)
    val_loss, val_re, val_kl = run_epoch(val_dataloader, False, optimizer=None, beta=1.0)
    

    print(
    f"Epoch {epoch:03d} | "
    f"beta={beta:.2f} | "
    f"Train: Loss={train_loss:.4f}, RE={train_re:.4f}, KL={train_kl:.4f} | "
    f"Val (beta=1): Loss={val_loss:.4f}, RE={val_re:.4f}, KL={val_kl:.4f}"
)

    if epoch >= warmup_epochs and val_loss < best_val_loss:
        best_val_loss = val_loss
        best_state = copy.deepcopy(vae.state_dict()) # Save the best model state based on validation loss after warm-up epochs since val loss and train loss are decreasing and they are close.

    if epoch % 2 == 0:
        train_loss_list.append(train_loss)
        train_re_list.append(train_re)
        train_kl_list.append(train_kl)
        epoch_list.append(epoch)
        val_loss_list.append(val_loss)
        val_re_list.append(val_re)
        val_kl_list.append(val_kl) # Store training and validation metrics every 2 epochs for plotting

plt.figure()
plt.plot(epoch_list, train_loss_list, 'bo-', label='Train Loss')
plt.plot(epoch_list, val_loss_list, 'co-', label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Total Loss Over Epochs')
plt.legend()
plt.savefig(PLOT_PATH + '/training_loss.png')
plt.close()

plt.figure()
plt.plot(epoch_list, train_re_list, 'ro-', label='Train RE')
plt.plot(epoch_list, val_re_list, 'mo-', label='Val RE')
plt.xlabel('Epoch')
plt.ylabel('Reconstruction_Error')
plt.title('Reconstruction Error Over Epochs')
plt.legend()
plt.savefig(PLOT_PATH + '/training_re.png')
plt.close()

plt.figure()
plt.plot(epoch_list, train_kl_list, 'go-', label='Train KL')
plt.plot(epoch_list, val_kl_list, 'yo-', label='Val KL')
plt.xlabel('Epoch')
plt.ylabel('KL_Divergence')
plt.title('KL Divergence Over Epochs')
plt.legend()
plt.savefig(PLOT_PATH + '/training_kl.png')
plt.close()

if best_state is not None:
    vae.load_state_dict(best_state)
    torch.save(best_state, "vae_best.pt") # Save the best model state to a file for future reference
test_loss, test_re, test_kl = run_epoch(test_dataloader, False, optimizer=None, beta=1.0) # Evaluate the best model on the test set using true ELBO scale (beta=1.0) to report final performance metrics on unseen states
print("TEST:", test_loss, test_re, test_kl)

