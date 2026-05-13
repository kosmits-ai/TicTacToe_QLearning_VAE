from typing import List, Tuple
import random
import pickle
import os
import copy

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist

from vae.model import VAE

Board = List[int]
Turn = int

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

PLOT_PATH = "./plots"
os.makedirs(PLOT_PATH, exist_ok=True)

Q_RANDOM_PATH = "Q_random.pkl"
Q_MINIMAX_PATH = "Q_minimax.pkl"
Q_SELF_PATH = "Q_self.pkl"

TRAIN_STATES_SELF_PATH = "train_states_self.pt"
VAL_STATES_SELF_PATH = "val_states_self.pt"
TEST_STATES_SELF_PATH = "test_states_self.pt"
VISITED_SELF_PATH = "visited_states_self.pt"


class TicTacToe_N_K:
    def __init__(self, n: int, k: int):
        self.n = n
        self.k = k

    def reset(self) -> Board:
        return [0] * (self.n * self.n)

    def encode_state(self, board: Board, turn: Turn) -> Tuple[int, ...]:
        return tuple(board + [turn])

    def get_legal_actions(self, board: Board) -> List[int]:
        return [i for i, v in enumerate(board) if v == 0]

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
        # rows
        for r in range(self.n):
            for c in range(self.n - self.k + 1):
                first = board[r * self.n + c]
                if first == 0:
                    continue
                if all(board[r * self.n + c + i] == first for i in range(self.k)):
                    return first

        # cols
        for c in range(self.n):
            for r in range(self.n - self.k + 1):
                first = board[r * self.n + c]
                if first == 0:
                    continue
                if all(board[(r + i) * self.n + c] == first for i in range(self.k)):
                    return first

        # main diagonals
        for r in range(self.n - self.k + 1):
            for c in range(self.n - self.k + 1):
                first = board[r * self.n + c]
                if first == 0:
                    continue
                if all(board[(r + t) * self.n + (c + t)] == first for t in range(self.k)):
                    return first

        # anti-diagonals
        for r in range(self.n - self.k + 1):
            for c in range(self.k - 1, self.n):
                first = board[r * self.n + c]
                if first == 0:
                    continue
                if all(board[(r + t) * self.n + (c - t)] == first for t in range(self.k)):
                    return first

        return 0

    def print_board(self, board: Board):
        symbols = {0: "-", 1: "X", -1: "O"}
        for i in range(self.n):
            row = [symbols[board[i * self.n + j]] for j in range(self.n)]
            print(" ".join(row))


def save_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def update_Q_zero(
    Q: dict,
    state: Tuple[int, ...],
    action: int,
    reward: float,
    next_state,
    legal_moves: List[int],
    legal_moves_next: List[int],
    alpha: float,
    gamma: float
):
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


def update_Q_minimax(
    Q: dict,
    state: Tuple[int, ...],
    action: int,
    reward: float,
    next_state,
    legal_moves: List[int],
    legal_moves_next: List[int],
    alpha: float,
    gamma: float
):
    if state not in Q:
        Q[state] = {a: 0.0 for a in legal_moves}

    if not legal_moves_next or next_state is None:
        max_next_Q = 0.0
    else:
        if next_state not in Q:
            Q[next_state] = {a: 0.0 for a in legal_moves_next}
        max_next_Q = max(Q[next_state].values())

    target = reward - gamma * max_next_Q
    Q[state][action] += alpha * (target - Q[state][action])
    return Q


def e_greedy_selection(Q: dict, state: Tuple[int, ...], legal_moves: List[int], epsilon: float) -> int:
    if random.random() < epsilon:
        return random.choice(legal_moves)

    state_Q = Q.get(state, {})
    max_Q = max(state_Q.get(a, 0.0) for a in legal_moves)
    best_actions = [a for a in legal_moves if state_Q.get(a, 0.0) == max_Q]
    return random.choice(best_actions)


def scheduled_epsilon(t: int, epsilon_start: float, epsilon_end: float, decay_steps: int) -> float:
    if t >= decay_steps:
        return epsilon_end
    return epsilon_start - (epsilon_start - epsilon_end) * (t / decay_steps)


def play_game_q_agent(game: TicTacToe_N_K, Q: dict, epsilon: float, games: int, states_visited: set) -> None:
    count_X_wins = 0
    count_draws = 0
    count_O_wins = 0
    loss_rates = []
    episodes = []

    total_X_wins = 0
    total_O_wins = 0
    total_draws = 0

    for i in range(games):
        board = game.reset()
        done = False
        start_turn = random.choice([1, -1])

        if start_turn == -1:
            states_visited.add(game.encode_state(board, -1))
            legal_O = game.get_legal_actions(board)
            action_O = random.choice(legal_O)
            board, _, _, done = game.step(board, -1, action_O)

        epsilon_i = scheduled_epsilon(i, epsilon_start=1.0, epsilon_end=0.1, decay_steps=games // 2)

        while not done:
            state_X = game.encode_state(board, 1)
            legal_actions_X = game.get_legal_actions(board)
            action_X = e_greedy_selection(Q, state_X, legal_actions_X, epsilon_i)
            board_after_X, _, reward_X, done = game.step(board, 1, action_X)

            states_visited.add(state_X)

            if done:
                states_visited.add(game.encode_state(board_after_X, -1))
                update_Q_zero(Q, state_X, action_X, reward_X, None, legal_actions_X, [], alpha=0.1, gamma=0.9)
                winner = game.check_winner(board_after_X)
                if winner == 1:
                    count_X_wins += 1
                    total_X_wins += 1
                elif winner == -1:
                    count_O_wins += 1
                    total_O_wins += 1
                else:
                    count_draws += 1
                    total_draws += 1
                break

            states_visited.add(game.encode_state(board_after_X, -1))
            legal_O = game.get_legal_actions(board_after_X)
            action_O = random.choice(legal_O)
            board_after_O, _, reward_O, done = game.step(board_after_X, -1, action_O)

            if done:
                states_visited.add(game.encode_state(board_after_O, 1))

            if done and reward_O == 1.0:
                r_for_X = -1.0
                legal_next_for_X = []
                count_O_wins += 1
                total_O_wins += 1
            elif done:
                r_for_X = 0.0
                legal_next_for_X = []
                count_draws += 1
                total_draws += 1
            else:
                r_for_X = 0.0
                legal_next_for_X = game.get_legal_actions(board_after_O)

            next_state_X = game.encode_state(board_after_O, 1) if legal_next_for_X else None
            update_Q_zero(Q, state_X, action_X, r_for_X, next_state_X, legal_actions_X, legal_next_for_X, alpha=0.1, gamma=0.9)

            board = board_after_O

        if (i + 1) % 1000 == 0 and i > 0:
            total = count_X_wins + count_O_wins + count_draws
            loss_rates.append(count_O_wins / total if total > 0 else 0)
            episodes.append(i + 1)
            count_O_wins = 0
            count_X_wins = 0
            count_draws = 0

    plt.figure()
    plt.plot(episodes, loss_rates, label="Loss Rate (O wins)")
    plt.xlabel("Episodes")
    plt.ylabel("Loss Rate")
    plt.title("Loss Rate of Q-Learning Agent Over Time")
    plt.savefig(PLOT_PATH + "/q_learning_random_loss_rate.png")
    plt.close()

    print(f"After {games} games: X wins: {total_X_wins}, O wins: {total_O_wins}, Draws: {total_draws}")


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
                if winner == 1:
                    count_X_wins += 1
                elif winner == -1:
                    count_O_wins += 1
                else:
                    count_draws += 1
                continue

        while not done:
            state_X = game.encode_state(board, 1)
            legal_actions_X = game.get_legal_actions(board)
            action_X = e_greedy_selection(Q, state_X, legal_actions_X, epsilon=0.0)
            board_after_X, _, _, done = game.step(board, 1, action_X)

            if done:
                winner = game.check_winner(board_after_X)
                if winner == 1:
                    count_X_wins += 1
                elif winner == -1:
                    count_O_wins += 1
                else:
                    count_draws += 1
                break

            legal_O = game.get_legal_actions(board_after_X)
            action_O = random.choice(legal_O)
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

    print(f"Evaluation after {games} games: X wins: {count_X_wins}, O wins: {count_O_wins}, Draws: {count_draws}")


def terminal_score(game, board):
    winner = game.check_winner(board)
    if winner == -1:
        return 1.0
    if winner == 1:
        return -1.0
    if not game.get_legal_actions(board):
        return 0.0
    return None


def alphabeta(game, board, turn, alpha, beta):
    score = terminal_score(game, board)
    if score is not None:
        return score

    legal = game.get_legal_actions(board)
    if turn == -1:
        value = -float("inf")
        for action in legal:
            nb = board.copy()
            nb[action] = turn
            value = max(value, alphabeta(game, nb, 1, alpha, beta))
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return value
    else:
        value = float("inf")
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
    best_score = float("-inf")
    best_actions = []
    alpha, beta = -float("inf"), float("inf")

    for action in legal:
        nb = board.copy()
        nb[action] = turn
        score = alphabeta(game, nb, -turn, alpha, beta)

        if score > best_score:
            best_score = score
            best_actions = [action]
            alpha = max(alpha, best_score)
        elif score == best_score:
            best_actions.append(action)

        alpha = max(alpha, best_score)

    return random.choice(best_actions)


def play_game_q_minimax_opponent(game: TicTacToe_N_K, Q: dict, epsilon: float, games: int, states_visited: set) -> None:
    count_X_wins = 0
    count_draws = 0
    count_O_wins = 0
    loss_rates = []
    episodes = []

    total_X_wins = 0
    total_O_wins = 0
    total_draws = 0

    for i in range(games):
        board = game.reset()
        done = False
        start_turn = random.choice([1, -1])

        if start_turn == -1:
            states_visited.add(game.encode_state(board, -1))
            action_O = minimax_move_alphabeta(game, board, turn=-1)
            board, _, _, done = game.step(board, -1, action_O)

        epsilon_i = scheduled_epsilon(i, epsilon_start=1.0, epsilon_end=0.1, decay_steps=games // 2)

        while not done:
            state_X = game.encode_state(board, 1)
            states_visited.add(state_X)
            legal_actions_X = game.get_legal_actions(board)
            action_X = e_greedy_selection(Q, state_X, legal_actions_X, epsilon_i)
            board_after_X, _, reward_X, done = game.step(board, 1, action_X)

            if done:
                states_visited.add(game.encode_state(board_after_X, -1))
                update_Q_zero(Q, state_X, action_X, reward_X, None, legal_actions_X, [], alpha=0.1, gamma=0.9)
                winner = game.check_winner(board_after_X)
                if winner == 1:
                    count_X_wins += 1
                    total_X_wins += 1
                elif winner == -1:
                    count_O_wins += 1
                    total_O_wins += 1
                else:
                    count_draws += 1
                    total_draws += 1
                break

            states_visited.add(game.encode_state(board_after_X, -1))
            action_O = minimax_move_alphabeta(game, board_after_X, turn=-1)
            board_after_O, _, reward_O, done = game.step(board_after_X, -1, action_O)

            if done:
                states_visited.add(game.encode_state(board_after_O, 1))

            if done and reward_O == 1.0:
                r_for_X = -1.0
                legal_next_for_X = []
                count_O_wins += 1
                total_O_wins += 1
            elif done:
                r_for_X = 0.0
                legal_next_for_X = []
                count_draws += 1
                total_draws += 1
            else:
                r_for_X = 0.0
                legal_next_for_X = game.get_legal_actions(board_after_O)

            next_state_X = game.encode_state(board_after_O, 1) if legal_next_for_X else None
            update_Q_zero(Q, state_X, action_X, r_for_X, next_state_X, legal_actions_X, legal_next_for_X, alpha=0.1, gamma=0.9)

            board = board_after_O

        if (i + 1) % 1000 == 0 and i > 0:
            total = count_X_wins + count_O_wins + count_draws
            loss_rates.append(count_O_wins / total if total > 0 else 0)
            episodes.append(i + 1)
            count_O_wins = 0
            count_X_wins = 0
            count_draws = 0

    print(f"After {games} games: X wins: {total_X_wins}, O wins: {total_O_wins}, Draws: {total_draws}")
    plt.figure()
    plt.plot(episodes, loss_rates, label="Loss Rate (O wins)")
    plt.xlabel("Episodes")
    plt.ylabel("Loss Rate")
    plt.title("Loss Rate of Q-Learning Agent Over Time")
    plt.savefig(PLOT_PATH + "/q_learning_minimax_loss_rate.png")
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
                if winner == 1:
                    count_X_wins += 1
                elif winner == -1:
                    count_O_wins += 1
                else:
                    count_draws += 1
                continue

        while not done:
            state_X = game.encode_state(board, 1)
            legal_actions_X = game.get_legal_actions(board)
            action_X = e_greedy_selection(Q, state_X, legal_actions_X, epsilon=0.0)
            board_after_X, _, _, done = game.step(board, 1, action_X)

            if done:
                winner = game.check_winner(board_after_X)
                if winner == 1:
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

    print(f"Evaluation after {games} games: X wins: {count_X_wins}, O wins: {count_O_wins}, Draws: {count_draws}")


def play_game_q_self_play(game: TicTacToe_N_K, Q: dict, games: int, states_visited: set) -> None:
    count_X_wins = 0
    count_O_wins = 0
    count_draws = 0

    total_X_wins = 0
    total_O_wins = 0
    total_draws = 0

    draw_rates = []
    episodes = []

    for i in range(games):
        board = game.reset()
        done = False
        turn = random.choice([1, -1])

        epsilon_i = scheduled_epsilon(i, epsilon_start=1.0, epsilon_end=0.1, decay_steps=games // 2)

        prev_state = None
        prev_action = None
        prev_legal = None

        while not done:
            state = game.encode_state(board, turn)
            states_visited.add(state)
            legal = game.get_legal_actions(board)

            action = e_greedy_selection(Q, state, legal, epsilon_i)
            next_board, _, reward, done = game.step(board, turn, action)

            if done:
                states_visited.add(game.encode_state(next_board, -turn))
                update_Q_minimax(Q, state, action, reward, None, legal, [], alpha=0.1, gamma=0.9)

                winner = game.check_winner(next_board)

                if winner != 0 and prev_state is not None and prev_action is not None:
                    update_Q_minimax(
                        Q, prev_state, prev_action, reward=-1.0, next_state=None,
                        legal_moves=prev_legal, legal_moves_next=[],
                        alpha=0.1, gamma=0.9
                    )

                if winner == 1:
                    count_X_wins += 1
                    total_X_wins += 1
                elif winner == -1:
                    count_O_wins += 1
                    total_O_wins += 1
                else:
                    count_draws += 1
                    total_draws += 1
                break

            next_turn = -turn
            next_state = game.encode_state(next_board, next_turn)
            states_visited.add(next_state)
            legal_next = game.get_legal_actions(next_board)

            update_Q_minimax(Q, state, action, reward, next_state, legal, legal_next, alpha=0.1, gamma=0.9)

            prev_state = state
            prev_action = action
            prev_legal = legal

            board = next_board
            turn = next_turn

        if (i + 1) % 1000 == 0 and i > 0:
            total = count_X_wins + count_O_wins + count_draws
            draw_rates.append(count_draws / total if total > 0 else 0)
            episodes.append(i + 1)
            count_O_wins = 0
            count_X_wins = 0
            count_draws = 0

    print(f"After {games} games (self-play): X wins: {total_X_wins}, O wins: {total_O_wins}, Draws: {total_draws}")
    plt.figure()
    plt.plot(episodes, draw_rates, label="Draw Rate")
    plt.xlabel("Episodes")
    plt.ylabel("Draw Rate")
    plt.title("Self-Play Draw Rate Over Time")
    plt.savefig(PLOT_PATH + "/self_play_draw_rate.png")
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
    board_states = states_tensor[:, :-1]
    turns = states_tensor[:, -1]

    board_mapped = board_states.clone()
    board_mapped[board_states == 1] = 0
    board_mapped[board_states == -1] = 1
    board_mapped[board_states == 0] = 2

    board_onehot = torch.nn.functional.one_hot(board_mapped, num_classes=3).float()
    board_flat = board_onehot.view(board_onehot.size(0), -1)

    turns_01 = (turns + 1) / 2
    X = torch.cat([board_flat, turns_01.unsqueeze(1)], dim=1)

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
            loss, re, kl = vae(x_enc, y_cells, y_turn, beta=beta, reduction="avg")
            if training:
                loss.backward()
                optimizer.step()

        bs = x_enc.size(0)
        total_loss += loss.item() * bs
        total_re += re.item() * bs
        total_kl += kl.item() * bs
        n_samples += bs

    return total_loss / n_samples, total_re / n_samples, total_kl / n_samples


def beta_schedule(epoch, warmup_epochs=10, max_beta=0.5):
    return min(max_beta, max_beta * (epoch / warmup_epochs))


def reconstruct_batch(vae, x_enc):
    vae.eval()
    with torch.no_grad():
        mu, logvar = vae.encoder.encode(x_enc)
        z = mu
        board_logits, turn_logit = vae.decoder.decode(z)
        board_pred = board_logits.argmax(dim=-1)
        turn_pred = (torch.sigmoid(turn_logit) > 0.5).float()
    return board_pred, turn_pred


def evaluate_reconstruction(vae, loader):
    vae.eval()
    total_cells = 0
    correct_cells = 0
    total_turns = 0
    correct_turns = 0

    with torch.no_grad():
        for x_enc, y_cells, y_turn in loader:
            y_cells = y_cells.to(device)
            y_turn = y_turn.to(device)
            board_pred, turn_pred = reconstruct_batch(vae, x_enc.to(device))

            total_cells += y_cells.numel()
            correct_cells += (board_pred == y_cells).sum().item()
            total_turns += y_turn.numel()
            correct_turns += (turn_pred == y_turn).sum().item()

    cell_acc = correct_cells / total_cells
    turn_acc = correct_turns / total_turns
    return cell_acc, turn_acc


def print_board_mapped(board):
    symbols = {0: "X", 1: "O", 2: "."}
    for i in range(3):
        row = board[i * 3:(i + 1) * 3]
        print(" ".join(symbols[int(c)] for c in row))
    print()


def convert_mapped_to_original(board_mapped):
    board_orig = []
    for v in board_mapped:
        if v == 0:
            board_orig.append(1)
        elif v == 1:
            board_orig.append(-1)
        else:
            board_orig.append(0)
    return board_orig


def visualize_latent_space(vae, loader):
    vae.eval()

    all_mu = []
    all_turns = []
    all_num_filled = []
    all_winners = []

    game = TicTacToe_N_K(3, 3)

    with torch.no_grad():
        for x_enc, y_cells, y_turn in loader:
            x_enc = x_enc.to(device)

            mu, _ = vae.encoder.encode(x_enc)
            all_mu.append(mu.cpu())
            all_turns.append(y_turn.cpu())

            num_filled = (y_cells != 2).sum(dim=1)
            all_num_filled.append(num_filled.cpu())

            yc = y_cells.cpu().numpy()
            winners_batch = []
            for i in range(yc.shape[0]):
                board_orig = convert_mapped_to_original(yc[i])
                w = game.check_winner(board_orig)
                if w == 1:
                    winners_batch.append(1)
                elif w == -1:
                    winners_batch.append(2)
                else:
                    winners_batch.append(0)

            all_winners.append(torch.tensor(winners_batch))

    all_mu = torch.cat(all_mu).numpy()
    all_turns = torch.cat(all_turns).numpy().flatten()
    all_num_filled = torch.cat(all_num_filled).numpy()
    all_winners = torch.cat(all_winners).numpy().flatten()

    pca = PCA(n_components=2)
    z_2d = pca.fit_transform(all_mu)

    print("Explained variance ratio:", pca.explained_variance_ratio_)
    unique, counts = np.unique(all_winners, return_counts=True)
    print("Winner counts:", dict(zip(unique, counts)))

    plt.figure(figsize=(6, 6))
    for turn_value, color, label in [(0, "blue", "O turn"), (1, "red", "X turn")]:
        mask = (all_turns == turn_value)
        plt.scatter(z_2d[mask, 0], z_2d[mask, 1], c=color, label=label, alpha=0.6)
    plt.legend()
    plt.title("PCA of Latent Space Colored by Turn")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.savefig(PLOT_PATH + "/latent_space_turn.png")
    plt.close()

    plt.figure(figsize=(6, 6))
    unique_progress = np.unique(all_num_filled)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_progress)))
    for prog_value, color in zip(unique_progress, colors):
        mask = (all_num_filled == prog_value)
        plt.scatter(z_2d[mask, 0], z_2d[mask, 1], color=color, label=f"{prog_value} filled", alpha=0.6)
    plt.legend(title="Game Progress", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.title("PCA of Latent Space Colored by Game Progress")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig(PLOT_PATH + "/latent_space_progress.png")
    plt.close()

    plt.figure(figsize=(6, 6))
    winner_labels = {
        0: ("gray", "No Winner"),
        1: ("green", "X Wins"),
        2: ("orange", "O Wins")
    }
    for w, (color, label) in winner_labels.items():
        mask = (all_winners == w)
        if mask.sum() > 0:
            plt.scatter(z_2d[mask, 0], z_2d[mask, 1], c=color, label=label, alpha=0.6)
    plt.legend()
    plt.title("PCA of Latent Space Colored by Winner")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.savefig(PLOT_PATH + "/latent_space_winner.png")
    plt.close()


def encode_single_state(board, turn, device):
    b = torch.tensor(board, dtype=torch.int64, device=device).unsqueeze(0)

    b_mapped = b.clone()
    b_mapped[b == 1] = 0
    b_mapped[b == -1] = 1
    b_mapped[b == 0] = 2

    b_oh = torch.nn.functional.one_hot(b_mapped, num_classes=3).float()
    b_flat = b_oh.view(1, -1)

    if turn in (-1, 1):
        t01 = torch.tensor([[(turn + 1) / 2]], dtype=torch.float32, device=device)
    elif turn in (0, 1):
        t01 = torch.tensor([[turn]], dtype=torch.float32, device=device)
    else:
        raise ValueError(f"Unexpected turn value: {turn}")

    x_enc = torch.cat([b_flat, t01], dim=1)
    return x_enc


def interpolate_between_boards(vae, board1, turn1, board2, turn2, steps=10):
    vae.eval()
    with torch.no_grad():
        x1 = encode_single_state(board1, turn1, device)
        x2 = encode_single_state(board2, turn2, device)

        mu1, _ = vae.encoder.encode(x1)
        mu2, _ = vae.encoder.encode(x2)

        interpolated_boards = []
        for alpha in np.linspace(0, 1, steps):
            z_interp = (1 - alpha) * mu1 + alpha * mu2
            board_logits, turn_logit = vae.decoder.decode(z_interp)
            board_pred = board_logits.argmax(dim=-1).squeeze(0).cpu().numpy()
            turn_pred = (torch.sigmoid(turn_logit) > 0.5).float().item()
            interpolated_boards.append((board_pred, turn_pred))

    return interpolated_boards


def winner(board, player):
    b = board.reshape(3, 3)
    lines = []
    lines.extend(b)
    lines.extend(b.T)
    lines.append([b[0, 0], b[1, 1], b[2, 2]])
    lines.append([b[0, 2], b[1, 1], b[2, 0]])
    return any(all(cell == player for cell in line) for line in lines)

def validity_board_check(board, turn):
    """
    board: mapped board with values {0:X, 1:O, 2:empty}
    turn:  mapped turn with values {1:X turn, 0:O turn}

    Conventions:
      board 0 = X, 1 = O, 2 = empty
      turn  1 = X to play next, 0 = O to play next
    """
    board = np.asarray(board)

    if not np.all(np.isin(board, [0, 1, 2])):
        return False

    if int(turn) not in [0, 1]:
        return False

    cells_X = int((board == 0).sum())
    cells_O = int((board == 1).sum())
    cells_empty = int((board == 2).sum())

    if cells_X + cells_O + cells_empty != 9:
        return False

    # Standard tic-tac-toe reachability: X starts
    if not (cells_X == cells_O or cells_X == cells_O + 1):
        return False

    X_win = winner(board, 0)
    O_win = winner(board, 1)

    if X_win and O_win:
        return False

    # Terminal winner parity
    if X_win:
        if cells_X != cells_O + 1:
            return False
        # after X just moved, next turn would be O
        if turn != 0:
            return False
        return True

    if O_win:
        if cells_X != cells_O:
            return False
        # after O just moved, next turn would be X
        if turn != 1:
            return False
        return True

    # Non-terminal states: next player determined by move parity
    if cells_X == cells_O:
        # X to play
        if turn != 1:
            return False
    elif cells_X == cells_O + 1:
        # O to play
        if turn != 0:
            return False

    return True

def build_latent_dataset(vae, states_tensor):
    vae.eval()
    with torch.no_grad():
        X, _, _ = preprocess(states_tensor)
        X = X.to(device)
        mu, _ = vae.encoder.encode(X)
    return mu.cpu().numpy()


def get_knn_indices(z_query, z_dataset, k=10):
    distances = cdist(z_query.reshape(1, -1), z_dataset, metric="euclidean")[0]
    k_eff = min(k, len(distances))
    idx = np.argsort(distances)[:k_eff]
    return idx, distances[idx]

def estimate_Q_from_latent(z_query, z_dataset, states_dataset, Q, legal_moves, k=10):
    """
    Q2.4(e):
      Q_tilde(x~, a) = sum_i w_i Q(x_i, a) / sum_i w_i
      w_i = exp( - ||z-z_i||^2 / sigma^2 )
    with sigma = median distance among the K nearest neighbors.

    Only neighbors that actually have Q-values are used.
    """
    idx, distances = get_knn_indices(z_query, z_dataset, k=k)

    if len(distances) == 0:
        return {a: 0.0 for a in legal_moves}

    sigma = np.median(distances)
    if sigma <= 1e-8:
        sigma = 1e-8

    raw_weights = np.exp(-(distances ** 2) / (sigma ** 2))

    Q_num = {a: 0.0 for a in legal_moves}
    Q_den = {a: 0.0 for a in legal_moves}

    for w, i in zip(raw_weights, idx):
        neighbor_state = tuple(states_dataset[i])

        if neighbor_state not in Q or len(Q[neighbor_state]) == 0:
            continue

        neighbor_Q = Q[neighbor_state]

        for a in legal_moves:
            Q_num[a] += w * neighbor_Q.get(a, 0.0)
            Q_den[a] += w

    Q_agg = {}
    for a in legal_moves:
        if Q_den[a] > 0:
            Q_agg[a] = Q_num[a] / Q_den[a]
        else:
            Q_agg[a] = 0.0

    return Q_agg

def select_action_baseline(Q, state, legal_moves):
    """
    Baseline from Q2.4(e):
    - if the state is unseen / has no Q-values -> uniform random legal move
    - otherwise use tabular Q(state, action)
    """
    if state not in Q or len(Q[state]) == 0:
        return random.choice(legal_moves)

    state_Q = Q[state]
    max_q = max(state_Q.get(a, 0.0) for a in legal_moves)
    best_actions = [a for a in legal_moves if state_Q.get(a, 0.0) == max_q]
    return random.choice(best_actions)


def select_action_vae(vae, Q, board, turn, legal_moves, z_dataset, states_dataset, k=10):
    x = encode_single_state(board, turn, device)

    with torch.no_grad():
        mu, _ = vae.encoder.encode(x)
        z_query = mu.cpu().numpy()[0]

    Q_est = estimate_Q_from_latent(
        z_query=z_query,
        z_dataset=z_dataset,
        states_dataset=states_dataset,
        Q=Q,
        legal_moves=legal_moves,
        k=k
    )

    max_q = max(Q_est[a] for a in legal_moves)
    best_actions = [a for a in legal_moves if Q_est[a] == max_q]
    return random.choice(best_actions)


def is_safe_action(game, board, action, turn):
    """
    Safe iff after this action the opponent has NO immediate winning move.
    """
    next_board = board.copy()
    next_board[action] = turn

    if game.check_winner(next_board) == turn:
        return True

    opp = -turn
    opp_legal = game.get_legal_actions(next_board)

    for a in opp_legal:
        tmp = next_board.copy()
        tmp[a] = opp
        if game.check_winner(tmp) == opp:
            return False

    return True


def convert_turn_to_original(turn_sampled):
    # mapped: 1 -> X turn, 0 -> O turn
    return 1 if int(turn_sampled) == 1 else -1


def sample_from_vae_prior(vae, num_samples=1000):
    """
    Q2.4(a): sample z ~ N(0, I) and decode.
    Returns mapped boards {0,1,2} and mapped turns {0,1}.
    """
    vae.eval()
    boards, turns = vae.sample(batch_size=num_samples, device=device)
    return boards, turns


def decode_and_filter_valid_states(boards, turns):
    """
    Q2.4(b)(c): keep only legal decoded states.
    Also remove duplicates so counts refer to unique states.
    """
    valid_states = []
    seen_valid = set()

    total = len(boards)

    for i in range(total):
        board_sampled = boards[i].cpu().numpy()
        turn_sampled = int(turns[i].item())

        if not validity_board_check(board_sampled, turn_sampled):
            continue

        board_orig = convert_mapped_to_original(board_sampled)
        turn_orig = convert_turn_to_original(turn_sampled)
        state = tuple(board_orig + [turn_orig])

        if state not in seen_valid:
            seen_valid.add(state)
            valid_states.append(state)

    valid_count = len(valid_states)
    acceptance_rate = valid_count / total if total > 0 else 0.0
    return valid_states, valid_count, acceptance_rate

def select_unseen_generated_states(valid_states, visited_states_set):
    """
    Q2.4(d):
    unseen iff the exact encoded state never appeared in the visited-state table
    of the same Q-learning regime being evaluated.
    Also deduplicate unseen states explicitly.
    """
    unseen = []
    seen_unseen = set()

    for s in valid_states:
        if s not in visited_states_set and s not in seen_unseen:
            seen_unseen.add(s)
            unseen.append(s)

    return unseen
def evaluate_generalization_2_4(game, vae, Q, z_dataset, states_dataset, generated_states, max_states=200, k=10):
    safe_baseline = 0
    safe_vae = 0
    total = 0

    for state in generated_states[:max_states]:
        board = list(state[:-1])
        turn = state[-1]

        legal_moves = game.get_legal_actions(board)
        if len(legal_moves) == 0:
            continue

        baseline_action = select_action_baseline(Q, state, legal_moves)

        vae_action = select_action_vae(
            vae=vae,
            Q=Q,
            board=board,
            turn=turn,
            legal_moves=legal_moves,
            z_dataset=z_dataset,
            states_dataset=states_dataset,
            k=k
        )

        if is_safe_action(game, board, baseline_action, turn):
            safe_baseline += 1

        if is_safe_action(game, board, vae_action, turn):
            safe_vae += 1

        total += 1

    if total == 0:
        print("No valid unseen non-terminal generated states were available for evaluation.")
        return {
            "evaluated_unseen_states": 0,
            "baseline_safe_rate": 0.0,
            "vae_safe_rate": 0.0,
            "evaluation_cap": max_states
        }

    baseline_rate = safe_baseline / total
    vae_rate = safe_vae / total

    print("\n--- Generalization Evaluation (Question 2.4) ---")
    print(f"Evaluation cap (max unseen states considered): {max_states}")
    print(f"Evaluated unseen non-terminal states: {total}")
    print(f"Baseline safe-action rate: {baseline_rate:.4f}")
    print(f"VAE latent-kNN safe-action rate: {vae_rate:.4f}")

    return {
        "evaluated_unseen_states": total,
        "baseline_safe_rate": baseline_rate,
        "vae_safe_rate": vae_rate,
        "evaluation_cap": max_states
    }

def print_q24_summary(total_generated, valid_count, acceptance_rate, unseen_count, eval_results):
    print("\n" + "=" * 68)
    print("Question 2.4 Summary")
    print("=" * 68)
    print(f"{'Total generated samples':35s}: {total_generated}")
    print(f"{'Valid decoded unique samples':35s}: {valid_count}")
    print(f"{'Acceptance rate':35s}: {acceptance_rate:.4f}")
    print(f"{'Unseen valid unique samples':35s}: {unseen_count}")
    print(f"{'Evaluation cap':35s}: {eval_results['evaluation_cap']}")
    print(f"{'Evaluated unseen non-terminal samples':35s}: {eval_results['evaluated_unseen_states']}")
    print(f"{'Baseline safe-action rate':35s}: {eval_results['baseline_safe_rate']:.4f}")
    print(f"{'VAE-based safe-action rate':35s}: {eval_results['vae_safe_rate']:.4f}")
    print("=" * 68)

    print("\nInterpretation:")
    print(
        "We generated candidate states from the VAE prior, retained unique valid decodings, "
        "selected the subset of unique unseen states relative to the visited-state table, "
        "and evaluated up to the specified cap of unseen non-terminal boards."
    )
    print(
        "If the VAE-based agent achieves a higher safe-action rate on unseen boards, "
        "this suggests that the latent representation captures useful structural "
        "similarities between board states and transfers value information beyond the "
        "exact states visited during tabular Q-learning."
    )

# -------------------- MAIN --------------------

if __name__ == "__main__":
    n = 3
    k = 3
    train_games = 1000
    test_games = 100

    game = TicTacToe_N_K(n, k)

    need_self_state_files = (
        not os.path.exists(TRAIN_STATES_SELF_PATH)
        or not os.path.exists(VAL_STATES_SELF_PATH)
        or not os.path.exists(TEST_STATES_SELF_PATH)
        or not os.path.exists(VISITED_SELF_PATH)
    )

    need_qtables = (
        not os.path.exists(Q_RANDOM_PATH)
        or not os.path.exists(Q_MINIMAX_PATH)
        or not os.path.exists(Q_SELF_PATH)
    )

    if need_self_state_files or need_qtables:
        states_visited_random = set()
        states_visited_minimax = set()
        states_visited_self = set()

        Q_random = {}
        play_game_q_agent(game, Q_random, epsilon=1.0, games=train_games, states_visited=states_visited_random)
        evaluate_q_agent(game, Q_random, games=test_games)

        Q_minimax = {}
        play_game_q_minimax_opponent(game, Q_minimax, epsilon=1.0, games=train_games, states_visited=states_visited_minimax)
        evaluate_q_agent_minimax_opponent(game, Q_minimax, games=test_games)

        Q_self = {}
        play_game_q_self_play(game, Q_self, games=train_games, states_visited=states_visited_self)
        evaluate_self_play_q_agent(game, Q_self, games=test_games)

        save_pickle(Q_random, Q_RANDOM_PATH)
        save_pickle(Q_minimax, Q_MINIMAX_PATH)
        save_pickle(Q_self, Q_SELF_PATH)

        print(f"Total unique self-play states visited during training: {len(states_visited_self)}")

        states_visited_self_list = list(states_visited_self)
        random.Random(42).shuffle(states_visited_self_list)
        total_n = len(states_visited_self_list)
        train_ratio, val_ratio, test_ratio = 0.7, 0.15, 0.15
        train_size = int(train_ratio * total_n)
        val_size = int(val_ratio * total_n)
        test_size = total_n - train_size - val_size

        train_states = states_visited_self_list[:train_size]
        test_states = states_visited_self_list[train_size:train_size + test_size]
        val_states = states_visited_self_list[train_size + test_size:]

        train_states_tensor = torch.tensor(np.array(train_states, dtype=np.int64), dtype=torch.int64)
        test_states_tensor = torch.tensor(np.array(test_states, dtype=np.int64), dtype=torch.int64)
        val_states_tensor = torch.tensor(np.array(val_states, dtype=np.int64), dtype=torch.int64)
        visited_self_tensor = torch.tensor(np.array(states_visited_self_list, dtype=np.int64), dtype=torch.int64)

        torch.save(train_states_tensor, TRAIN_STATES_SELF_PATH)
        torch.save(test_states_tensor, TEST_STATES_SELF_PATH)
        torch.save(val_states_tensor, VAL_STATES_SELF_PATH)
        torch.save(visited_self_tensor, VISITED_SELF_PATH)

        print("Self-play states tensor saved.")

    else:
        train_states_tensor = torch.load(TRAIN_STATES_SELF_PATH)
        test_states_tensor = torch.load(TEST_STATES_SELF_PATH)
        val_states_tensor = torch.load(VAL_STATES_SELF_PATH)
        visited_self_tensor = torch.load(VISITED_SELF_PATH)

        Q_random = load_pickle(Q_RANDOM_PATH)
        Q_minimax = load_pickle(Q_MINIMAX_PATH)
        Q_self = load_pickle(Q_SELF_PATH)

        print("Self-play state tensors loaded.")
        print("Q-tables loaded from pickle files")

    # ---------------- Dataset ----------------
    X_train, y_cells_train, y_turn_train = preprocess(train_states_tensor)
    X_val, y_cells_val, y_turn_val = preprocess(val_states_tensor)
    X_test, y_cells_test, y_turn_test = preprocess(test_states_tensor)

    train_dataset = TensorDataset(X_train, y_cells_train, y_turn_train)
    val_dataset = TensorDataset(X_val, y_cells_val, y_turn_val)
    test_dataset = TensorDataset(X_test, y_cells_test, y_turn_test)

    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # ---------------- VAE ----------------
    input_dim = X_train.shape[1]
    latent_dim = 8
    hidden_dim = 64
    D = 9
    num_moves = 3

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
        nn.Linear(hidden_dim, D * num_moves + 1)
    )

    vae = VAE(encoder_net, decoder_net, D=D, L=latent_dim, num_vals=num_moves)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae.to(device)

    optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)

    epochs = 50
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

    for epoch in range(epochs):
        beta = beta_schedule(epoch, warmup_epochs, max_beta=1.0)

        train_loss, train_re, train_kl = run_epoch(
            train_dataloader, training=True, optimizer=optimizer, beta=beta
        )

        val_loss, val_re, val_kl = run_epoch(
            val_dataloader, training=False, optimizer=None, beta=1.0
        )

        print(
            f"Epoch {epoch:03d} | "
            f"beta={beta:.2f} | "
            f"Train: Loss={train_loss:.4f}, RE={train_re:.4f}, KL={train_kl:.4f} | "
            f"Val (beta=1): Loss={val_loss:.4f}, RE={val_re:.4f}, KL={val_kl:.4f}"
        )

        if epoch >= warmup_epochs and val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(vae.state_dict())

        if epoch % 2 == 0:
            train_loss_list.append(train_loss)
            train_re_list.append(train_re)
            train_kl_list.append(train_kl)
            epoch_list.append(epoch)
            val_loss_list.append(val_loss)
            val_re_list.append(val_re)
            val_kl_list.append(val_kl)

    plt.figure()
    plt.plot(epoch_list, train_loss_list, "bo-", label="Train Loss")
    plt.plot(epoch_list, val_loss_list, "co-", label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Total Loss Over Epochs")
    plt.legend()
    plt.savefig(PLOT_PATH + "/training_loss.png")
    plt.close()

    plt.figure()
    plt.plot(epoch_list, train_re_list, "ro-", label="Train RE")
    plt.plot(epoch_list, val_re_list, "mo-", label="Val RE")
    plt.xlabel("Epoch")
    plt.ylabel("Reconstruction_Error")
    plt.title("Reconstruction Error Over Epochs")
    plt.legend()
    plt.savefig(PLOT_PATH + "/training_re.png")
    plt.close()

    plt.figure()
    plt.plot(epoch_list, train_kl_list, "go-", label="Train KL")
    plt.plot(epoch_list, val_kl_list, "yo-", label="Val KL")
    plt.xlabel("Epoch")
    plt.ylabel("KL_Divergence")
    plt.title("KL Divergence Over Epochs")
    plt.legend()
    plt.savefig(PLOT_PATH + "/training_kl.png")
    plt.close()

    if best_state is not None:
        vae.load_state_dict(best_state)
        torch.save(best_state, "vae_best.pt")

    test_loss, test_re, test_kl = run_epoch(test_dataloader, False, optimizer=None, beta=1.0)
    print("TEST:", test_loss, test_re, test_kl)

    # ---------------- Extra evaluation ----------------
    vae = VAE(encoder_net=encoder_net, decoder_net=decoder_net, D=9, L=8, num_vals=3).to(device)
    vae.load_state_dict(torch.load("vae_best.pt", map_location=device))
    vae.eval()

    board_pred, turn_pred = reconstruct_batch(vae, X_test.to(device))
    cell_acc, turn_acc = evaluate_reconstruction(vae, test_dataloader)
    print(f"Test Cell Accuracy: {cell_acc:.4f}, Test Turn Accuracy: {turn_acc:.4f}")

    print("\n--- Sample Reconstructions from Test Set ---\n")
    with torch.no_grad():
        x_sample = X_test[:100].to(device)
        y_cells_sample = y_cells_test[:100]
        y_turn_sample = y_turn_test[:100]

        board_pred, turn_pred = reconstruct_batch(vae, x_sample)
        invalid_recon_count = 0

        num_examples = min(100, X_test.shape[0])
        for i in range(num_examples):
            print(f"Example {i + 1}")
            print("Original Board:")
            print_board_mapped(y_cells_sample[i].cpu())

            print("Reconstructed Board:")
            print_board_mapped(board_pred[i].cpu())

            print("Original Turn:", int(y_turn_sample[i].item()))
            print("Predicted Turn:", int(turn_pred[i].item()))
            print("-" * 40)

            if not validity_board_check(board_pred[i].cpu().numpy(), int(turn_pred[i].item())):
                print("Warning: Invalid reconstructed board state detected!")
                invalid_recon_count += 1

        print(f"Total invalid reconstructed board states in 100 examples from test set: {invalid_recon_count}")

    visualize_latent_space(vae, test_dataloader)

    board1 = [1, -1, 0, 0, 1, 0, 0, 0, 0]
    turn1 = 1
    board2 = [1, -1, 0, -1, 1, 0, 0, 0, 0]
    turn2 = 0

    interpolated = interpolate_between_boards(vae, board1, turn1, board2, turn2, steps=10)
    invalid_interp = 0
    for board, turn in interpolated:
        if not validity_board_check(board, int(turn)):
            invalid_interp += 1
    print(f"Total invalid states during interpolation: {invalid_interp} out of {len(interpolated)}")

    vae.eval()
    samples = 100
    boards, turns = vae.sample(batch_size=samples, device=device)

    invalid_prior = 0
    for i in range(samples):
        board = boards[i].cpu().numpy()
        turn = int(turns[i].item())
        if not validity_board_check(board, turn):
            invalid_prior += 1
    print(f"Total invalid states in sampled prior: {invalid_prior} out of {samples}")
    visited_states_set = set(map(tuple, visited_self_tensor.numpy()))

    states_dataset_np = visited_self_tensor.numpy()

    z_dataset = build_latent_dataset(vae, visited_self_tensor)

    # latent kNN database uses all visited self-play states
    

    # Q2.4(a): choose number of samples from prior
    total_generated_samples = 1000

    # Q2.4(a)(b)
    sampled_boards, sampled_turns = sample_from_vae_prior(
        vae=vae,
        num_samples=total_generated_samples
    )

    # Q2.4(c)
    valid_generated_states, valid_count, acceptance_rate = decode_and_filter_valid_states(
        sampled_boards,
        sampled_turns
    )

    print("\n--- Question 2.4(a)(b)(c) ---")
    print(f"Total generated samples from prior: {total_generated_samples}")
    print(f"Valid decoded samples: {valid_count}")
    print(f"Acceptance rate: {acceptance_rate:.4f}")

    # Q2.4(d)
    generated_unseen_states = select_unseen_generated_states(
        valid_generated_states,
        visited_states_set
    )
    print(f"Unseen valid generated states: {len(generated_unseen_states)}")

    # Q2.4(e)
    eval_results = evaluate_generalization_2_4(
        game=game,
        vae=vae,
        Q=Q_self,
        z_dataset=z_dataset,
        states_dataset=states_dataset_np,
        generated_states=generated_unseen_states,
        max_states=200,
        k=10
    )

    # Q2.4(f)
    print_q24_summary(
        total_generated=total_generated_samples,
        valid_count=valid_count,
        acceptance_rate=acceptance_rate,
        unseen_count=len(generated_unseen_states),
        eval_results=eval_results
    )