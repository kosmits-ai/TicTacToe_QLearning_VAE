
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from vae.model import VAE
from qlearning_tic_tac_toe import TicTacToe_N_K, preprocess, device, PLOT_PATH, encoder_net, decoder_net
from torch.utils.data import DataLoader, TensorDataset
import os
import pickle
import random

if not os.path.exists(PLOT_PATH):
    os.makedirs(PLOT_PATH)

def convert_mapped_to_original(board_mapped):
    """
    mapped for VAE:
        0 = X
        1 = O
        2 = empty
    original:
        1 = X
       -1 = O
        0 = empty
    """
    board_orig = []
    for v in board_mapped:
        if v == 0:      # X
            board_orig.append(1)
        elif v == 1:    # O
            board_orig.append(-1)
        else:           # empty
            board_orig.append(0)
    return board_orig

def reconstruct_batch(vae, x_enc):
    vae.eval()
    with torch.no_grad():
        mu, logvar = vae.encoder.encode(x_enc)
        z = mu # we use mean for reconstruction because sampling would add noise and we want to evaluate the mean of posterior and metrics become more stable.
        board_logits, turn_logit = vae.decoder.decode(z)
        board_pred = board_logits.argmax(dim=-1) # use argmax to get discrete cell values
        turn_pred = (torch.sigmoid(turn_logit) > 0.5).float() # threshold at 0.5 to get binary turn prediction
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

def print_board(board):
    symbols = {0: "X", 1: "O", 2: "."}  # <-- correct for board_mapped after preprocess
    for i in range(3):
        row = board[i*3:(i+1)*3]
        print(" ".join(symbols[int(c)] for c in row))
    print()


def visualize_latent_space(vae, loader):
    vae.eval()

    all_mu = []
    all_turns = []
    all_num_filled = []
    all_winners = []  # 0=None, 1=X wins, 2=O wins

    game = TicTacToe_N_K(3, 3)  

    with torch.no_grad():
        for x_enc, y_cells, y_turn in loader:  # use loader
            x_enc = x_enc.to(device)

            mu, logvar = vae.encoder.encode(x_enc)
            all_mu.append(mu.cpu())

            all_turns.append(y_turn.cpu())

            num_filled = (y_cells != 2).sum(dim=1)  
            all_num_filled.append(num_filled.cpu())

            # winner labels from ground-truth boards
            yc = y_cells.cpu().numpy()
            winners_batch = []
            for i in range(yc.shape[0]):
                board_orig = convert_mapped_to_original(yc[i])
                w = game.check_winner(board_orig)  # 1 (X), -1 (O), 0 (none)
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
    print(dict(zip(unique, counts)))

    # Plot 1: Turn
    plt.figure(figsize=(6,6))
    plt.figure(figsize=(6,6))

    for turn_value, color, label in [(0, 'blue', 'O turn'),
                                    (1, 'red', 'X turn')]:
        mask = (all_turns == turn_value)
        plt.scatter(z_2d[mask,0], z_2d[mask,1],
                    c=color, label=label, alpha=0.6)

    plt.legend()
    plt.title("PCA of Latent Space Colored by Turn")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.savefig(PLOT_PATH + '/latent_space_turn.png')
    plt.close()

    # Plot 2: Game progress (number of filled cells)

    plt.figure(figsize=(6,6))

    unique_progress = np.unique(all_num_filled)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_progress)))

    for prog_value, color in zip(unique_progress, colors):
        mask = (all_num_filled == prog_value)
        plt.scatter(
            z_2d[mask, 0],
            z_2d[mask, 1],
            color=color,
            label=f"{prog_value} filled",
            alpha=0.6
        )

    plt.legend(title="Game Progress", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title("PCA of Latent Space Colored by Game Progress")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig(PLOT_PATH + '/latent_space_progress.png')
    plt.close()
    

    # Plot 3: Winner
    plt.figure(figsize=(6,6))

    winner_labels = {
        0: ('gray', 'No Winner'),
        1: ('green', 'X Wins'),
        2: ('orange', 'O Wins')
    }

    for w, (color, label) in winner_labels.items():
        mask = (all_winners == w)
        if mask.sum() > 0:
            plt.scatter(z_2d[mask,0], z_2d[mask,1],
                        c=color, label=label, alpha=0.6)

    plt.legend()
    plt.title("PCA of Latent Space Colored by Winner")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.savefig(PLOT_PATH + '/latent_space_winner.png')
    plt.close()

def encode_single_state(board, turn, device): # similar to preprocess but for single state 
    """
    board: list/np array length 9 with values in {-1,0,1} (your game format)
    turn:  -1 or +1 (game turn), OR 0/1 if you already mapped it
    returns: x_enc float tensor shape (1, 28)
    """
    b = torch.tensor(board, dtype=torch.int64, device=device).unsqueeze(0)  # (1,9)

    # same mapping as preprocess()
    b_mapped = b.clone()
    b_mapped[b == 1]  = 0
    b_mapped[b == -1] = 1
    b_mapped[b == 0]  = 2

    b_oh = torch.nn.functional.one_hot(b_mapped, num_classes=3).float()  # (1,9,3)
    b_flat = b_oh.view(1, -1)  # (1,27)

    # turn mapping
    if turn in (-1, 1):
        t01 = torch.tensor([[(turn + 1) / 2]], dtype=torch.float32, device=device)  # (1,1)
    else:
        # already 0/1
        t01 = torch.tensor([[turn]], dtype=torch.float32, device=device)

    x_enc = torch.cat([b_flat, t01], dim=1)  # (1,28)
    return x_enc

def interpolate_between_boards(vae, board1, turn1, board2, turn2, steps=10):
    vae.eval()
    with torch.no_grad():
        # Build x_enc the same way as training 
        x1 = encode_single_state(board1, turn1, device)  # (1, 28) float
        x2 = encode_single_state(board2, turn2, device)  # (1, 28) float

        # Your VAE has: vae.encoder.encode(...)
        mu1, _ = vae.encoder.encode(x1)
        mu2, _ = vae.encoder.encode(x2)

        interpolated_boards = []
        for alpha in np.linspace(0, 1, steps):
            z_interp = (1 - alpha) * mu1 + alpha * mu2

            board_logits, turn_logit = vae.decoder.decode(z_interp)
            board_pred = board_logits.argmax(dim=-1).squeeze(0).cpu().numpy()  # (9,)
            turn_pred = (torch.sigmoid(turn_logit) > 0.5).float().item()

            interpolated_boards.append((board_pred, turn_pred))

    return interpolated_boards

def winner(board, player):
    b = board.reshape(3, 3)
    lines = []

    # rows + cols
    lines.extend(b) # add all rows
    lines.extend(b.T) # add all columns by transposing and adding rows again

    # diagonals
    lines.append([b[0,0], b[1,1], b[2,2]]) # main diagonal
    lines.append([b[0,2], b[1,1], b[2,0]]) # anti-diagonal

    return any(all(cell == player for cell in line) for line in lines)

def validity_board_check(board, turn):
    cells_X = (board == 0).sum()
    cells_O = (board == 1).sum()

    # move parity
    if abs(cells_X - cells_O) > 1:
        return False

    # allow either starter
    if turn == 1:  # X to play
        if not (cells_X == cells_O or cells_O == cells_X + 1):
            return False
    else:  # O to play
        if not (cells_X == cells_O or cells_X == cells_O + 1):
            return False

    X_win = winner(board, 0)
    O_win = winner(board, 1)

    if X_win and O_win:
        return False

    # winner implies last mover has one extra move
    if X_win and cells_X != cells_O + 1:
        return False
    if O_win and cells_O != cells_X + 1:
        return False

    return True


# -------------------------------------------------------------------------
# Q2.4 helpers
# -------------------------------------------------------------------------

def is_safe_action(game, board, turn, action):
    """Returns True if playing 'action' leaves no immediate winning reply for the opponent."""
    next_board, _, _, done = game.step(board.copy(), turn, action)
    if done:
        return True  # game ended — agent won or draw
    opp_turn = -turn
    for opp_action in game.get_legal_actions(next_board):
        opp_board, _, _, opp_done = game.step(next_board.copy(), opp_turn, opp_action)
        if opp_done and game.check_winner(opp_board) == opp_turn:
            return False
    return True


def generalization_stress_test(vae, Q, game, train_tensor, n_samples=50000, temperature=0.8, K=10):
    """Q2.4 Generalization Stress Test: Baseline (random) vs VAE-based KNN on Q-table-unseen states."""
    vae.eval()

    # KNN reference built exclusively from Q-table keys so every neighbour has Q-values
    q_keys = list(Q.keys())  # tuples: (b0..b8, turn) in {-1,0,1} / {-1,1}
    ref_tensor = torch.tensor(q_keys, dtype=torch.int64)
    X_knn, _, _ = preprocess(ref_tensor)
    with torch.no_grad():
        mu_knn, _ = vae.encoder.encode(X_knn.to(device))
    mu_knn_np = mu_knn.cpu().numpy()  # (N_q, L)

    # 2.4a: Sample z ~ N(0, I)
    z = torch.randn(n_samples, vae.L, device=device)

    # 2.4b: Decode with temperature sampling — argmax collapses to very few unique boards;
    # temperature < 1 keeps outputs near-modal while adding enough diversity to find unseen states
    with torch.no_grad():
        board_logits, turn_logit = vae.decoder.decode(z)
        boards_decoded = torch.distributions.Categorical(
            logits=board_logits / temperature
        ).sample().cpu().numpy()                                                            # (N, 9) in {0,1,2}
        turns_decoded  = (torch.sigmoid(turn_logit) > 0.5).float().cpu().numpy().squeeze(-1)  # (N,) in {0,1}

    # 2.4c: Validity filtering — keep only non-terminal, legal boards
    valid = [(boards_decoded[i], int(turns_decoded[i]))
             for i in range(n_samples)
             if validity_board_check(boards_decoded[i], int(turns_decoded[i]))
             and not winner(boards_decoded[i], 0)
             and not winner(boards_decoded[i], 1)]
    n_valid = len(valid)
    acceptance_rate = n_valid / n_samples

    # Build set of keys that were in the VAE training set
    train_keys = set()
    for row in train_tensor.numpy():
        train_keys.add(game.encode_state(row[:9].tolist(), int(row[9])))

    # Diagnostics: unique valid keys and Q-table coverage
    unique_valid_keys = set()
    for board_mapped, turn_01 in valid:
        board_orig = convert_mapped_to_original(board_mapped.tolist())
        turn_orig  = 1 if turn_01 == 1 else -1
        unique_valid_keys.add(game.encode_state(board_orig, turn_orig))
    print(f"{'Q-table states':<35} {len(Q)}")
    print(f"{'VAE training states':<35} {len(train_keys)}")
    print(f"{'Unique valid generated states':<35} {len(unique_valid_keys)}")
    print(f"{'Already in Q-table':<35} {len(unique_valid_keys & set(Q.keys()))}")
    print(f"{'Outside Q-table':<35} {len(unique_valid_keys - set(Q.keys()))}")
    print(f"{'Already in VAE train set':<35} {len(unique_valid_keys & train_keys)}")
    print(f"{'Outside VAE train set':<35} {len(unique_valid_keys - train_keys)}")

    # 2.4d: Unseen = no Q-row (assignment definition: not seen in Q-learning training); deduplicate
    seen_keys = set()
    unseen = []
    for board_mapped, turn_01 in valid:
        board_orig = convert_mapped_to_original(board_mapped.tolist())
        turn_orig  = 1 if turn_01 == 1 else -1
        key = game.encode_state(board_orig, turn_orig)
        if key not in Q and key not in seen_keys:
            seen_keys.add(key)
            unseen.append((board_orig, turn_orig))
    n_unseen = len(unseen)

    # 2.4f header: always print, even when n_unseen == 0
    sep = "=" * 55
    print(f"\n{sep}")
    print("Q2.4 Generalization Stress Test — Summary")
    print(sep)
    print(f"{'Total generated samples':<35} {n_samples}")
    print(f"{'Valid samples':<35} {n_valid}  ({acceptance_rate:.1%})")
    print(f"{'Unseen (unique, no Q-row)':<35} {n_unseen}")
    if n_unseen == 0:
        print("No unseen boards found — Q-table covers the full reachable state space for 3×3.")
        print(sep)
        return

    # 2.4e: Decision test
    baseline_safe   = 0
    vae_safe        = 0
    evaluated_count = 0  # full-board draws have no legal moves; excluded from the rate

    for board_orig, turn_orig in unseen:
        legal = game.get_legal_actions(board_orig)
        if not legal:
            continue
        evaluated_count += 1

        # Baseline: always random — no Q-row exists by construction
        baseline_action = random.choice(legal)

        # VAE-based: weighted KNN; per action, skip neighbours where it is absent and renormalise
        x_enc = encode_single_state(board_orig, turn_orig, device)
        with torch.no_grad():
            mu_q, _ = vae.encoder.encode(x_enc)
        mu_q_np  = mu_q.cpu().numpy()
        dists    = np.linalg.norm(mu_knn_np - mu_q_np, axis=1)
        nn_idx   = np.argsort(dists)[:K]
        nn_dists = dists[nn_idx]
        sigma    = np.median(nn_dists) + 1e-8
        raw_w    = np.exp(-nn_dists ** 2 / sigma ** 2)

        Q_tilde = {}
        for a in legal:
            w_sum = 0.0
            q_sum = 0.0
            for j, idx in enumerate(nn_idx):
                q_val = Q.get(q_keys[idx], {}).get(a)
                if q_val is not None:  # neighbour has action a; contributes to the estimate
                    q_sum += raw_w[j] * q_val
                    w_sum += raw_w[j]
            Q_tilde[a] = q_sum / w_sum if w_sum > 0 else 0.0
        vae_action = max(Q_tilde, key=Q_tilde.get)

        baseline_safe += is_safe_action(game, board_orig, turn_orig, baseline_action)
        vae_safe      += is_safe_action(game, board_orig, turn_orig, vae_action)

    if evaluated_count == 0:
        print("All unseen boards were full-board draws — nothing to evaluate.")
        return

    baseline_rate = baseline_safe / evaluated_count
    vae_rate      = vae_safe      / evaluated_count

    # 2.4f: remaining stats (header already printed above)
    print(f"{'Evaluated (has legal moves)':<35} {evaluated_count}")
    print(f"{'Safe-action rate  Baseline':<35} {baseline_rate:.2%}")
    print(f"{'Safe-action rate  VAE-based KNN':<35} {vae_rate:.2%}")
    print(sep)
    if vae_rate > baseline_rate:
        print("=> VAE latent-space transfer improves generalisation on unseen states.")
    else:
        print("=> No clear generalisation gain for this sample (may vary with n_samples).")


# ------------------------------------------------------------------------- * -----------------------------------------------------------------
# Evaluation code to run after training is complete
# Loads best model, evaluates reconstruction accuracy, and visualizes latent space
# -------------------------------------------------------------------------- * -----------------------------------------------------------------

train_state_tensor = torch.load("train_states.pt")
val_state_tensor = torch.load("val_states.pt")
test_states_tensor = torch.load("test_states.pt")
X_test, y_cells_test, y_turn_test = preprocess(test_states_tensor)
test_dataset = TensorDataset(X_test, y_cells_test, y_turn_test)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

vae = VAE(encoder_net=encoder_net, decoder_net=decoder_net, D=9, L=8, num_vals=3).to(device)
vae.load_state_dict(torch.load("vae_best.pt", map_location=device))
vae.eval()

board_pred, turn_pred = reconstruct_batch(vae, X_test.to(device))
cell_acc, turn_acc = evaluate_reconstruction(vae, test_dataloader)

print(f"Test Cell Accuracy: {cell_acc:.4f}, Test Turn Accuracy: {turn_acc:.4f}")

print("\n--- Sample Reconstructions from Test Set ---\n")

vae.eval()
with torch.no_grad():
    # Take first 100 examples from test set
    x_sample = X_test[:100].to(device)
    y_cells_sample = y_cells_test[:100]
    y_turn_sample = y_turn_test[:100]

    board_pred, turn_pred = reconstruct_batch(vae, x_sample)
    flag = 0
    for i in range(100):
        print(f"Example {i+1}")
        print("Original Board:")
        print_board(y_cells_sample[i].cpu())

        print("Reconstructed Board:")
        print_board(board_pred[i].cpu())

        print("Original Turn:", int(y_turn_sample[i].item()))
        print("Predicted Turn:", int(turn_pred[i].item()))
        print("-" * 40)
        if not validity_board_check(board_pred[i].cpu().numpy(), turn_pred[i].item()):
            print("Warning: Invalid reconstructed board state detected!")
            flag += 1

print(f"Total invalid reconstructed board states in 100 examples from test set: {flag}")

visualize_latent_space(vae, test_dataloader)

# interpolation between two boards
board1 = [1, -1, 0, 0, 1, 0, 0, 0, 0]  
turn1 = 0  
board2 = [1, -1, 0, -1, 1, 0, 0, 0, 0]  
turn2 = 0
flag = 0
interpolated = interpolate_between_boards(vae, board1, turn1, board2, turn2, steps=10)
print("\n--- Interpolation Between Two Boards ---\n")
flag = 0
for i, (board, turn) in enumerate(interpolated):
    # print(f"Step {i+1}")
    # print_board(board)
    # print("Predicted Turn:", int(turn))
    # print("-" * 40)
    if not validity_board_check(board, turn):
        #print("Warning: Invalid board state detected during interpolation!")
        flag += 1

print(f"Total invalid states during interpolation: {flag} out of {len(interpolated)}")

# Plot interpolated boards (Q2.3c)
_sym = {0: "X", 1: "O", 2: "."}
_n_steps = len(interpolated)
_cols = min(_n_steps, 5)
_rows = (_n_steps + _cols - 1) // _cols
fig, axes = plt.subplots(_rows, _cols, figsize=(3 * _cols, 4 * _rows), squeeze=False)
axes = axes.flatten()
for _idx, (_board_i, _turn_i) in enumerate(interpolated):
    _ax = axes[_idx]
    _grid = np.array(_board_i).reshape(3, 3)
    _text = "\n".join(" ".join(_sym[int(_c)] for _c in _row) for _row in _grid)
    _is_valid = validity_board_check(_board_i, _turn_i)
    _t_val = _idx / max(_n_steps - 1, 1)
    _ax.text(0.5, 0.5, _text, ha='center', va='center', fontsize=14, family='monospace')
    _ax.set_title(f"t={_t_val:.2f}  {'valid' if _is_valid else 'INVALID'}",
                  fontsize=9, color='black' if _is_valid else 'red')
    _ax.axis('off')
for _ax in axes[_n_steps:]:
    _ax.axis('off')
plt.suptitle("Latent Space Interpolation Between Two Boards")
plt.tight_layout()
plt.savefig(PLOT_PATH + '/interpolation_boards.png')
plt.close()
print("Interpolation boards plot saved to:", PLOT_PATH + '/interpolation_boards.png')

# sampling from prio distribution

vae.eval()
samples = 100
boards, turns = vae.sample(batch_size=samples, device=device)

print("\n--- Sampled Boards from Prior ---\n")
flag= 0

for i in range(samples):
    board = boards[i].cpu().numpy()
    turn  = int(turns[i].item())

    # print(f"Sample {i+1}")
    # print_board(board)
    # print("Predicted Turn:", turn)
    # print("-" * 40)
    if not validity_board_check(board, turn):
        #print("Warning: Invalid board state detected in sampled prior!")
        flag += 1

print(f"Total invalid states in sampled prior: {flag} out of {samples}")

# Plot sampled boards from prior (Q2.3d)
_sym = {0: "X", 1: "O", 2: "."}
_n_show = min(20, samples)
fig, axes = plt.subplots(4, 5, figsize=(12, 10), squeeze=False)
axes = axes.flatten()
for _i in range(_n_show):
    _board_i  = boards[_i].cpu().numpy()
    _turn_i   = int(turns[_i].item())
    _grid     = np.array(_board_i).reshape(3, 3)
    _text     = "\n".join(" ".join(_sym[int(_c)] for _c in _row) for _row in _grid)
    _is_valid = validity_board_check(_board_i, _turn_i)
    axes[_i].text(0.5, 0.5, _text, ha='center', va='center',
                  fontsize=13, family='monospace')
    axes[_i].set_title('valid' if _is_valid else 'INVALID', fontsize=8,
                       color='green' if _is_valid else 'red')
    axes[_i].axis('off')
for _ax in axes[_n_show:]:
    _ax.axis('off')
plt.suptitle("Boards Sampled from the Prior  p(z) = N(0, I)")
plt.tight_layout()
plt.savefig(PLOT_PATH + '/sampled_boards.png')
plt.close()
print("Sampled boards plot saved to:", PLOT_PATH + '/sampled_boards.png')

# -------------------------------------------------------------------------
# Q2.4 Generalization Stress Test
# -------------------------------------------------------------------------
if os.path.exists("Q_self.pkl"):
    with open("Q_self.pkl", "rb") as _f:
        Q_self_loaded = pickle.load(_f)
    stress_game = TicTacToe_N_K(3, 3)
    generalization_stress_test(vae, Q_self_loaded, stress_game, train_state_tensor, n_samples=50000, K=10)
else:
    print("Q_self.pkl not found — re-run qlearning_tic_tac_toe.py first.")