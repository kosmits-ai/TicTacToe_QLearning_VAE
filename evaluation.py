
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from vae.model import VAE
from qlearning_tic_tac_toe import TicTacToe_N_K, preprocess, device, PLOT_PATH, encoder_net, decoder_net
from torch.utils.data import DataLoader, TensorDataset
import os

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