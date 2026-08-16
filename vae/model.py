import torch
import torch.nn as nn

from auxiliary.log_prob_utilities import  log_categorical_indices, log_normal_diag, log_standard_normal 

class Encoder(nn.Module):
    def __init__(self, encoder_net):
        super().__init__()
        self.encoder = encoder_net
    
    @staticmethod
    # Reparameterization trick to sample z from q(z|x) = N(mu, sigma^2). This allows backpropagation.
    def reparameterization(mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    #Calculate mu, log_var from encoder output, with clamping for numerical stability
    def encode(self, x):
        h = self.encoder(x)
        mu, log_var = torch.chunk(h, 2, dim=1)
        log_var = torch.clamp(log_var, min=-10, max=10)

        return mu, log_var
    
    #Return sampled z from q(z|x) using reparameterization trick. 
    def sample(self, x=None, mu_e=None, log_var_e=None):
        if mu_e is None and log_var_e is None:
            mu_e, log_var_e = self.encode(x)
        else:
            if mu_e is None or log_var_e is None:
                raise ValueError("Both mu_e and log_var_e must be provided if one is provided.")
        
        return self.reparameterization(mu_e, log_var_e)
    
    #Computes log q(z|x) to measure how much the encoder distribution differs from the prior. This is used in the KL divergence term of the ELBO.
    def log_prob(self, x=None, mu_e=None, log_var_e=None, z=None):
        if x is not None:
            mu_e, log_var_e = self.encode(x)
            z = self.sample(mu_e=mu_e, log_var_e=log_var_e)
        else:
            if (mu_e is None) or (log_var_e is None) or (z is None):
                raise ValueError("Need (mu_e, log_var_e, z) if x is None.")
        return log_normal_diag(z, mu_e, log_var_e)  #how likely is z under q(z|x) = N(mu_e, sigma_e^2)

class Decoder(nn.Module):
    def __init__(self, decoder_net, num_vals=3):
        super().__init__()
        self.decoder = decoder_net
        self.num_vals = num_vals

    def decode(self, z):
        # # z: (B, L)
        h = self.decoder(z)  # # (B, D*V + 1)

        
        B = h.shape[0]
        turn_logit = h[:, -1:]
        board_flat = h[:, :-1]

        D = board_flat.shape[1] // self.num_vals

        # # reshape into logits per pixel
        board_logits = board_flat.view(B, D, self.num_vals)  # # (B, D, V)

        return board_logits, turn_logit

    def sample(self, z):
        board_logits, turn_logit = self.decode(z)
        board_probs = torch.softmax(board_logits, dim=-1) #Softmax is applied to obtain categorical distributions per cell
        B, D, V = board_probs.shape
        p = board_probs.reshape(-1, V)  # # (B*D, V)
        board_sampled = torch.multinomial(p, num_samples=1).view(B, D)  # # (B, D) samples are drawn using multinomial sampling.

        turn_prob = torch.sigmoid(turn_logit)
        turn_sampled = torch.bernoulli(turn_prob) #a Bernoulli sample is drawn after applying the sigmoid function
        
        return board_sampled, turn_sampled

    def log_prob(self, x_idx,turn, z):
      
        
        board_logits, turn_logit = self.decode(z)
        weight_turn = 1.0 
        log_px_board = log_categorical_indices(x_idx, board_logits)
        
        turn_0_1 = turn.float()
   
        bce = nn.functional.binary_cross_entropy_with_logits(turn_logit, turn_0_1, reduction="none")  # (B,1)
        log_px_turn = -weight_turn*bce.squeeze(-1)  # (B,)

     
        return log_px_board + log_px_turn
        



class Prior(nn.Module):
    

    def __init__(self, L):
        super().__init__()
        self.L = L

    def sample(self, batch_size, device, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        return torch.randn((batch_size, self.L), device=device)

    def log_prob(self, z):
        return log_standard_normal(z)  # # (B, L)




class VAE(nn.Module):
   

    def __init__(self, encoder_net, decoder_net, D, L, num_vals=3):
        super().__init__()
        self.D = D
        self.L = L
        self.num_vals = num_vals

        self.encoder = Encoder(encoder_net)
        self.decoder = Decoder(decoder_net, num_vals=num_vals)
        self.prior = Prior(L=L)

    
    
    def forward(self, x_enc,y_cells, y_turn, beta, reduction="avg"):
        
        mu_e, log_var_e = self.encoder.encode(x_enc)  
        z = self.encoder.sample(mu_e=mu_e, log_var_e=log_var_e)  

        log_px = self.decoder.log_prob(y_cells, y_turn, z) 
        log_pz = self.prior.log_prob(z).sum(-1)
        log_qz = self.encoder.log_prob(mu_e=mu_e, log_var_e=log_var_e, z=z).sum(-1)  
        
        re = (-log_px)
        kl = (log_qz - log_pz)

        elbo = re + beta * kl
        if reduction == "avg":
            return elbo.mean(), re.mean() , kl.mean()  
        else:
            return elbo.sum(), re.sum(), kl.sum()  

        
    @torch.no_grad()
    def sample(self, batch_size=64, device=None):
        # # determine device from model if not provided
        if device is None:
            device = next(self.parameters()).device

        # # sample z on device
        z = self.prior.sample(batch_size, device=device, seed=42)  # # (B, L)

        # # sample x on device (returns integer-valued tensor)
        x_new, turn_new = self.decoder.sample(z)  # # (B, D)

        return x_new, turn_new