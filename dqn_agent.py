# dqn_agent.py
# This file is the responsibility of Person 3

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random

###################################################################
# 1. The Neural Network Architecture
###################################################################

class QNetwork(nn.Module):
    """
    The neural network that learns to approximate the Q-Value function.
    It takes a flat state vector as input and outputs 26 Q-values (one for each letter).
    """
    def __init__(self, state_dim, action_dim):
        """
        Initializes the network layers.
        
        Args:
            state_dim (int): The total size of the flattened state vector.
                             (Your team agreed this is 77)
            action_dim (int): The number of possible actions.
                              (Your team agreed this is 26)
        """
        super(QNetwork, self).__init__()
        
        # A simple network with two hidden layers
        # Input layer will be 77
        self.layer1 = nn.Linear(state_dim, 128)
        self.layer2 = nn.Linear(128, 128)
        # Output layer will be 26
        self.layer3 = nn.Linear(128, action_dim) 
    
    def forward(self, x):
        """The forward pass of the network."""
        # Pass the input state through the layers
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

###################################################################
# 2. The Agent Class
###################################################################

class DQNAgent:
    """
    The agent that manages the Q-Network, action selection, and learning.
    This class is imported and used by train.py.
    """
    def __init__(self, state_dim, action_dim, replay_buffer, hmm_models,
                 gamma=0.99, batch_size=128, lr=1e-4):
        """
        Initializes the agent.
        
        Args:
            state_dim (int): The size of the state vector (77).
            action_dim (int): The number of actions (26).
            replay_buffer (ReplayBuffer): The replay buffer (from Person 4).
            hmm_models (dict): The dictionary of trained HMMs (from Person 1).
            gamma (float): The discount factor.
            batch_size (int): The size of the batch to learn from.
            lr (float): The learning rate for the optimizer.
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.replay_buffer = replay_buffer
        self.hmm_models = hmm_models
        self.gamma = gamma
        self.batch_size = batch_size
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # --- Create the Networks ---
        # The Policy Network is the one we actively train
        self.policy_net = QNetwork(state_dim, action_dim).to(self.device)
        # The Target Network is used to stabilize learning
        self.target_net = QNetwork(state_dim, action_dim).to(self.device)
        # Copy the weights
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval() # Put target net in evaluation mode
        
        # Define the optimizer (AdamW is a good, modern default)
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=lr, amsgrad=True)

    
    def _get_hmm_probs(self, masked_word_arr, word_length):
        """
        Gets the letter probability distribution from the HMM.
        This is the "Oracle" part.
        
        Args:
            masked_word_arr (np.array): The int-encoded masked word (e.g., [26, 26, 15, 26])
            word_length (int): The length of the secret word.
            
        Returns:
            np.array: A 26-element (action_dim) array of probabilities.
        """
        # 1. Get the correct HMM for this word length
        if word_length not in self.hmm_models:
            # Fallback: return uniform probabilities if no HMM
            return np.full(self.action_dim, 1.0 / self.action_dim)
        
        model = self.hmm_models[word_length]
        
        # 2. Find the indices of the blank spots (where value == 26)
        # 26 is the agreed-upon int for a blank '_'
        blank_indices = [i for i, val in enumerate(masked_word_arr) if i < word_length and val == 26]
        
        if not blank_indices:
            # No blanks, return zeros
            return np.zeros(self.action_dim)
            
        # 3. Get emission probabilities from the HMM
        # (Make sure Person 1's HMMs save with this attribute)
        if hasattr(model, 'emissionprob_'):
            emission_probs = model.emissionprob_
        else:
            # Handle potential HMM model attribute name difference
            print(f"Warning: HMM for length {word_length} has no 'emissionprob_'. Using uniform.")
            return np.full(self.action_dim, 1.0 / self.action_dim)

        # 4. Sum the probabilities for all blank states
        total_probs = np.zeros(self.action_dim)
        for state_index in blank_indices:
            # Ensure state_index is within bounds of emission_probs
            if state_index < emission_probs.shape[0]:
                total_probs += emission_probs[state_index]
            
        # 5. Normalize the probabilities to sum to 1
        prob_sum = np.sum(total_probs)
        if prob_sum > 0:
            return total_probs / prob_sum
        else:
            # Fallback
            return np.full(self.action_dim, 1.0 / self.action_dim)

    def _state_to_tensor(self, obs, hmm_probs):
        """
        Flattens the complete state (env obs + hmm probs) into
        a single vector and converts it to a PyTorch tensor.
        The resulting vector MUST have size state_dim (77).
        """
        
        # Get components from the environment observation (from Person 2)
        masked_word_vec = obs['masked_word']   # (np.array, size 24)
        guessed_vec = obs['guessed_letters']   # (np.array, size 26)
        lives_vec = np.array([obs['lives_left']])  # (np.array, size 1)
        
        # Get HMM probabilities
        hmm_vec = hmm_probs                    # (np.array, size 26)
        
        # Concatenate all into one flat vector (size 24+26+1+26 = 77)
        flat_state = np.concatenate([
            masked_word_vec,
            guessed_vec,
            lives_vec,
            hmm_vec
        ])
        
        # Convert to a PyTorch tensor and add a batch dimension (B=1)
        return torch.FloatTensor(flat_state).unsqueeze(0).to(self.device)

    def choose_action(self, obs, word_length, epsilon):
        """
        Chooses an action using an epsilon-greedy policy.
        
        Args:
            obs (dict): The observation from the env (from Person 2).
            word_length (int): The length of the secret word.
            epsilon (float): The exploration factor.
            
        Returns:
            int: The action (0-25) to take.
        """
        
        # --- 1. Explore vs. Exploit ---
        if random.random() > epsilon:
            # --- Exploit (Choose best action) ---
            with torch.no_grad(): # No need to track gradients
                
                # 1. Get HMM probabilities
                hmm_probs = self._get_hmm_probs(obs['masked_word'], word_length)
                
                # 2. Convert state to tensor
                state_tensor = self._state_to_tensor(obs, hmm_probs)
                
                # 3. Get Q-values from the policy network
                q_values = self.policy_net(state_tensor)
                
                # 4. MASKING: Don't re-guess letters
                # Set Q-value of already-guessed letters to -infinity
                guessed_indices = np.where(obs['guessed_letters'] == 1)[0]
                if len(guessed_indices) > 0:
                    q_values[0, guessed_indices] = -float('inf')
                
                # 5. Choose the action with the highest Q-value
                action = q_values.argmax().item()
        
        else:
            # --- Explore (Choose random *valid* action) ---
            
            # Get indices of letters NOT yet guessed
            possible_actions = np.where(obs['guessed_letters'] == 0)[0]
            if len(possible_actions) > 0:
                action = random.choice(possible_actions)
            else:
                # No valid actions left (shouldn't happen if game ends)
                action = 0 # Default to 'a'
        
        return action

    def learn(self):
        """
        Samples a batch from the replay buffer and updates the network.
        This is called by train.py (Person 4).
        """
        if len(self.replay_buffer) < self.batch_size:
            return  # Not enough experiences in buffer yet
            
        # 1. Sample a batch of experiences (from Person 4's buffer)
        batch = self.replay_buffer.sample(self.batch_size)
        
        # Unzip the batch
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # 2. Convert batch components to Tensors
        # We must re-calculate HMM probs for all states in the batch
        
        state_tensors = []
        next_state_tensors = []
        
        for i in range(self.batch_size):
            # For state
            s_obs = states[i]['obs']
            s_len = states[i]['len']
            s_hmm = self._get_hmm_probs(s_obs['masked_word'], s_len)
            s_tensor = self._state_to_tensor(s_obs, s_hmm).squeeze(0) # Remove batch dim
            state_tensors.append(s_tensor)
            
            # For next_state
            ns_obs = next_states[i]['obs']
            ns_len = next_states[i]['len']
            ns_hmm = self._get_hmm_probs(ns_obs['masked_word'], ns_len)
            ns_tensor = self._state_to_tensor(ns_obs, ns_hmm).squeeze(0) # Remove batch dim
            next_state_tensors.append(ns_tensor)

        # Stack into a batch
        state_batch = torch.stack(state_tensors).to(self.device)
        next_state_batch = torch.stack(next_state_tensors).to(self.device)
        
        action_batch = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        reward_batch = torch.FloatTensor(rewards).to(self.device)
        done_batch = torch.FloatTensor(dones).to(self.device) # 1.0 if done, 0.0 if not

        # --- 3. Calculate Q-Values (Standard DQN) ---
        
        # Q(s, a) - Q-values for the actions we *actually took*
        q_values = self.policy_net(state_batch).gather(1, action_batch)
        
        # V(s') - Max Q-value for the *next* state, from target network
        with torch.no_grad():
            next_q_values = self.target_net(next_state_batch).max(1)[0]
        
        # Target Q-Value: R + γ * V(s')
        # We set V(s') to 0 if the game was 'done'
        target_q_values = reward_batch + (self.gamma * next_q_values * (1 - done_batch))
        
        # --- 4. Calculate Loss ---
        # We use Smooth L1 Loss (Huber Loss), which is more stable
        loss = F.smooth_l1_loss(q_values, target_q_values.unsqueeze(1))
        
        # --- 5. Backpropagation ---
        self.optimizer.zero_grad() # Clear old gradients
        loss.backward()            # Calculate new gradients
        
        # Clip gradients to prevent them from exploding
        for param in self.policy_net.parameters():
            if param.grad is not None:
                param.grad.data.clamp_(-1.0, 1.0)
            
        self.optimizer.step()      # Update the network's weights

    def update_target_net(self):
        """Copies weights from policy_net to target_net."""
        self.target_net.load_state_dict(self.policy_net.state_dict())