"""
Neural Network Model for Nine Men's Morris PPO Agent
"""

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple, Optional

class NineMensMorrisNet(nn.Module):
    """
    Actor-Critic Network for Nine Men's Morris
    Input: 7x24 board state
    Output: Policy (Action probabilities) and Value (State value)
    """
    def __init__(self, action_size: int = 624):
        super(NineMensMorrisNet, self).__init__()
        
        # Feature extractor
        self.conv1 = nn.Conv1d(7, 64, kernel_size=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=1)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=1)
        
        self.fc_shared = nn.Linear(256 * 24, 512)
        
        # Actor head (Policy)
        self.actor_fc = nn.Linear(512, 256)
        self.actor_out = nn.Linear(256, action_size)
        
        # Critic head (Value)
        self.critic_fc = nn.Linear(512, 256)
        self.critic_out = nn.Linear(256, 1)
        
    def forward(self, x):
        # x shape: (batch_size, 7, 24)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        x = x.view(x.size(0), -1)  
        x = F.relu(self.fc_shared(x))
        
        # Actor
        policy = F.relu(self.actor_fc(x))
        policy_logits = self.actor_out(policy)
        
        # Critic
        value = F.relu(self.critic_fc(x))
        value = self.critic_out(value)
        
        return policy_logits, value

def load_model(path: str, device: torch.device) -> NineMensMorrisNet:
    """Load a trained model"""
    model = NineMensMorrisNet()
    try:
        # Try loading as full model
        loaded = torch.load(path, map_location=device)
        if isinstance(loaded, NineMensMorrisNet):
            model = loaded
        elif isinstance(loaded, dict):
            # Check if it's a state dict or checkpoint
            if 'model_state_dict' in loaded:
                model.load_state_dict(loaded['model_state_dict'])
            else:
                model.load_state_dict(loaded)
    except Exception as e:
        print(f"Error loading model {path}: {e}")
        # Return initialized random model if fail, just to not crash
        pass
        
    model.to(device)
    model.eval()
    return model

def get_ai_move(model: NineMensMorrisNet, env, device: torch.device) -> Tuple:
    """Get the next move from the AI model"""
    state = env.get_state()
    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
    
    with torch.no_grad():
        policy_logits, _ = model(state_tensor)
        
    # Mask invalid actions
    valid_mask = torch.FloatTensor(env.get_valid_action_mask()).to(device)
    
    # Apply mask (set invalid logits to -inf)
    policy_logits = policy_logits.squeeze(0)
    masked_logits = policy_logits.clone()
    masked_logits[valid_mask == 0] = -float('inf')
    
    # Softmax to get probabilities
    probs = F.softmax(masked_logits, dim=0)
    
    # Handle numerical instability
    if torch.isnan(probs).any() or torch.isinf(probs).any() or probs.sum() == 0:
        # Fallback: choose random valid action
        valid_indices = torch.nonzero(valid_mask).squeeze()
        if valid_indices.numel() > 0:
            if valid_indices.numel() == 1:
                action_idx = valid_indices.item()
            else:
                idx = torch.randint(0, valid_indices.numel(), (1,)).item()
                action_idx = valid_indices[idx].item()
        else:
            # Should not happen if env provides valid mask, but just in case
            return ('move', 0, 0) # Dummy/Invalid
    else:
        # Sample action
        action_idx = torch.multinomial(probs, 1).item()
    
    return env.index_to_action(action_idx)

def get_ai_capture(model: NineMensMorrisNet, env, device: torch.device) -> Tuple:
    """Get capture action from AI"""
    # Simply use the same valid action masking logic but restricted to captures
    state = env.get_state()
    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
    
    with torch.no_grad():
        policy_logits, _ = model(state_tensor)
    
    # Manual mask for capture actions only
    valid_captures = env.get_valid_capture_actions()
    mask = torch.zeros(624).to(device)
    
    for action in valid_captures:
        idx = env.action_to_index(action)
        mask[idx] = 1.0
        
    policy_logits = policy_logits.squeeze(0)
    masked_logits = policy_logits.clone()
    masked_logits[mask == 0] = -float('inf')
    
    probs = F.softmax(masked_logits, dim=0)
    
    try:
        action_idx = torch.multinomial(probs, 1).item()
        return env.index_to_action(action_idx)
    except:
        # Fallback if something goes wrong shouldn't if mask is correct        
        return valid_captures[0] if valid_captures else None
