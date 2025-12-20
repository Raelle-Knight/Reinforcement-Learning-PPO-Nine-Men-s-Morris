import numpy as np

class NineMensMorrisEnv:
    """Environment untuk Nine Men's Morris"""
    
    BOARD_POSITIONS = 24
    PIECES_PER_PLAYER = 9
    
    # Definisi mills (barisan 3)
    MILLS = [
        [0,1,2], [3,4,5], [6,7,8], [9,10,11], [12,13,14], [15,16,17], [18,19,20], [21,22,23],
        [0,9,21], [3,10,18], [6,11,15], [1,4,7], [16,19,22], [8,12,17], [5,13,20], [2,14,23]
    ]
    
    # Adjacency list untuk pergerakan
    ADJACENCY = {
        0:[1,9], 1:[0,2,4], 2:[1,14], 3:[4,10], 4:[1,3,5,7], 5:[4,13],
        6:[7,11], 7:[4,6,8], 8:[7,12], 9:[0,10,21], 10:[3,9,11,18],
        11:[6,10,15], 12:[8,13,17], 13:[5,12,14,20], 14:[2,13,23],
        15:[11,16], 16:[15,17,19], 17:[12,16], 18:[10,19], 19:[16,18,20,22],
        20:[13,19], 21:[9,22], 22:[19,21,23], 23:[14,22]
    }
    
    # Action space encoding 
    ACTION_PLACEMENT_START = 0
    ACTION_MOVEMENT_START = 24
    ACTION_CAPTURE_START = 600
    ACTION_SPACE_SIZE = 624
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.board = np.zeros(self.BOARD_POSITIONS, dtype=np.int8)
        self.global_phase = 'placement'
        self.player_phase = {
            1: 'placement',
            -1: 'placement'
        }
        self.current_player = 1
        self.pieces_in_hand = {1: self.PIECES_PER_PLAYER, -1: self.PIECES_PER_PLAYER}
        self.pieces_on_board = {1: 0, -1: 0}
        self.winner = None
        self.move_count = 0
        self.last_mill_formed = False
        self.last_capture = False
        self.move_history = []
        return self.get_state()
    
    def get_state(self):
        # 7 channels: current pieces, opponent pieces, placement phase, movement phase, flying phase, pieces in hand, valid moves hint
        state = np.zeros((7, self.BOARD_POSITIONS), dtype=np.float32)
        
        # Channel 0: Current player pieces
        state[0] = (self.board == self.current_player).astype(np.float32)
        
        # Channel 1: Opponent pieces
        state[1] = (self.board == -self.current_player).astype(np.float32)
        
        # Channels 2-4 Phase encoding 
        phase = self.player_phase[self.current_player]
        if phase == 'placement':
            state[2] = np.ones(self.BOARD_POSITIONS)
        elif phase == 'movement':
            state[3] = np.ones(self.BOARD_POSITIONS)
        else:  # flying
            state[4] = np.ones(self.BOARD_POSITIONS)
        
        # Channel 5: Pieces in hand (normalized)
        state[5] = np.ones(self.BOARD_POSITIONS) * (self.pieces_in_hand[self.current_player] / self.PIECES_PER_PLAYER)
        
        # Channel 6: Empty positions
        state[6] = (self.board == 0).astype(np.float32)
        
        return state
    
    def get_valid_actions(self):
        """Returns list of valid actions with proper type annotation"""
        valid_actions = []
        
        phase = self.player_phase[self.current_player]
        
        if phase == 'placement':
            # Check if player still has pieces in hand
            if self.pieces_in_hand[self.current_player] > 0:
                for pos in range(self.BOARD_POSITIONS):
                    if self.board[pos] == 0:
                        valid_actions.append(('place', None, pos))
        
        elif phase == 'movement':
            for from_pos in range(self.BOARD_POSITIONS):
                if self.board[from_pos] == self.current_player:
                    for to_pos in self.ADJACENCY[from_pos]:
                        if self.board[to_pos] == 0:
                            valid_actions.append(('move', from_pos, to_pos))
        
        elif phase == 'flying':
            for from_pos in range(self.BOARD_POSITIONS):
                if self.board[from_pos] == self.current_player:
                    for to_pos in range(self.BOARD_POSITIONS):
                        if self.board[to_pos] == 0:
                            valid_actions.append(('move', from_pos, to_pos))
        
        return valid_actions
    
    def get_valid_capture_actions(self):
        """Get valid capture actions when a mill is formed"""
        valid_captures = []
        opponent = -self.current_player
        for pos in range(self.BOARD_POSITIONS):
            if self.board[pos] == opponent:
                if not self._is_in_mill(pos, opponent) or self._all_in_mills(opponent):
                    valid_captures.append(('capture', pos, None))
        return valid_captures
    
    def get_valid_action_mask(self):
        """PPO CRITICAL: Return mask for valid actions"""
        mask = np.zeros(self.ACTION_SPACE_SIZE, dtype=np.float32)
        valid_actions = self.get_valid_actions()
        for action in valid_actions:
            idx = self.action_to_index(action)
            mask[idx] = 1.0
        return mask
    
    def action_to_index(self, action):
        """Convert action tuple to index"""
        action_type, from_pos, to_pos = action
        
        if action_type == 'place':
            return self.ACTION_PLACEMENT_START + to_pos
        elif action_type == 'move':
            return self.ACTION_MOVEMENT_START + from_pos * self.BOARD_POSITIONS + to_pos
        elif action_type == 'capture':
            return self.ACTION_CAPTURE_START + from_pos
        else:
            raise ValueError(f"Unknown action type: {action_type}")
    
    def index_to_action(self, index):
        """Convert index back to action tuple"""
        if index < self.ACTION_MOVEMENT_START:
            pos = index - self.ACTION_PLACEMENT_START
            return ('place', None, pos)
        elif index < self.ACTION_CAPTURE_START:
            offset = index - self.ACTION_MOVEMENT_START
            from_pos = offset // self.BOARD_POSITIONS
            to_pos = offset % self.BOARD_POSITIONS
            return ('move', from_pos, to_pos)
        else:
            pos = index - self.ACTION_CAPTURE_START
            return ('capture', pos, None)
    
    def step(self, action):
        """Execute action and return (state, reward, done, info)"""
        action_type, from_pos, to_pos = action
        
        # SIMPAN ACTING PLAYER SEBELUM ACTION
        acting_player = self.current_player
        
        # Reset event flags
        self.last_mill_formed = False
        self.last_capture = False
        
        reward = -0.0035  
        
        # Track move for history
        move_description = ""
        
        if action_type == 'capture':
            # Execute capture
            self.board[from_pos] = 0
            self.pieces_on_board[-self.current_player] -= 1
            self.current_player = -self.current_player
            self.move_count += 1
            self.last_capture = True
            reward += 0.05
            move_description = f"Player {acting_player} captured piece at position {from_pos}"
            
        elif action_type == 'place':
            # Place piece
            self.board[to_pos] = self.current_player
            self.pieces_in_hand[self.current_player] -= 1
            self.pieces_on_board[self.current_player] += 1
            self.move_count += 1
            move_description = f"Player {acting_player} placed piece at position {to_pos}"
            
            # Check if mill formed
            if self._is_in_mill(to_pos, self.current_player):
                self.last_mill_formed = True
                reward += 0.1
                move_description += " (Mill formed!)"
                captures = self.get_valid_capture_actions()
                if len(captures) > 0:
                    self.move_history.append(move_description)
                    return self.get_state(), reward, False, {'needs_capture': True, 'formed_mill': True}
            
            # Switch player if no mill
            self.current_player = -self.current_player
                
        elif action_type == 'move':
            # Move piece
            self.board[from_pos] = 0
            self.board[to_pos] = self.current_player
            self.move_count += 1
            move_description = f"Player {acting_player} moved piece from {from_pos} to {to_pos}"
            
            # Check if mill formed
            if self._is_in_mill(to_pos, self.current_player):
                self.last_mill_formed = True
                reward += 0.1
                move_description += " (Mill formed!)"
                captures = self.get_valid_capture_actions()
                if len(captures) > 0:
                    self.move_history.append(move_description)
                    return self.get_state(), reward, False, {'needs_capture': True, 'formed_mill': True}
            
            # Switch player
            self.current_player = -self.current_player
        
        # Add move to history
        self.move_history.append(move_description)
        
        # Check placement -> movement phase transition
        if self.global_phase == 'placement':
            if self.pieces_in_hand[1] == 0 and self.pieces_in_hand[-1] == 0:
                self.global_phase = 'movement'
                self.player_phase[1] = 'movement'
                self.player_phase[-1] = 'movement'

        # Check for flying phase PER PLAYER
        if self.global_phase == 'movement':
            for p in [1, -1]:
                if self.pieces_on_board[p] == 3:
                    self.player_phase[p] = 'flying'
        
        # Check terminal conditions
        done = False
        
        # Win by reducing opponent to < 3 pieces
        if self.global_phase != 'placement':
            if self.pieces_on_board[-acting_player] < 3:
                done = True
                reward = 1.5
                self.winner = acting_player
        
        # Win by blocking opponent
        if not done:
            next_player = self.current_player
            phase = self.player_phase[next_player]
            
            has_move = False
            
            if phase == 'placement':
                has_move = np.any(self.board == 0) and self.pieces_in_hand[next_player] > 0
            elif phase == 'movement':
                for from_pos in range(self.BOARD_POSITIONS):
                    if self.board[from_pos] == next_player:
                        for to_pos in self.ADJACENCY[from_pos]:
                            if self.board[to_pos] == 0:
                                has_move = True
                                break
                        if has_move:
                            break
            elif phase == 'flying':
                has_move = np.any(self.board == 0)
            
            if not has_move:
                done = True
                reward = 1.5
                self.winner = acting_player
        
        # Draw by move limit
        if self.move_count > 200:
            done = True
            reward = 0.0
            self.winner = 0
        
        info = {
            'formed_mill': self.last_mill_formed,
            'capture': self.last_capture
        }
        
        return self.get_state(), reward, done, info
    
    def _is_in_mill(self, pos, player):
        for mill in self.MILLS:
            if pos in mill:
                if all(self.board[p] == player for p in mill):
                    return True
        return False
    
    def _all_in_mills(self, player):
        for pos in range(self.BOARD_POSITIONS):
            if self.board[pos] == player:
                if not self._is_in_mill(pos, player):
                    return False
        return True
    
    def clone(self):
        new_env = NineMensMorrisEnv()
        new_env.board = self.board.copy()
        new_env.global_phase = self.global_phase
        new_env.player_phase = self.player_phase.copy()
        new_env.current_player = self.current_player
        new_env.pieces_in_hand = self.pieces_in_hand.copy()
        new_env.pieces_on_board = self.pieces_on_board.copy()
        new_env.winner = self.winner
        new_env.move_count = self.move_count
        new_env.move_history = self.move_history.copy()
        return new_env
