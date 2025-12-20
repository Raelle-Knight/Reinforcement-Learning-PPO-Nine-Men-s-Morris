"""
Nine Men's Morris Streamlit Application
Strict AI vs AI Mode (Model 1 vs Model 2)
"""

import os
import time
import torch
import numpy as np
import streamlit as st

from datetime import datetime

# Import game modules
import database as db

from game import NineMensMorrisEnv
from board import draw_board, BOARD_SIZE
from model import NineMensMorrisNet, load_model, get_ai_move, get_ai_capture

# Page configuration
st.set_page_config(
    page_title="Nine Men's Morris Reinforcement Learning",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for dark theme and styling 
st.markdown("""
<style>
    /* Hide sidebar */
    [data-testid="stSidebar"] {
        display: none;
    }
    
    /* Dark theme */
    .stApp {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    }
    
    /* Title styling */
    .main-title {
        text-align: center;
        font-size: 1.6rem;
        font-weight: bold;
        color: #e94560;
        padding: 1rem;
        background: linear-gradient(90deg, rgba(233, 69, 96, 0.1), rgba(233, 69, 96, 0.2), rgba(233, 69, 96, 0.1));
        border-radius: 10px;
        margin-bottom: 0.5rem;
        text-shadow: 0 0 20px rgba(233, 69, 96, 0.5);
    }
    
    .subtitle {
        text-align: center;
        font-size: 1rem;
        color: #ffffff;
        background: rgba(0, 0, 0, 0.3);
        padding: 0.5rem;
        border-radius: 5px;
        margin-bottom: 1rem;
    }
    
    /* Player colors - high contrast */
    .player-blue {
        color: #00ccff;
        font-weight: bold;
        text-shadow: 0 0 10px rgba(0, 204, 255, 0.5);
    }
    
    .player-red {
        color: #ff6b6b;
        font-weight: bold;
        text-shadow: 0 0 10px rgba(255, 107, 107, 0.5);
    }
    
    /* Piece counter boxes */
    .piece-counter {
        background: rgba(0, 0, 0, 0.6);
        border-radius: 15px;
        padding: 1.5rem;
        text-align: center;
        border: 2px solid;
        margin: 0.5rem;
        min-height: 380px; /* Fixed height to prevent resizing */
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
    }
    
    .piece-counter-blue {
        border-color: #00ccff;
        box-shadow: 0 0 20px rgba(0, 204, 255, 0.3);
    }
    
    .piece-counter-red {
        border-color: #ff6b6b;
        box-shadow: 0 0 20px rgba(255, 107, 107, 0.3);
    }
    
    .piece-count-number {
        font-size: 3rem;
        font-weight: bold;
        margin: 0.5rem 0;
    }
    
    .piece-count-label {
        font-size: 0.9rem;
        color: #ffffff;
        opacity: 0.8;
    }
    
    /* Legend box */
    .legend-container {
        background: rgba(0, 0, 0, 0.5);
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .legend-item {
        padding: 0.3rem 0;
        color: #ffffff;
    }
    
    /* Log container */
    .log-container {
        background: rgba(0, 0, 0, 0.5);
        border-radius: 10px;
        padding: 1rem;
        font-family: monospace;
        font-size: 0.85rem;
        max-height: 300px;
        overflow-y: auto;
        border: 1px solid rgba(255, 255, 255, 0.2);
        color: #ffffff;
    }
    
    .log-entry {
        padding: 3px 0;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        color: #ffffff;
    }
    
    .log-blue {
        color: #00ccff;
    }
    
    .log-red {
        color: #ff6b6b;
    }
    
    .log-mill {
        color: #ffd700;
        font-weight: bold;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #e94560, #c23a51);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 20px rgba(233, 69, 96, 0.4);
    }
    
    /* Winner announcement */
    .winner-blue {
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        color: #00ccff;
        padding: 1rem;
        background: rgba(0, 204, 255, 0.2);
        border-radius: 10px;
        border: 2px solid #00ccff;
    }
    
    .winner-red {
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        color: #ff6b6b;
        padding: 1rem;
        background: rgba(255, 107, 107, 0.2);
        border-radius: 10px;
        border: 2px solid #ff6b6b;
    }
    
    .winner-draw {
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        color: #ffd700;
        padding: 1rem;
        background: rgba(255, 215, 0, 0.2);
        border-radius: 10px;
        border: 2px solid #ffd700;
    }
    
    /* Game info */
    .game-info {
        background: rgba(0, 0, 0, 0.4);
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        color: #ffffff;
        margin: 0.5rem 0;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    /* Status text */
    .status-text {
        color: #ffffff;
        font-size: 1.1rem;
        text-align: center;
        padding: 0.5rem;
        background: rgba(0, 0, 0, 0.3);
        border-radius: 5px;
        margin: 0.5rem 0;
        min-height: 50px; /* Fixed height to prevent jumping */
        display: flex;
        justify-content: center;
        align-items: center;
    }
    
    /* Link icon styling */
    [data-testid="stMarkdownContainer"] a {
        color: white !important;
        fill: white !important;
    }
    
    /* Statistics styling */
    .stats-container {
        display: flex;
        justify-content: center;
        gap: 2rem;
        margin-top: 1rem;
    }
    
    .stat-box {
        background: rgba(0, 0, 0, 0.4);
        border-radius: 10px;
        padding: 1rem 2rem;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .stat-label {
        color: #ffffff;
        font-size: 0.9rem;
        opacity: 0.8;
    }
    
    .stat-value {
        color: #ffffff;
        font-size: 1.8rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


# Initialize session state
def init_session_state():
    if 'env' not in st.session_state:
        st.session_state.env = NineMensMorrisEnv()
    if 'game_started' not in st.session_state:
        st.session_state.game_started = False
    if 'game_over' not in st.session_state:
        st.session_state.game_over = False
    if 'game_id' not in st.session_state:
        st.session_state.game_id = None
    if 'move_log' not in st.session_state:
        st.session_state.move_log = []
    if 'move_count' not in st.session_state:
        st.session_state.move_count = 0
    if 'auto_play' not in st.session_state:
        st.session_state.auto_play = False
    if 'model1' not in st.session_state:
        st.session_state.model1 = None
    if 'model2' not in st.session_state:
        st.session_state.model2 = None
    if 'device' not in st.session_state:
        st.session_state.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if 'last_move' not in st.session_state:
        st.session_state.last_move = None
    if 'models_loaded' not in st.session_state:
        st.session_state.models_loaded = False


def load_models():
    """Load both AI models"""
    if st.session_state.models_loaded:
        return True
    
    model_dir = os.path.dirname(__file__)
    model1_path = os.path.join(model_dir, "final_ppo_model_1.pt")
    model2_path = os.path.join(model_dir, "final_ppo_model_2.pt")
    
    if not os.path.exists(model1_path) or not os.path.exists(model2_path):
        model1_path_alt = os.path.join(model_dir, "Model 1.pt")
        model2_path_alt = os.path.join(model_dir, "Model 2.pt")
        
        if os.path.exists(model1_path_alt) and os.path.exists(model2_path_alt):
             model1_path = model1_path_alt
             model2_path = model2_path_alt
        else:
             st.error("❌ Model files not found! Please ensure models are in the directory.")
             return False
    
    try:
        with st.spinner("🔄 Loading AI models..."):
            st.session_state.model1 = load_model(model1_path, st.session_state.device)
            st.session_state.model2 = load_model(model2_path, st.session_state.device)
            st.session_state.models_loaded = True
        return True
    except Exception as e:
        st.error(f"❌ Error loading models: {e}")
        return False


def start_game():
    st.session_state.env = NineMensMorrisEnv()
    st.session_state.game_started = True
    st.session_state.game_over = False
    st.session_state.move_log = []
    st.session_state.move_count = 0
    st.session_state.auto_play = True
    st.session_state.last_move = None
    
    st.session_state.game_id = db.log_game_start(
        mode="AI vs AI",
        model1_name="Model 1 (Biru)",
        model2_name="Model 2 (Merah)"
    )


def log_move(player: int, action: tuple, formed_mill: bool = False):
    """Log a move"""
    env = st.session_state.env
    st.session_state.move_count += 1
    
    player_name = "Biru" if player == 1 else "Merah"
    action_type, from_pos, to_pos = action
    
    if action_type == 'place':
        desc = f"placed at {to_pos}"
    elif action_type == 'move':
        desc = f"moved {from_pos} → {to_pos}"
    elif action_type == 'capture':
        desc = f"captured at {from_pos}"
    else:
        desc = str(action)
        
    mill_str = " ⭐ MILL!" if formed_mill else ""
    log_entry = f"{st.session_state.move_count:3d}. [{player_name}] {desc}{mill_str}"
    st.session_state.move_log.append(log_entry)
    
    if st.session_state.game_id:
        db.log_move(
            game_id=st.session_state.game_id,
            move_number=st.session_state.move_count,
            player=player_name,
            action_type=action_type,
            from_pos=from_pos,
            to_pos=to_pos,
            description=desc,
            formed_mill=formed_mill
        )


def execute_turn():
    """Execute one turn (or part of turn) for the current AI"""
    env = st.session_state.env
    
    model = st.session_state.model1 if env.current_player == 1 else st.session_state.model2
    player_num = env.current_player
    
    # 1. Get Action
    action = get_ai_move(model, env, st.session_state.device)
    
    # Update last move for visualization
    if action[0] == 'move':
        st.session_state.last_move = (action[1], action[2])
    elif action[0] == 'place':
        st.session_state.last_move = (None, action[2])
    elif action[0] == 'capture':
        # Don't update last move arrow for capture, maybe just highlight?
        # Let's keep the previous move indicator or indicate capture target
        # Currently the draw_board function highlights capture separately if pending,
        # but here we execute immediately.
        pass
    
    # 2. Execute Action
    state, reward, done, info = env.step(action)
    log_move(player_num, action, info.get('formed_mill', False))
    
    # 3. Handle Capture if needed
    if info.get('needs_capture', False):
        capture_action = get_ai_capture(model, env, st.session_state.device)
        state, reward, done, _ = env.step(capture_action)
        log_move(player_num, capture_action, False)
        # Maybe highlight capture position for a moment?
    
    return done, env.winner


def main():
    init_session_state()
    
    # --- HEADER SECTION ---
    st.markdown("""
        <div class="main-title">
            Implementasi Reinforcement Learning menggunakan Algoritma<br>
            Proximal Policy Optimization (PPO) pada Permainan Nine Men's Morris
        </div>
        <div class="subtitle">
            Rangga Wahyu Pratama-4233250028-PSIK 23 C
        </div>
    """, unsafe_allow_html=True)
    
    # Load models
    if not load_models():
        st.stop()
        
    # --- UI LAYOUT ---
    board_col_left, board_col_center, board_col_right = st.columns([1, 3, 1])
    
    with board_col_left:
        # Blue player piece count
        env = st.session_state.env
        pieces_remaining = max(0, env.pieces_in_hand[1])
        pieces_on_board_blue = min(9, max(0, env.pieces_on_board[1]))
        
        # Get current phase logic
        blue_phase = env.player_phase[1]
        
        st.markdown(f"""
            <div class="piece-counter piece-counter-blue">
                <div class="piece-count-label">MODEL 1 (BIRU)</div>
                <div class="piece-count-number player-blue">{pieces_remaining}</div>
                <div class="piece-count-label">Bidak Tersisa</div>
                <div style="margin-top: 0.5rem; font-size: 0.8rem; color: #00ccff;">Di Papan: {pieces_on_board_blue}</div>
                <div style="margin-top: 0.3rem; font-size: 0.75rem; color: #aaffff;">{blue_phase.upper()}</div>
            </div>
        """, unsafe_allow_html=True)
    
    with board_col_center:
        # Status Text
        status_text = "Siap untuk Memulai"
        if st.session_state.game_started:
            if st.session_state.game_over:
                if env.winner == 1:
                    status_text = "🏆 MODEL 1 (BIRU) MENANG!"
                elif env.winner == -1:
                    status_text = "🏆 MODEL 2 (MERAH) MENANG!"
                else:
                    status_text = "🤝 SERI!"
            else:
                player_name = "MODEL 1 (BIRU)" if env.current_player == 1 else "MODEL 2 (MERAH)"
                status_text = f"🔄 GILIRAN: {player_name}"
        
        st.markdown(f"<div class='status-text'>{status_text}</div>", unsafe_allow_html=True)

        # Render Game Board
        board_img = draw_board(
            board_state=env.board,
            highlights=None,
            selected_piece=None,
            last_move=st.session_state.last_move,
            current_player=env.current_player
        )
        st.image(board_img, use_container_width=True)
        
        # Action Buttons
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if not st.session_state.game_started or st.session_state.game_over:
                if st.button("▶ START AI MATCH", use_container_width=True):
                    start_game()
                    st.rerun()
            else:
                 if st.button("⏹ STOP", use_container_width=True):
                    st.session_state.game_started = False
                    st.session_state.auto_play = False
                    st.rerun()
        with col_btn2:
            if st.button("🔄 RESET BOARD", use_container_width=True):
                st.session_state.env = NineMensMorrisEnv()
                st.session_state.game_started = False
                st.session_state.game_over = False
                st.session_state.move_log = []
                st.session_state.move_count = 0
                st.rerun()

    with board_col_right:
        # Red player piece count
        env = st.session_state.env
        pieces_remaining = max(0, env.pieces_in_hand[-1])
        pieces_on_board_red = min(9, max(0, env.pieces_on_board[-1]))
        
        red_phase = env.player_phase[-1]
        
        st.markdown(f"""
            <div class="piece-counter piece-counter-red">
                <div class="piece-count-label">MODEL 2 (MERAH)</div>
                <div class="piece-count-number player-red">{pieces_remaining}</div>
                <div class="piece-count-label">Bidak Tersisa</div>
                <div style="margin-top: 0.5rem; font-size: 0.8rem; color: #ff6b6b;">Di Papan: {pieces_on_board_red}</div>
                <div style="margin-top: 0.3rem; font-size: 0.75rem; color: #ffaaaa;">{red_phase.upper()}</div>
            </div>
        """, unsafe_allow_html=True)

    # --- BELOW COLUMNS (Legend & Logs) ---
    st.markdown("---")
    
    # Legend
    st.markdown("""
        <div class="legend-container">
            <h4 style="color: #ffffff; text-align: center; margin-bottom: 1rem;">📖 Legenda</h4>
            <div style="display: flex; justify-content: space-around; flex-wrap: wrap;">
                <div class="legend-item">🔵 <span style="color: #00ccff;">Biru</span> = Model 1</div>
                <div class="legend-item">🔴 <span style="color: #ff6b6b;">Merah</span> = Model 2</div>
                <div class="legend-item">🟣 <span style="color: #b464ff;">Ungu</span> = Langkah Terakhir</div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Logs
    st.markdown("<h3 class='log-title'>📜 Log Permainan</h3>", unsafe_allow_html=True)
    if st.session_state.move_log:
        log_html = "<div class='log-container'>"
        for entry in reversed(st.session_state.move_log):
            if "[Biru]" in entry: css = "log-blue"
            elif "[Merah]" in entry: css = "log-red"
            else: css = ""
            
            if "MILL" in entry:
                entry = entry.replace("MILL!", "<span class='log-mill'>MILL!</span>")
            
            log_html += f"<div class='log-entry {css}'>{entry}</div>"
        log_html += "</div>"
        st.markdown(log_html, unsafe_allow_html=True)
        
        # Statistics
        mills_blue = sum(1 for e in st.session_state.move_log if "[Biru]" in e and "MILL" in e)
        mills_red = sum(1 for e in st.session_state.move_log if "[Merah]" in e and "MILL" in e)
        
        st.markdown(f"""
            <div class='stats-container'>
                <div class='stat-box'>
                    <div class='stat-label'>Total Langkah</div>
                    <div class='stat-value'>{st.session_state.move_count}</div>
                </div>
                <div class='stat-box'>
                    <div class='stat-label'>Mill Biru</div>
                    <div class='stat-value' style='color: #00ccff;'>{mills_blue}</div>
                </div>
                <div class='stat-box'>
                    <div class='stat-label'>Mill Merah</div>
                    <div class='stat-value' style='color: #ff6b6b;'>{mills_red}</div>
                </div>
            </div>
        """, unsafe_allow_html=True)

    # --- AUTO PLAY LOGIC ---
    if st.session_state.game_started and not st.session_state.game_over and st.session_state.auto_play:
        time.sleep(1.0) 
        done, winner = execute_turn()
        
        if done:
            st.session_state.game_over = True
            st.session_state.auto_play = False
            if winner == 1: w_name = "Biru (Model 1)"
            elif winner == -1: w_name = "Merah (Model 2)"
            else: w_name = "Seri"
            db.log_game_end(st.session_state.game_id, w_name, st.session_state.move_count)
            st.rerun() # Rerun to show winner banner
        else:
            st.rerun() # Rerun to show next state

if __name__ == "__main__":
    main()
