"""
Board Visualization Module for Nine Men's Morris
Renders game board as image using PIL (no Pygame dependency for Streamlit Cloud)
"""

import base64
import numpy as np

from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
from typing import List, Optional, Tuple, Dict

# Board configuration
BOARD_SIZE = 600
MARGIN = 60
CELL_SIZE = (BOARD_SIZE - 2 * MARGIN) // 6

# Colors
COLOR_BACKGROUND = (30, 30, 40)  # Dark background
COLOR_BOARD = (45, 45, 60)  # Board background
COLOR_LINE = (100, 100, 120)  # Lines
COLOR_LINE_HIGHLIGHT = (150, 150, 180)  # Highlighted lines
COLOR_PLAYER1 = (65, 145, 255)  # Blue for Player 1
COLOR_PLAYER1_GLOW = (100, 170, 255)  # Glow effect
COLOR_PLAYER2 = (255, 85, 85)  # Red for Player 2
COLOR_PLAYER2_GLOW = (255, 120, 120)  # Glow effect
COLOR_EMPTY = (80, 80, 100)  # Empty position
COLOR_HIGHLIGHT = (255, 215, 0)  # Gold for valid moves
COLOR_SELECTED = (0, 255, 150)  # Green for selected piece
COLOR_LAST_MOVE = (180, 100, 255)  # Purple for last move

# Piece sizes
PIECE_RADIUS = 22
POSITION_RADIUS = 10
HIGHLIGHT_RADIUS = 28

# Position coordinates on board (pixel positions)
def get_position_coords(pos: int, board_size: int = BOARD_SIZE, margin: int = MARGIN) -> Tuple[int, int]:
    """Convert position index to pixel coordinates"""
    # Board position mapping (normalized 0-6 grid)
    POSITION_GRID = {
        0: (0, 0), 1: (3, 0), 2: (6, 0),
        3: (1, 1), 4: (3, 1), 5: (5, 1),
        6: (2, 2), 7: (3, 2), 8: (4, 2),
        9: (0, 3), 10: (1, 3), 11: (2, 3),
        12: (4, 3), 13: (5, 3), 14: (6, 3),
        15: (2, 4), 16: (3, 4), 17: (4, 4),
        18: (1, 5), 19: (3, 5), 20: (5, 5),
        21: (0, 6), 22: (3, 6), 23: (6, 6)
    }
    
    if pos not in POSITION_GRID:
        return (0, 0)

    grid_x, grid_y = POSITION_GRID[pos]
    cell_size = (board_size - 2 * margin) // 6
    
    x = margin + grid_x * cell_size
    y = margin + grid_y * cell_size
    
    return (x, y)


def draw_board(board_state: np.ndarray, 
               highlights: Optional[List[int]] = None,
               selected_piece: Optional[int] = None,
               last_move: Optional[Tuple[int, int]] = None,
               current_player: int = 1,
               pending_capture: bool = False) -> Image.Image:
    """
    Draw the Nine Men's Morris board
    
    Args:
        board_state: Array of 24 positions (0=empty, 1=player1, -1=player2)
        highlights: List of positions to highlight as valid moves
        selected_piece: Position of currently selected piece
        last_move: Tuple of (from_pos, to_pos) for last move
        current_player: Current player (1 or -1)
        pending_capture: Whether we're in capture mode
    
    Returns:
        PIL Image of the board
    """
    highlights = highlights or []
    
    # Create image
    img = Image.new('RGB', (BOARD_SIZE, BOARD_SIZE), COLOR_BACKGROUND)
    draw = ImageDraw.Draw(img)
    
    # Draw board background with rounded corners effect
    draw.rounded_rectangle(
        [(MARGIN - 20, MARGIN - 20), (BOARD_SIZE - MARGIN + 20, BOARD_SIZE - MARGIN + 20)],
        radius=15,
        fill=COLOR_BOARD
    )
    
    # Draw the three squares
    squares = [
        (0, 6),   # Outer square
        (1, 5),   # Middle square  
        (2, 4)    # Inner square
    ]
    
    cell_size = (BOARD_SIZE - 2 * MARGIN) // 6
    
    for start, end in squares:
        x1 = MARGIN + start * cell_size
        y1 = MARGIN + start * cell_size
        x2 = MARGIN + end * cell_size
        y2 = MARGIN + end * cell_size
        
        # Draw square
        draw.rectangle([(x1, y1), (x2, y2)], outline=COLOR_LINE, width=3)
    
    # Draw connecting lines (middle lines)
    mid = BOARD_SIZE // 2
    
    # Horizontal middle lines
    draw.line([(MARGIN, mid), (MARGIN + 2 * cell_size, mid)], fill=COLOR_LINE, width=3)
    draw.line([(MARGIN + 4 * cell_size, mid), (BOARD_SIZE - MARGIN, mid)], fill=COLOR_LINE, width=3)
    
    # Vertical middle lines
    draw.line([(mid, MARGIN), (mid, MARGIN + 2 * cell_size)], fill=COLOR_LINE, width=3)
    draw.line([(mid, MARGIN + 4 * cell_size), (mid, BOARD_SIZE - MARGIN)], fill=COLOR_LINE, width=3)
    
    # Draw last move indicator
    if last_move:
        from_pos, to_pos = last_move
        
        # Draw line between from and to if both exist (movement)
        if from_pos is not None and to_pos is not None:
             fx, fy = get_position_coords(from_pos)
             tx, ty = get_position_coords(to_pos)
             draw.line([(fx, fy), (tx, ty)], fill=COLOR_LAST_MOVE, width=2)

        # Highlight source position
        if from_pos is not None:
            fx, fy = get_position_coords(from_pos)
            draw.ellipse(
                [(fx - HIGHLIGHT_RADIUS, fy - HIGHLIGHT_RADIUS),
                 (fx + HIGHLIGHT_RADIUS, fy + HIGHLIGHT_RADIUS)],
                outline=COLOR_LAST_MOVE, width=3
            )
            
        # Highlight dest position
        if to_pos is not None:
            tx, ty = get_position_coords(to_pos)
            draw.ellipse(
                [(tx - HIGHLIGHT_RADIUS, ty - HIGHLIGHT_RADIUS),
                 (tx + HIGHLIGHT_RADIUS, ty + HIGHLIGHT_RADIUS)],
                outline=COLOR_LAST_MOVE, width=3
            )
    
    # Draw valid move highlights
    for pos in highlights:
        px, py = get_position_coords(pos)
        
        if pending_capture:
            # Red pulsing highlight for capture targets
            color = (255, 100, 100)
        else:
            color = COLOR_HIGHLIGHT
        
        draw.ellipse(
            [(px - HIGHLIGHT_RADIUS, py - HIGHLIGHT_RADIUS),
             (px + HIGHLIGHT_RADIUS, py + HIGHLIGHT_RADIUS)],
            outline=color, width=4
        )
    
    # Draw selected piece highlight
    if selected_piece is not None:
        sx, sy = get_position_coords(selected_piece)
        draw.ellipse(
            [(sx - HIGHLIGHT_RADIUS - 2, sy - HIGHLIGHT_RADIUS - 2),
             (sx + HIGHLIGHT_RADIUS + 2, sy + HIGHLIGHT_RADIUS + 2)],
            outline=COLOR_SELECTED, width=4
        )
    
    # Draw positions and pieces
    for pos in range(24):
        px, py = get_position_coords(pos)
        piece = board_state[pos]
        
        if piece == 1:
            # Player 1 (Blue) piece
            # Draw glow effect
            draw.ellipse(
                [(px - PIECE_RADIUS - 4, py - PIECE_RADIUS - 4),
                 (px + PIECE_RADIUS + 4, py + PIECE_RADIUS + 4)],
                fill=COLOR_PLAYER1_GLOW
            )
            # Draw piece
            draw.ellipse(
                [(px - PIECE_RADIUS, py - PIECE_RADIUS),
                 (px + PIECE_RADIUS, py + PIECE_RADIUS)],
                fill=COLOR_PLAYER1
            )
            # Inner highlight for 3D effect
            draw.ellipse(
                [(px - PIECE_RADIUS + 6, py - PIECE_RADIUS + 4),
                 (px - PIECE_RADIUS + 14, py - PIECE_RADIUS + 10)],
                fill=(130, 190, 255)
            )
            
        elif piece == -1:
            # Player 2 (Red) piece
            # Draw glow effect
            draw.ellipse(
                [(px - PIECE_RADIUS - 4, py - PIECE_RADIUS - 4),
                 (px + PIECE_RADIUS + 4, py + PIECE_RADIUS + 4)],
                fill=COLOR_PLAYER2_GLOW
            )
            # Draw piece
            draw.ellipse(
                [(px - PIECE_RADIUS, py - PIECE_RADIUS),
                 (px + PIECE_RADIUS, py + PIECE_RADIUS)],
                fill=COLOR_PLAYER2
            )
            # Inner highlight for 3D effect
            draw.ellipse(
                [(px - PIECE_RADIUS + 6, py - PIECE_RADIUS + 4),
                 (px - PIECE_RADIUS + 14, py - PIECE_RADIUS + 10)],
                fill=(255, 150, 150)
            )
            
        else:
            # Empty position
            draw.ellipse(
                [(px - POSITION_RADIUS, py - POSITION_RADIUS),
                 (px + POSITION_RADIUS, py + POSITION_RADIUS)],
                fill=COLOR_EMPTY
            )
    
    return img
