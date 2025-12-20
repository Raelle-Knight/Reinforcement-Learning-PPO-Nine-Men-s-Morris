"""
Database Module for Nine Men's Morris
Logs game stats and moves to SQLite database
"""

import os
import sqlite3
import datetime

DB_NAME = "ninemensmorris.db"

def init_db():
    """Initialize database tables"""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    
    # Games table
    c.execute('''
    CREATE TABLE IF NOT EXISTS games (
        game_id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        mode TEXT,
        model1_name TEXT,
        model2_name TEXT,
        winner TEXT,
        total_moves INTEGER
    )
    ''')
    
    # Moves table
    c.execute('''
    CREATE TABLE IF NOT EXISTS moves (
        move_id INTEGER PRIMARY KEY AUTOINCREMENT,
        game_id INTEGER,
        move_number INTEGER,
        player TEXT,
        action_type TEXT,
        from_pos INTEGER,
        to_pos INTEGER,
        description TEXT,
        formed_mill BOOLEAN,
        FOREIGN KEY(game_id) REFERENCES games(game_id)
    )
    ''')
    
    conn.commit()
    conn.close()

def log_game_start(mode: str, model1_name: str, model2_name: str) -> int:
    """Log start of a new game"""
    init_db()  # Ensure tables exist
    
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    c.execute('''
    INSERT INTO games (timestamp, mode, model1_name, model2_name, winner, total_moves)
    VALUES (?, ?, ?, ?, NULL, 0)
    ''', (timestamp, mode, model1_name, model2_name))
    
    game_id = c.lastrowid
    conn.commit()
    conn.close()
    
    return game_id

def log_move(game_id: int, move_number: int, player: str, action_type: str, 
             from_pos: int, to_pos: int, description: str, formed_mill: bool = False):
    """Log a single move"""
    if not game_id:
        return
        
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    
    # Handle None values for SQL
    from_pos_val = from_pos if from_pos is not None else -1
    to_pos_val = to_pos if to_pos is not None else -1
    
    c.execute('''
    INSERT INTO moves (game_id, move_number, player, action_type, from_pos, to_pos, description, formed_mill)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (game_id, move_number, player, action_type, from_pos_val, to_pos_val, description, formed_mill))
    
    conn.commit()
    conn.close()

def log_game_end(game_id: int, winner: str, total_moves: int):
    """Update game record with winner and total moves"""
    if not game_id:
        return
        
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    
    c.execute('''
    UPDATE games 
    SET winner = ?, total_moves = ?
    WHERE game_id = ?
    ''', (winner, total_moves, game_id))
    
    conn.commit()
    conn.close()
