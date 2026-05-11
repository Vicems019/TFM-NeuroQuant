import sqlite3
import os
from datetime import datetime

DB_NAME = "neuroquant_db.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS portfolio (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            balance REAL DEFAULT 100000.0
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            crypto TEXT,
            type TEXT,
            amount REAL,
            price REAL,
            total REAL,
            pnl REAL
        )
    ''')
    cursor.execute('SELECT COUNT(*) FROM portfolio')
    if cursor.fetchone()[0] == 0:
        cursor.execute('INSERT INTO portfolio (balance) VALUES (100000.0)')
    conn.commit()
    conn.close()

def get_balance():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('SELECT balance FROM portfolio ORDER BY id DESC LIMIT 1')
    res = cursor.fetchone()
    conn.close()
    return res[0] if res else 100000.0

def update_balance(new_balance):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('UPDATE portfolio SET balance = ?', (new_balance,))
    conn.commit()
    conn.close()

def get_last_operations(user_id: int, limit: int = 20):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT 
            o.id,
            o.tipo,
            o.cantidad,
            o.precio,
            o.fecha,
            c.nombre AS crypto_nombre,
            c.descripcion
        FROM operaciones o
        JOIN criptomonedas c ON o.crypto_id = c.id
        WHERE o.user_id = ?
        ORDER BY o.fecha DESC
        LIMIT ?
    """, (user_id, limit))

    data = cursor.fetchall()
    conn.close()

    return data

def get_trades():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute('SELECT nombre FROM criptomonedas WHERE nombre IN (SELECT crypto_id FROM operaciones)')
    rows = cursor.fetchall()
    
    cursor.execute('SELECT timestamp, crypto, type, amount, price, total, pnl FROM trades ORDER BY id DESC LIMIT 20')
    rows = cursor.fetchall()
    conn.close()
    res = []
    for r in rows:
        res.append({
            "fecha": r[0],
            "cripto": r[1],
            "tipo": r[2],
            "cantidad": r[3],
            "precio": r[4],
            "total": r[5],
            "pnl": r[6] if r[6] is not None else 0.0
        })
    return res

init_db()
