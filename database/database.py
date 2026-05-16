import sqlite3

DB_NAME = "neuroquant_db.db"

def create_database():
    with sqlite3.connect(DB_NAME) as conn:
        cursor = conn.cursor()
        cursor.executescript("""
            CREATE TABLE IF NOT EXISTS usuarios (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL,
                saldo REAL NOT NULL DEFAULT 10000.0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS operaciones (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                crypto_id INTEGER NOT NULL,
                tipo TEXT NOT NULL CHECK(tipo IN ('BUY','SELL')),
                cantidad REAL NOT NULL,
                precio REAL NOT NULL,
                fecha TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(user_id) REFERENCES usuarios(id),
                FOREIGN KEY(crypto_id) REFERENCES criptomonedas(id)
            );
            CREATE TABLE IF NOT EXISTS criptomonedas (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                nombre TEXT NOT NULL,
                descripcion TEXT NOT NULL,
                symbol TEXT NOT NULL
            );
        """)
        print("✅ Base de datos lista")

def insert_data():
    with sqlite3.connect(DB_NAME) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO criptomonedas (nombre, descripcion, symbol) VALUES
            ('Bitcoin',   'Criptomoneda descentralizada', 'BTC'),
            ('Ethereum',  'Plataforma de contratos',      'ETH'),
            ('Solana',    'Blockchain de alta velocidad', 'SOL'),
            ('Avalanche', 'Red multicadena',              'AVAX');
        """)
        conn.commit()

if __name__ == "__main__":
    create_database()