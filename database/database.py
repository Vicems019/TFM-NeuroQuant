import sqlite3

DB_NAME = "neuroquant_db.db"

def create_database():
    with sqlite3.connect(DB_NAME) as conn:
        cursor = conn.cursor()
        cursor.executescript("""
            CREATE TABLE IF NOT EXISTS usuarios (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS operaciones (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                crypto_id TEXT NOT NULL,
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
                descripcion TEXT NOT NULL
            );
        """)
        print("✅ Base de datos lista")

if __name__ == "__main__":
    create_database()