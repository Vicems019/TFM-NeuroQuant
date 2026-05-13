import sqlite3
import os

DB_PATH = "neuroquant_db.db"


def run():
    with sqlite3.connect(DB_PATH) as conn:

        # ── 1. symbol en criptomonedas ────────────────────────────────────────
        try:
            conn.execute("ALTER TABLE criptomonedas ADD COLUMN symbol TEXT DEFAULT ''")
            print("✅ Columna 'symbol' añadida a criptomonedas")
        except Exception:
            print("ℹ️  'symbol' ya existía en criptomonedas")

        # ── 2. saldo_cash en usuarios ─────────────────────────────────────────
        try:
            conn.execute("ALTER TABLE usuarios ADD COLUMN saldo_cash REAL DEFAULT 100000.0")
            print("✅ Columna 'saldo_cash' añadida a usuarios")
        except Exception:
            print("ℹ️  'saldo_cash' ya existía en usuarios")

        # ── 3. Seed / actualizar criptomonedas ────────────────────────────────
        existing = conn.execute("SELECT COUNT(*) FROM criptomonedas").fetchone()[0]
        CRYPTOS  = [
            ("Bitcoin",   "Criptomoneda descentralizada creada en 2009", "BTC"),
            ("Ethereum",  "Plataforma de contratos inteligentes",         "ETH"),
            ("Solana",    "Blockchain de alta velocidad y bajo coste",    "SOL"),
            ("Avalanche", "Red multicadena escalable",                    "AVAX"),
        ]

        if existing == 0:
            conn.executemany(
                "INSERT INTO criptomonedas (nombre, descripcion, symbol) VALUES (?,?,?)",
                CRYPTOS,
            )
            print("✅ Criptomonedas insertadas")
        else:
            for nombre, _, symbol in CRYPTOS:
                conn.execute(
                    "UPDATE criptomonedas SET symbol = ? WHERE nombre = ? AND (symbol = '' OR symbol IS NULL)",
                    (symbol, nombre),
                )
            print("✅ Symbols actualizados en criptomonedas existentes")

        conn.commit()

    print("\n🚀 Migración completada — ya puedes arrancar el papertrade.")


if __name__ == "__main__":
    run()