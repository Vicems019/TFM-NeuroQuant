import sqlite3
import os

DB_PATH = "neuroquant_db.db"


def _conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


# ── Consultas de lectura ──────────────────────────────────────────────────────

def get_saldo_cash(username: str) -> float:
    with _conn() as conn:
        row = conn.execute(
            "SELECT saldo FROM usuarios WHERE username = ?", (username,)
        ).fetchone()
    return float(row["saldo"]) if row else 100_000.0


def get_criptomonedas() -> list[dict]:
    with _conn() as conn:
        rows = conn.execute(
            "SELECT id, nombre, symbol FROM criptomonedas ORDER BY nombre"
        ).fetchall()
    return [dict(r) for r in rows]


def get_posiciones(username: str) -> list[dict]:
    with _conn() as conn:
        rows = conn.execute("""
            SELECT
                c.symbol,
                c.nombre,
                SUM(CASE WHEN o.tipo = 'BUY'  THEN o.cantidad           ELSE 0 END) AS qty_buy,
                SUM(CASE WHEN o.tipo = 'SELL' THEN o.cantidad           ELSE 0 END) AS qty_sell,
                SUM(CASE WHEN o.tipo = 'BUY'  THEN o.cantidad * o.precio ELSE 0 END) AS coste_total,
                SUM(CASE WHEN o.tipo = 'BUY'  THEN o.cantidad            ELSE 0 END) -
                SUM(CASE WHEN o.tipo = 'SELL' THEN o.cantidad            ELSE 0 END) AS qty_net
            FROM operaciones o
            JOIN usuarios      u ON o.user_id  = u.id
            JOIN criptomonedas c ON o.crypto_id = c.id
            WHERE u.username = ?
            GROUP BY c.id, c.symbol, c.nombre
            HAVING qty_net > 0.0000001
        """, (username,)).fetchall()
    results = []
    for r in rows:
        d = dict(r)
        d["precio_medio"] = d["coste_total"] / d["qty_buy"] if d["qty_buy"] else 0.0
        results.append(d)
    return results


def get_historial(username: str, limit: int = 30) -> list[dict]:
    with _conn() as conn:
        rows = conn.execute("""
            SELECT
                o.id,
                c.symbol,
                c.nombre,
                o.tipo,
                o.cantidad,
                o.precio,
                o.fecha,
                o.cantidad * o.precio AS valor_total
            FROM operaciones o
            JOIN usuarios      u ON o.user_id  = u.id
            JOIN criptomonedas c ON o.crypto_id = c.id
            WHERE u.username = ?
            ORDER BY o.fecha DESC
            LIMIT ?
        """, (username, limit)).fetchall()
    return [dict(r) for r in rows]


def get_resumen_portfolio(username: str) -> dict:
    cash = get_saldo_cash(username)
    posiciones = get_posiciones(username)
    coste_invertido = sum(p["precio_medio"] * p["qty_net"] for p in posiciones)
    return {
        "cash":             cash,
        "coste_invertido":  coste_invertido,
        "capital_total_en_coste": cash + coste_invertido,
    }

def ejecutar_compra(
    username: str,
    crypto_symbol: str,
    cantidad: float,
    precio_actual: float,
) -> tuple[bool, str]:
    coste = cantidad * precio_actual
    with _conn() as conn:
        user = conn.execute(
            "SELECT id, saldo_cash FROM usuarios WHERE username = ?", (username,)
        ).fetchone()
        if not user:
            return False, "Usuario no encontrado."

        if float(user["saldo_cash"]) < coste:
            return False, (
                f"Saldo insuficiente. "
                f"Disponible: ${user['saldo_cash']:,.2f} | "
                f"Necesario: ${coste:,.2f}"
            )

        crypto = conn.execute(
            "SELECT id FROM criptomonedas WHERE symbol = ?", (crypto_symbol,)
        ).fetchone()
        if not crypto:
            return False, f"Criptomoneda '{crypto_symbol}' no encontrada en la BD."

        conn.execute(
            "INSERT INTO operaciones (user_id, crypto_id, tipo, cantidad, precio) "
            "VALUES (?, ?, 'BUY', ?, ?)",
            (user["id"], crypto["id"], cantidad, precio_actual),
        )
        conn.execute(
            "UPDATE usuarios SET saldo_cash = saldo_cash - ? WHERE id = ?",
            (coste, user["id"]),
        )
        conn.commit()

    return True, f"✅ Compra ejecutada: {cantidad:.6f} {crypto_symbol} a ${precio_actual:,.2f}"


def ejecutar_venta(
    username: str,
    crypto_symbol: str,
    cantidad: float,
    precio_actual: float,
) -> tuple[bool, str]:
    with _conn() as conn:
        user = conn.execute(
            "SELECT id FROM usuarios WHERE username = ?", (username,)
        ).fetchone()
        if not user:
            return False, "Usuario no encontrado."

        crypto = conn.execute(
            "SELECT id FROM criptomonedas WHERE symbol = ?", (crypto_symbol,)
        ).fetchone()
        if not crypto:
            return False, f"Criptomoneda '{crypto_symbol}' no encontrada en la BD."

        pos = conn.execute("""
            SELECT
                SUM(CASE WHEN tipo = 'BUY'  THEN cantidad ELSE 0 END) -
                SUM(CASE WHEN tipo = 'SELL' THEN cantidad ELSE 0 END) AS qty_net
            FROM operaciones
            WHERE user_id = ? AND crypto_id = ?
        """, (user["id"], crypto["id"])).fetchone()

        qty_net = float(pos["qty_net"] or 0)
        if qty_net < cantidad:
            return False, (
                f"Posición insuficiente. "
                f"Tienes {qty_net:.6f} {crypto_symbol} | "
                f"Intentas vender {cantidad:.6f}"
            )

        ingreso = cantidad * precio_actual
        conn.execute(
            "INSERT INTO operaciones (user_id, crypto_id, tipo, cantidad, precio) "
            "VALUES (?, ?, 'SELL', ?, ?)",
            (user["id"], crypto["id"], cantidad, precio_actual),
        )
        conn.execute(
            "UPDATE usuarios SET saldo_cash = saldo_cash + ? WHERE id = ?",
            (ingreso, user["id"]),
        )
        conn.commit()

    return True, f"✅ Venta ejecutada: {cantidad:.6f} {crypto_symbol} a ${precio_actual:,.2f}"

