import sqlite3
import bcrypt

DB_NAME = "neuroquant_db.db"

def register_user(username, password):

    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    password_hash = bcrypt.hashpw(
        password.encode(),
        bcrypt.gensalt()
    ).decode()

    try:

        cursor.execute("""
        INSERT INTO users (username, password_hash)
        VALUES (?, ?)
        """, (username, password_hash))
        conn.commit()

        return True

    except sqlite3.IntegrityError:
        return False

    finally:
        conn.close()

def login_user(username, password):

    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("""
    SELECT password_hash
    FROM users
    WHERE username = ?
    """, (username,))

    result = cursor.fetchone()

    conn.close()

    if result is None:
        return False

    stored_hash = result[0]

    return bcrypt.checkpw(
        password.encode(),
        stored_hash.encode()
    )


def user_exists(username):

    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("""
    SELECT id
    FROM users
    WHERE username = ?
    """, (username,))

    result = cursor.fetchone()

    conn.close()

    return result is not None