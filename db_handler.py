
import sqlite3
import os

class DatabaseHandler:
    def __init__(self):
        self.db_file = 'users.db'
        self.initialize_db()
    
    def initialize_db(self):
        if not os.path.exists(self.db_file):
            conn = sqlite3.connect(self.db_file)
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    username TEXT PRIMARY KEY,
                    password TEXT,
                    email TEXT,
                    joined_date TEXT,
                    chat_history TEXT
                )
            ''')
            conn.commit()
            conn.close()
    
    def add_user(self, username, password, email, joined_date):
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        try:
            cursor.execute('INSERT INTO users VALUES (?, ?, ?, ?, ?)', 
                         (username, password, email, joined_date, '[]'))
            conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False
        finally:
            conn.close()
    
    def get_user(self, username):
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM users WHERE username = ?', (username,))
        user = cursor.fetchone()
        conn.close()
        if user:
            return {
                'username': user[0],
                'password': user[1],
                'email': user[2],
                'joined_date': user[3],
                'chat_history': user[4]
            }
        return None
    
    def update_chat_history(self, username, chat_history):
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        cursor.execute('UPDATE users SET chat_history = ? WHERE username = ?',
                      (chat_history, username))
        conn.commit()
        conn.close()
