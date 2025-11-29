import sqlite3
import datetime
from threading import Lock

class Database:
    _instance = None
    _lock = Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(Database, cls).__new__(cls)
                cls._instance.init_db()
        return cls._instance

    def init_db(self):
        self.conn = sqlite3.connect('garbage_data.db', check_same_thread=False)
        self.cursor = self.conn.cursor()
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                label TEXT,
                confidence REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        self.conn.commit()

    def log_detection(self, label, confidence):
        self.cursor.execute('INSERT INTO detections (label, confidence) VALUES (?, ?)', (label, confidence))
        self.conn.commit()

    def get_stats(self):
        self.cursor.execute('SELECT label, COUNT(*) FROM detections GROUP BY label')
        return dict(self.cursor.fetchall())

    def get_recent_logs(self, limit=10):
        self.cursor.execute('SELECT label, confidence, timestamp FROM detections ORDER BY id DESC LIMIT ?', (limit,))
        return self.cursor.fetchall()
