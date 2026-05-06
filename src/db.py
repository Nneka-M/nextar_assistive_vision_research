import sqlite3, json, time, os

DB_PATH = "./data/nextar.db"

def init_db():
    """Initialize the SQLite database and create the necessary table if it doesn't exist."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS interactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            image_path TEXT NOT NULL,
            detections TEXT NOT NULL,
            heatmap_region TEXT NOT NULL,
            scene_description TEXT NOT NULL,
            audio_path TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

def log_detection(image_path, detections, heatmap_region, scene_description, audio_path):
    con = sqlite3.connect(DB_PATH)
    cursor = con.cursor()
    cursor.execute("INSERT INTO interactions (timestamp, image_path, detections, heatmap_region, scene_description, audio_path) VALUES (?, ?, ?, ?, ?, ?)", (time.time(), image_path, json.dumps(detections), json.dumps(heatmap_region), scene_description, audio_path))
    con.commit()
    con.close()