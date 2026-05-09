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
            phase TEXT --'phase1' or 'phase2'
            image_path TEXT NOT NULL,
            detections TEXT NOT NULL,
            heatmap_region TEXT NOT NULL,
            scene_description TEXT NOT NULL,
            audio_path TEXT NOT NULL
        )
    ''')
    columns = [row[1] for row in conn.execute("PRAGMA table_info(interactions)")]
    if "phase" not in columns:
        conn.execute("ALTER TABLE interactions ADD COLUMN phase TEXT")
        print("[DB] Migrated: added phase column")

    conn.commit()
    conn.close()

def log_detection(image_path, phase, detections, heatmap_region, scene_description, audio_path):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO interactions (timestamp, phase, image_path, detections, heatmap_region, scene_description, audio_path) VALUES (?, ?, ?, ?, ?, ?, ?)", (time.time(), phase, image_path, json.dumps(detections), json.dumps(heatmap_region), scene_description, audio_path))
    conn.commit()
    conn.close()