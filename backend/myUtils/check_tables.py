import sqlite3
def check_tables(db_path):

    # Connect to your SQLite database
    conn = sqlite3.connect(db_path)

    # Create a cursor object
    cursor = conn.cursor()

    # Retrieve the schema of all tables
    cursor.execute("SELECT sql FROM sqlite_master WHERE type='table';")
    for row in cursor.fetchall():
        print(row[0])

    # Close the connection
    conn.close()

if __name__ == '__main__':
    import os

    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(current_dir)

    db_path = os.path.join(root_dir, 'data/LEXs/LEXs.db')
    check_tables(db_path)

    db_path = os.path.join(root_dir, 'data/users/users.db')
    check_tables(db_path)
