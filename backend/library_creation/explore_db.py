import sqlite3
project_name = 'ressources_humaines'
model_name = 'dangvantuan__sentence-camembert-large'
# Connect to database
conn = sqlite3.connect(f'{project_name}__{model_name}.db')

# Create a cursor
c = conn.cursor()

c.execute("SELECT id, source_doc_id, page_number FROM big_chunks ORDER BY source_doc_id, page_number")
print(c.fetchall())