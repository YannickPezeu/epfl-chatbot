
import numpy as np

from myUtils.connect_acad import reconnect_on_failure
@reconnect_on_failure
def retrieve_embedding_from_db_online(embedding_ids: list, cursor):

    # Create a tuple of question marks with length equal to embedding_ids length
    placeholders = ','.join(['%s'] * len(embedding_ids))

    # Use the placeholders in the query
    query = f"SELECT model_id, embedding FROM embeddings WHERE id IN ({placeholders}) ORDER BY id"
    print('query', query)
    cursor.execute(query, embedding_ids)

    result = cursor.fetchall()

    model_id = result[0][0]
    embedding_binaries = [row[1] for row in result]

    # get number structure of model
    cursor.execute("SELECT dtype FROM embeddings_models WHERE id=%s", (model_id,))
    dtype = cursor.fetchone()[0]

    if dtype == 'float64':
        embedding_array = [np.frombuffer(embedding, dtype=np.float64).reshape(1, -1) for embedding in embedding_binaries]
        #transform into numpy array
        embedding_array = np.concatenate(embedding_array, axis=0)
    elif dtype == 'float32':
        embedding_array = [np.frombuffer(embedding, dtype=np.float32).reshape(1, -1) for embedding in embedding_binaries]
        #transform into numpy array
        embedding_array = np.concatenate(embedding_array, axis=0)
    else:
        raise ValueError('dtype not supported')

    return embedding_array



if __name__ == '__main__':
    import os
    import sys

    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(current_dir)

    print('root_dir:', root_dir)

    db_path = os.path.join(root_dir, 'data/LEXs/LEXs.db')
    emb = retrieve_embedding_from_db(db_path, [1])

    print(emb)
    print(emb.shape)
