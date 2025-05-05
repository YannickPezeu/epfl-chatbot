from myUtils.redisStateManager import RedisStateManager

# Initialize Redis managers similar to how it's done in your code
string_redis = RedisStateManager(decode_responses=True)  # For JSON/string data
binary_redis = RedisStateManager(decode_responses=False)  # For binary data

# Define the keys you want to delete
model_id = 8  # Replace with actual model ID
library = "LEX AND RH"    # Replace with actual library name
username = "all_users"  # Replace with actual username

# Create the keys
index_key = f"faiss:index:{model_id}:{library}:{username}"
embedding_key = f"faiss:embeddings:{model_id}:{library}:{username}"

# Check if keys exist
index_exists = binary_redis.redis_client.exists(index_key)
embedding_exists = string_redis.redis_client.exists(embedding_key)

# Report on existence
if index_exists:
    print(f"Found index key: {index_key}")
    # Delete if needed
    binary_redis.redis_client.delete(index_key)
    print(f"Deleted index key: {index_key}")
else:
    print(f"Index key does not exist: {index_key}")

if embedding_exists:
    print(f"Found embedding key: {embedding_key}")
    # Delete if needed
    string_redis.redis_client.delete(embedding_key)
    print(f"Deleted embedding key: {embedding_key}")
else:
    print(f"Embedding key does not exist: {embedding_key}")

# Delete the keys
binary_redis.redis_client.delete(index_key)
string_redis.redis_client.delete(embedding_key)

print(f"Deleted Redis entries for {index_key} and {embedding_key}")