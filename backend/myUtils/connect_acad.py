import functools
import pymysql
from sshtunnel import SSHTunnelForwarder, create_logger
import asyncio
import aiomysql

# logging.basicConfig(level=logging.DEBUG)
# logger = create_logger(loglevel=logging.DEBUG)
import os
import dotenv
dotenv.load_dotenv()

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
ssh_dir = os.path.join(root_dir, 'ssh')

ssh_host = os.getenv('FAC_TUNNEL_HOST')
ssh_user = os.getenv('FAC_TUNNEL_USER')
ssh_private_key = os.path.join(ssh_dir,os.getenv('FAC_TUNNEL_KEY'))

db_host = os.getenv('FAC_HOST')
db_port = int(os.getenv('FAC_PORT'))
db_user = os.getenv('FAC_USER')
db_password = os.getenv('FAC_PASSWORD')
db_name = 'acad'

# Global variables for conn and tunnel
connection = None
tunnel = None
import traceback
import time

from pymysql.cursors import DictCursor
from dbutils.pooled_db import PooledDB

db_config = {
    'host': os.getenv('FAC_HOST'),
    'port': int(os.getenv('FAC_PORT')),
    'user': os.getenv('FAC_USER'),
    'password': os.getenv('FAC_PASSWORD'),
    'database': 'acad',
}

ssh_config = {
    'ssh_host': os.getenv('FAC_TUNNEL_HOST'),
    'ssh_user': os.getenv('FAC_TUNNEL_USER'),
    'ssh_private_key': os.path.join(os.path.dirname(os.path.dirname(__file__)), 'ssh', os.getenv('FAC_TUNNEL_KEY')),
}
pool = None
tunnel = None

print('ssh_config:', ssh_config)
print('db_config:', db_config)

def setup_ssh_tunnel():
    global tunnel
    if tunnel is None or not tunnel.is_active:
        tunnel = SSHTunnelForwarder(
            (ssh_config['ssh_host'], 22),
            ssh_username=ssh_config['ssh_user'],
            ssh_private_key=ssh_config['ssh_private_key'],
            remote_bind_address=(db_config['host'], db_config['port']),
            local_bind_address=('0.0.0.0', 0)
        )
        tunnel.start()
        # print(f"SSH tunnel established. Local bind port: {tunnel.local_bind_port}")
    return tunnel

tunnel = setup_ssh_tunnel()

def get_connection_pool():
    global pool, tunnel
    start = time.time()
    if tunnel is None or not tunnel.is_active:
        setup_ssh_tunnel()
    end = time.time()
    print(f"Time taken to setup tunnel: {end-start}")

    if pool is None:
        pool = PooledDB(
            creator=pymysql,
            maxconnections=6,
            mincached=2,
            maxcached=0,
            blocking=True,
            host='127.0.0.1',
            port=tunnel.local_bind_port,
            user=db_config['user'],
            password=db_config['password'],
            database=db_config['database'],
            reset=True,
        )
    end2 = time.time()
    print(f"Time taken to setup pool: {end2-end}")

    return pool

def get_pool_status():
    global pool
    if pool:
        return {
            "size": pool._connections,  # Total number of connections in the pool
            # "active": len(pool._used_connections),  # Currently in-use connections
            # "idle": len(pool._idle_connections),  # Available connections
            "maxconnections": pool._maxconnections,  # Maximum allowed connections
            "maxcached": pool._maxcached,
            "blocking": pool._blocking,
            'test': pool._failures,
        }
    return None



from contextlib import contextmanager

@contextmanager
def get_db_connection():
    conn = None
    cursor = None
    try:
        # print("\nPool status before getting connection:", get_pool_status())
        conn = pool.connection()
        # print("Pool status after getting connection:", get_pool_status())
        cursor = conn.cursor()
        yield conn, cursor
    finally:
        if cursor:
            try:
                cursor.close()
            except:
                print("Error closing cursor")
        if conn:
            try:
                conn.close()
                print("Pool status after closing connection:", get_pool_status())
            except:
                print("Error closing connection")

def initialize_all_connection():
    print('initialize_all_connection')
    global pool
    if pool is None:
        print('pool is None')
        pool = get_connection_pool()
    else:
        print('pool is not None')
    start = time.time()
    try:
        # print("Pool status before getting connection:", get_pool_status())
        conn = pool.connection()
        # print("Pool status after getting connection:", get_pool_status())
        # print('time taken to get connection:', time.time()-start)
        start = time.time()
        cursor = conn.cursor()
        # print('time taken to get cursor:', time.time()-start)
        return conn, cursor
    except Exception as e:
        print(f"Database error occurred in initialize: {e}")
        print("Traceback:")
        print(traceback.format_exc())
        raise

pool = get_connection_pool()
conn = pool.connection()
cursor = conn.cursor()

def reconnect_on_failure(func):
    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs):
        max_retries = 1
        retry_count = 0
        try:
            result = await func(*args, **kwargs)
            return result
        except Exception as e:
            print(f"Database error occurred: {e}")
            while retry_count < max_retries:
                try:
                    pool = get_connection_pool()
                    start = time.time()
                    conn = pool.connection()
                    cursor = conn.cursor()
                    if 'cursor' in kwargs:
                        kwargs['cursor'] = cursor
                    result = await func(*args, **kwargs)
                    conn.commit()
                    return result
                except Exception as e:
                    print(f"Database error occurred2: {e}")
                    retry_count += 1
                    if retry_count >= max_retries:
                        raise
                    await asyncio.sleep(2 ** retry_count)
                finally:
                    if cursor:
                        try:
                            print('closing cursor, reconnect_on_failure')
                            cursor.close()
                        except:
                            print("Error closing cursor")
                    if conn:
                        try:
                            print('closing connection, reconnect_on_failure')
                            conn.close()
                        except:
                            print("Error closing connection")

    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs):
        max_retries = 1
        retry_count = 0
        try:
            result = func(*args, **kwargs)
            return result
        except Exception as e:
            print(f"Database error occurred: {e}")
            while retry_count < max_retries:
                try:
                    pool = get_connection_pool()
                    start = time.time()
                    conn = pool.connection()
                    cursor = conn.cursor()
                    if 'cursor' in kwargs:
                        kwargs['cursor'] = cursor
                    result = func(*args, **kwargs)
                    conn.commit()
                    return result
                except Exception as e:
                    print(f"Database error occurred2: {e}")
                    retry_count += 1
                    if retry_count >= max_retries:
                        raise
                    time.sleep(2 ** retry_count)
                finally:
                    if cursor:
                        try:
                            print('closing cursor, reconnect_on_failure')
                            cursor.close()
                        except:
                            print("Error closing cursor")
                    if conn:
                        try:
                            print('closing connection, reconnect_on_failure')
                            conn.close()
                        except:
                            print("Error closing connection")

    # Check if the function is async or not
    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    return sync_wrapper

def reconnect_on_failure_sync(func):
    print('reconnect_on_failure')
    @functools.wraps(func)
    def wrapper_reconnect_on_failure(*args, **kwargs):
        max_retries = 1
        retry_count = 0
        try:
            result = func(*args, **kwargs)
            return result

        except Exception as e:
            print(f"Database error occurred: {e}")

            while retry_count < max_retries:
                try:
                    pool = get_connection_pool()
                    # print("pool:", pool)
                    start = time.time()
                    conn = pool.connection()
                    cursor = conn.cursor()
                    # print('time taken to get connection:', time.time()-start)
                    if 'cursor' in kwargs:
                        kwargs['cursor'] = cursor
                    # print('conn', conn, 'cursor', cursor)
                    result = func(*args, **kwargs)
                    conn.commit()
                    return result
                except Exception as e:
                    print(f"Database error occurred2: {e}")
                    # print("Traceback:")
                    # print(traceback.format_exc())
                    retry_count += 1
                    if retry_count >= max_retries:
                        raise
                    # print(f"Retrying... (attempt {retry_count} of {max_retries})")
                    time.sleep(2 ** retry_count)
                finally:
                    if cursor:
                        try:
                            print('closing cursor, reconnect_on_failure')
                            cursor.close()
                        except:
                            print("Error closing cursor")
                    if conn:
                        try:
                            print('closing connection, reconnect_on_failure')
                            conn.close()
                        except:
                            print("Error closing connection")

    return wrapper_reconnect_on_failure


@reconnect_on_failure
def create_pdfs_table_if_not_exists():
    # print('cursor', cursor)
    cursor.execute("""CREATE TABLE IF NOT EXISTS pdfs (
    id INT AUTO_INCREMENT PRIMARY KEY,
    file LONGBLOB,
    date_detected TEXT,
    date_extracted TEXT,
    url TEXT,
    title TEXT,
    breadCrumb TEXT,
    checksum TEXT,
    library TEXT,
    username TEXT,
    UNIQUE KEY unique_pdf (url(255), title(255), library(255), username(255))
    );""")
    # print("pdfs table created")


def execute_queries():
    create_pdfs_table_if_not_exists(cursor=cursor)



if __name__ == '__main__':
    pass


# def connect_to_db():
#     global connection, tunnel
#     if tunnel is None or not tunnel.is_active:
#         tunnel = SSHTunnelForwarder(
#             (ssh_host, 22),
#             ssh_username=ssh_user,
#             ssh_private_key=ssh_private_key,
#             remote_bind_address=(db_host, db_port),
#             local_bind_address=('0.0.0.0', 0)
#         )
#         tunnel.start()
#         print(f"SSH tunnel established. Local bind port: {tunnel.local_bind_port}")
#
#     start = time.time()
#     if connection is None or not connection.open:
#         connection = pymysql.connect(
#             host='127.0.0.1',
#             port=tunnel.local_bind_port,
#             user=db_user,
#             password=db_password,
#             database=db_name,
#             connect_timeout=10,
#         )
#         print("Database conn established.")
#     end = time.time()
#     print(f"Time taken to connect to db: {end-start}")
#
#     return connection



