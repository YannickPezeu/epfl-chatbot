import os
import time
import logging
import threading
from contextlib import contextmanager
from typing import Optional, Tuple, Any
import pymysql
from dbutils.pooled_db import PooledDB
from sshtunnel import SSHTunnelForwarder
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(message)s - %(filename)s:%(lineno)d'
)
logger = logging.getLogger(__name__)


class DatabaseConfig:
    def __init__(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.dirname(current_dir)
        ssh_dir = os.path.join(root_dir, 'ssh')

        self.db_config = {
            'host': os.getenv('FAC_HOST'),
            'port': int(os.getenv('FAC_PORT')),
            'user': os.getenv('FAC_USER'),
            'password': os.getenv('FAC_PASSWORD'),
            'database': 'acad',
        }

        self.ssh_config = {
            'ssh_host': os.getenv('FAC_TUNNEL_HOST'),
            'ssh_user': os.getenv('FAC_TUNNEL_USER'),
            'ssh_private_key': os.path.join(ssh_dir, os.getenv('FAC_TUNNEL_KEY')),
        }


class DatabaseManager:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(DatabaseManager, cls).__new__(cls)
                    cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """Initialize database manager"""
        self.config = DatabaseConfig()
        self.pool = None
        self.tunnel = None
        self.connection_count = 0
        self.max_connections = 10
        self.setup_ssh_tunnel()
        self.initialize_pool()

    def setup_ssh_tunnel(self) -> None:
        """Setup SSH tunnel if not already active"""
        if self.tunnel is None or not self.tunnel.is_active:
            try:
                self.tunnel = SSHTunnelForwarder(
                    (self.config.ssh_config['ssh_host'], 22),
                    ssh_username=self.config.ssh_config['ssh_user'],
                    ssh_private_key=self.config.ssh_config['ssh_private_key'],
                    remote_bind_address=(self.config.db_config['host'], self.config.db_config['port']),
                    local_bind_address=('0.0.0.0', 0)
                )
                self.tunnel.start()
                logger.info(f"SSH tunnel established on port {self.tunnel.local_bind_port}")
            except Exception as e:
                logger.error(f"Failed to establish SSH tunnel: {e}")
                raise

    def initialize_pool(self) -> None:
        """Initialize the connection pool"""
        if self.pool is None:
            try:
                self.pool = PooledDB(
                    creator=pymysql,
                    maxconnections=self.max_connections,
                    mincached=2,
                    maxcached=5,
                    maxshared=3,
                    blocking=True,
                    maxusage=10000,
                    setsession=[],
                    host='127.0.0.1',
                    port=self.tunnel.local_bind_port,
                    user=self.config.db_config['user'],
                    password=self.config.db_config['password'],
                    database=self.config.db_config['database'],
                    reset=True,
                    ping=1
                )
                logger.info("Database pool initialized")
            except Exception as e:
                logger.error(f"Failed to initialize connection pool: {e}")
                raise

    def reset_pool(self) -> None:
        """Reset the connection pool"""
        with self._lock:
            if self.pool:
                try:
                    self.pool.close()
                except Exception as e:
                    logger.error(f"Error closing pool: {e}")
                finally:
                    self.pool = None
                    self.initialize_pool()
                    logger.info("Connection pool reset")

    def get_connection(self) -> Tuple[Any, Any]:
        """Get a connection from the pool"""
        if self.connection_count >= self.max_connections:
            logger.warning("Maximum connections reached, resetting pool")
            self.reset_pool()

        conn = self.pool.connection()
        cursor = conn.cursor()
        self.connection_count += 1
        return conn, cursor

    def get_pool_status(self) -> dict:
        """Get current pool status"""
        if self.pool:
            return {
                "active_connections": self.connection_count,
                "max_connections": self.max_connections,
                "pool_connections": self.pool._connections
            }
        return {}


# Create singleton instance
db_manager = DatabaseManager()


@contextmanager
def get_db_connection():
    """Context manager for database connections"""
    conn = None
    cursor = None
    start_time = time.time()

    try:
        logger.debug("Getting database connection...")
        conn, cursor = db_manager.get_connection()
        logger.info(f"Connection established in {time.time() - start_time:.2f}s")
        yield conn, cursor
    except Exception as e:
        logger.error(f"Database connection error: {e}", exc_info=True)
        raise
    finally:
        cleanup_start = time.time()
        if cursor:
            try:
                cursor.close()
                logger.debug("Cursor closed")
            except Exception as e:
                logger.error(f"Error closing cursor: {e}")
        if conn:
            try:
                conn.close()
                db_manager.connection_count -= 1
                logger.debug(f"Connection closed in {time.time() - cleanup_start:.2f}s")
            except Exception as e:
                logger.error(f"Error closing connection: {e}")
        logger.debug(f"Total connection handling time: {time.time() - start_time:.2f}s")


def initialize_all_connection() -> Tuple[Any, Any]:
    """Legacy function for backward compatibility"""
    return db_manager.get_connection()


def check_database_health() -> bool:
    """Check if database connection is healthy"""
    try:
        with get_db_connection() as (conn, cursor):
            cursor.execute("SELECT 1")
            return True
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return False


import functools
import asyncio
import time
import logging
from typing import Any, Callable, TypeVar, Union, Awaitable

logger = logging.getLogger(__name__)

T = TypeVar('T')


def reconnect_on_failure(func: Callable[..., Union[T, Awaitable[T]]]) -> Callable[..., Union[T, Awaitable[T]]]:
    """Decorator to handle database reconnection for both sync and async functions"""

    @functools.wraps(func)
    async def async_wrapper(*args: Any, **kwargs: Any) -> T:
        max_retries = 1
        retry_count = 0
        last_error = None

        try:
            return await func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Database error in async function {func.__name__}: {e}")
            last_error = e

            while retry_count < max_retries:
                try:
                    logger.info(f"Retry attempt {retry_count + 1} for {func.__name__}")

                    # Get fresh connection from pool
                    with get_db_connection() as (conn, cursor):
                        if 'cursor' in kwargs:
                            kwargs['cursor'] = cursor
                        if 'conn' in kwargs:
                            kwargs['conn'] = conn

                        result = await func(*args, **kwargs)
                        if not kwargs.get('no_commit'):
                            conn.commit()
                        return result

                except Exception as retry_error:
                    logger.error(f"Retry {retry_count + 1} failed: {retry_error}")
                    last_error = retry_error
                    retry_count += 1

                    if retry_count >= max_retries:
                        logger.error(f"Max retries ({max_retries}) reached for {func.__name__}")
                        raise last_error

                    await asyncio.sleep(2 ** retry_count)

            raise last_error

    @functools.wraps(func)
    def sync_wrapper(*args: Any, **kwargs: Any) -> T:
        max_retries = 1
        retry_count = 0
        last_error = None

        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Database error in sync function {func.__name__}: {e}")
            last_error = e

            while retry_count < max_retries:
                try:
                    logger.info(f"Retry attempt {retry_count + 1} for {func.__name__}")

                    # Get fresh connection from pool
                    with get_db_connection() as (conn, cursor):
                        if 'cursor' in kwargs:
                            kwargs['cursor'] = cursor
                        if 'conn' in kwargs:
                            kwargs['conn'] = conn

                        result = func(*args, **kwargs)
                        if not kwargs.get('no_commit'):
                            conn.commit()
                        return result

                except Exception as retry_error:
                    logger.error(f"Retry {retry_count + 1} failed: {retry_error}")
                    last_error = retry_error
                    retry_count += 1

                    if retry_count >= max_retries:
                        logger.error(f"Max retries ({max_retries}) reached for {func.__name__}")
                        raise last_error

                    time.sleep(2 ** retry_count)

            raise last_error

    return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper


# Example usage:
@reconnect_on_failure
async def async_database_operation(cursor=None, conn=None):
    """Example async database operation"""
    # Your async code here
    pass


@reconnect_on_failure
def sync_database_operation(cursor=None, conn=None):
    """Example sync database operation"""
    # Your sync code here
    pass


