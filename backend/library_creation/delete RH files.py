#!/usr/bin/env python3
import logging
from typing import List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

# Import your database connection
from myUtils.connect_acad2 import initialize_all_connection


def get_all_tables(cursor) -> List[str]:
    """Get a list of all tables in the database."""
    cursor.execute("SHOW TABLES")
    return [table[0] for table in cursor.fetchall()]


def table_has_library_column(cursor, table_name: str) -> bool:
    """Check if the specified table has a 'library' column."""
    cursor.execute(f"""
        SELECT COUNT(*) 
        FROM INFORMATION_SCHEMA.COLUMNS 
        WHERE TABLE_SCHEMA = DATABASE() 
        AND TABLE_NAME = '{table_name}' 
        AND COLUMN_NAME = 'library'
    """)
    return cursor.fetchone()[0] > 0


def delete_rh_from_tables(cursor) -> int:
    """Delete all rows with library = 'RH' from all tables except source_docs."""
    tables = get_all_tables(cursor)
    deletion_count = 0

    logger.info("Deleting all rows with library = 'RH' from all tables (except source_docs)")
    for table in tables:
        # Skip the source_docs table
        if table.lower() == 'source_docs':
            logger.info(f"  - Skipping table '{table}' as requested")
            continue

        if table_has_library_column(cursor, table):
            try:
                # First check how many rows will be affected
                cursor.execute(f"SELECT COUNT(*) FROM {table} WHERE library = 'RH'")
                row_count = cursor.fetchone()[0]

                if row_count > 0:
                    cursor.execute(f"DELETE FROM {table} WHERE library = 'RH'")
                    affected_rows = cursor.rowcount
                    deletion_count += affected_rows
                    logger.info(f"  - Deleted {affected_rows} rows from table '{table}'")
                else:
                    logger.debug(f"  - No 'RH' rows in table '{table}', skipping")
            except Exception as e:
                logger.error(f"  - Error deleting from table '{table}': {e}")
        else:
            logger.debug(f"  - Table '{table}' doesn't have a 'library' column, skipping")

    logger.info(f"Total rows deleted: {deletion_count}")
    return deletion_count


def main():
    """Main function to execute the database operations."""
    logger.info("Database Cleanup - Delete RH Library Entries")
    logger.info("===========================================")

    try:
        # Get database connection using your existing method
        conn, cursor = initialize_all_connection()

        try:
            # Delete rows with library = "RH" from all tables except source_docs
            deleted_count = delete_rh_from_tables(cursor)

            # Summary
            logger.info("\nOperation Summary:")
            logger.info(f"  - Total rows deleted: {deleted_count}")

            # Confirm changes
            confirm = input("\nCommit these changes to the database? (yes/no): ").lower()
            if confirm == 'yes' or confirm == 'y':
                conn.commit()
                logger.info("Changes committed successfully")
            else:
                conn.rollback()
                logger.info("Changes rolled back, no modifications were made to the database")

        except Exception as e:
            conn.rollback()
            logger.error(f"An error occurred: {e}", exc_info=True)
            logger.info("All changes have been rolled back")

        finally:
            # Close cursor and connection
            cursor.close()
            conn.close()
            logger.info("Database connection closed")

    except Exception as e:
        logger.error(f"Error establishing database connection: {e}", exc_info=True)


if __name__ == "__main__":
    main()