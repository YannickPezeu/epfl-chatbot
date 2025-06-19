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


def delete_lex_and_rh_from_all_tables(cursor) -> int:
    """Delete all rows with library = 'LEX AND RH' from all tables."""
    tables = get_all_tables(cursor)
    deletion_count = 0

    logger.info("Step 1: Deleting all rows with library = 'LEX AND RH' from all tables")
    for table in tables:
        if table_has_library_column(cursor, table):
            try:
                cursor.execute(f"DELETE FROM {table} WHERE library = 'LEX AND RH'")
                affected_rows = cursor.rowcount
                deletion_count += affected_rows
                logger.info(f"  - Deleted {affected_rows} rows from table '{table}'")
            except Exception as e:
                logger.error(f"  - Error deleting from table '{table}': {e}")
        else:
            logger.debug(f"  - Table '{table}' doesn't have a 'library' column, skipping")

    logger.info(f"Total rows deleted: {deletion_count}")
    return deletion_count


def get_table_columns(cursor, table_name: str) -> List[str]:
    """Get a list of all column names for the specified table."""
    cursor.execute(f"SHOW COLUMNS FROM {table_name}")
    return [column[0] for column in cursor.fetchall()]


def insert_lex_and_rh_to_source_docs(cursor) -> int:
    """
    Insert new rows into the source_docs table with library = 'LEX AND RH'
    for all existing rows with library = 'LEX' or library = 'RH'.
    Process each row individually for better error handling.
    """
    logger.info("Step 2: Inserting new rows with library = 'LEX AND RH' into source_docs table")

    # Check if source_docs table exists
    cursor.execute("SHOW TABLES LIKE 'source_docs'")
    if not cursor.fetchone():
        logger.warning("  - The source_docs table does not exist in the database")
        return 0

    # Check if source_docs table has library column
    if not table_has_library_column(cursor, 'source_docs'):
        logger.warning("  - The source_docs table does not have a 'library' column")
        return 0

    # Get all columns from the source_docs table
    all_columns = get_table_columns(cursor, 'source_docs')

    # Determine if the table has an auto-increment primary key
    cursor.execute("""
        SELECT COLUMN_NAME 
        FROM INFORMATION_SCHEMA.COLUMNS 
        WHERE TABLE_SCHEMA = DATABASE() 
        AND TABLE_NAME = 'source_docs' 
        AND EXTRA LIKE '%auto_increment%'
    """)
    auto_increment_columns = [column[0] for column in cursor.fetchall()]

    # If we have an auto-increment column, exclude it from the INSERT
    if auto_increment_columns:
        logger.info(f"  - Found auto-increment columns: {', '.join(auto_increment_columns)}")
        columns = [col for col in all_columns if col not in auto_increment_columns]
    else:
        columns = all_columns

    columns_str = ', '.join([f"`{col}`" for col in columns])

    # Let's debug what's in the unique constraint
    cursor.execute("""
        SELECT COLUMN_NAME
        FROM INFORMATION_SCHEMA.KEY_COLUMN_USAGE
        WHERE TABLE_SCHEMA = DATABASE()
        AND TABLE_NAME = 'source_docs'
        AND CONSTRAINT_NAME != 'PRIMARY'
    """)
    unique_columns = [column[0] for column in cursor.fetchall()]
    logger.info(f"  - Unique constraint columns: {unique_columns}")

    # Check if we already have any rows with "LEX AND RH"
    cursor.execute("SELECT COUNT(*) FROM source_docs WHERE library = 'LEX AND RH'")
    existing_count = cursor.fetchone()[0]
    logger.info(f"  - Existing 'LEX AND RH' rows: {existing_count}")

    # Get all distinct combinations of url, title, username for LEX and RH
    cursor.execute("""
        SELECT DISTINCT url, title, username 
        FROM source_docs 
        WHERE library = 'LEX' OR library = 'RH'
    """)
    distinct_combinations = cursor.fetchall()
    logger.info(f"  - Found {len(distinct_combinations)} distinct combinations to process")

    # First try: insert all the rows with library = LEX
    logger.info("  - First attempt: Insert rows from library = 'LEX'")
    try:
        cursor.execute(f"""
            INSERT INTO source_docs ({columns_str})
            SELECT {', '.join([f"`{col}`" if col != 'library' else "'LEX AND RH'" for col in columns])}
            FROM source_docs
            WHERE library = 'LEX'
        """)
        inserted_count = cursor.rowcount
        logger.info(f"  - Inserted {inserted_count} rows with LEX data")
    except Exception as e:
        logger.error(f"  - Error inserting LEX rows: {e}")
        inserted_count = 0

    # Second try: insert rows with library = RH that don't conflict with LEX
    logger.info("  - Second attempt: Insert rows from library = 'RH' where not conflicting")
    try:
        # Create a more complex query to avoid conflicts
        # We join with the LEX table to find RH rows that don't have a LEX counterpart
        cursor.execute(f"""
            INSERT INTO source_docs ({columns_str})
            SELECT {', '.join([f"rh.`{col}`" if col != 'library' else "'LEX AND RH'" for col in columns])}
            FROM source_docs rh
            LEFT JOIN (
                SELECT url, title, username FROM source_docs WHERE library = 'LEX'
            ) lex ON rh.url = lex.url AND rh.title = lex.title AND rh.username = lex.username
            WHERE rh.library = 'RH' AND lex.url IS NULL
        """)
        rh_inserted_count = cursor.rowcount
        logger.info(f"  - Inserted {rh_inserted_count} additional rows with RH data")
        inserted_count += rh_inserted_count
    except Exception as e:
        logger.error(f"  - Error inserting RH rows: {e}")

    return inserted_count


def delete_rh_library(cursor) -> int:
    """Delete all rows with library = 'RH' from all tables except source_docs."""
    tables = get_all_tables(cursor)
    deletion_count = 0

    logger.info("Step 3: Deleting all rows with library = 'RH' from all tables (except pdfs)")
    for table in tables:
        # Skip the pdfs table
        if table.lower() == 'pdfs':
            logger.info(f"  - Skipping table '{table}' to preserve pdfs")
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

    logger.info(f"Total RH rows deleted: {deletion_count}")
    return deletion_count


def main():
    """Main function to execute the database operations."""
    logger.info("MariaDB Data Migration - LEX AND RH Library Processing")
    logger.info("====================================================")

    try:
        # Get database connection using your existing method
        conn, cursor = initialize_all_connection()

        try:
            # Step 1: Delete all rows with library = "LEX AND RH" from all tables
            deleted_count = delete_lex_and_rh_from_all_tables(cursor)

            # Step 2: Insert new rows into pdfs table
            inserted_count = insert_lex_and_rh_to_source_docs(cursor)

            # Summary before Step 3
            logger.info("\nIntermediate Summary:")
            logger.info(f"  - Total rows deleted with 'LEX AND RH': {deleted_count}")
            logger.info(f"  - Total rows inserted with 'LEX AND RH': {inserted_count}")

            # Confirm to proceed with Step 3 (deleting RH)
            # proceed = input("\nDo you want to proceed with deleting all 'RH' entries (except pdfs)? (yes/no): ").lower()
            # if pyesroceed == 'yes' or proceed == 'y':
            #     # Step 3: Delete all rows with library = "RH" from all tables except pdfs
            #     rh_deleted_count = delete_rh_library(cursor)
            #     logger.info(f"  - Total 'RH' rows deleted: {rh_deleted_count}")
            # else:
            #     logger.info("Skipping the deletion of 'RH' entries")

            # Confirm changes
            confirm = input("\nCommit all changes to the database? (yes/no): ").lower()
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