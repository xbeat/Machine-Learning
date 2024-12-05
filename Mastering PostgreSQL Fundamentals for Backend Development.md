## Mastering PostgreSQL Fundamentals for Backend Development
Slide 1: PostgreSQL Database Connection with Python

Python's psycopg2 library serves as a robust PostgreSQL adapter, enabling seamless interaction between Python applications and PostgreSQL databases. This fundamental connection setup establishes the foundation for all subsequent database operations and implements crucial error handling mechanisms.

```python
import psycopg2
from psycopg2 import Error

def create_db_connection():
    try:
        # Connection parameters
        connection = psycopg2.connect(
            user="your_username",
            password="your_password",
            host="127.0.0.1",
            port="5432",
            database="your_database"
        )
        
        # Create cursor object for executing queries
        cursor = connection.cursor()
        
        # Print PostgreSQL details
        print("Connected to PostgreSQL:")
        print(connection.get_dsn_parameters())
        
        return connection, cursor
        
    except (Exception, Error) as error:
        print(f"Error connecting to PostgreSQL: {error}")
        return None, None

# Usage example
connection, cursor = create_db_connection()
if connection:
    cursor.close()
    connection.close()
    print("Database connection closed.")

# Output:
# Connected to PostgreSQL:
# {'dbname': 'your_database', 'user': 'your_username', ...}
# Database connection closed.
```

Slide 2: Creating Tables and Data Types

PostgreSQL offers a comprehensive set of data types and constraints for table creation. This implementation demonstrates the creation of a complex table structure utilizing various data types, primary keys, foreign keys, and check constraints to maintain data integrity.

```python
def create_complex_table(cursor, connection):
    try:
        # Create table with various PostgreSQL data types
        create_table_query = """
        CREATE TABLE employee_records (
            id SERIAL PRIMARY KEY,
            first_name VARCHAR(50) NOT NULL,
            last_name VARCHAR(50) NOT NULL,
            email VARCHAR(100) UNIQUE,
            birth_date DATE CHECK (birth_date > '1900-01-01'),
            salary DECIMAL(10,2) CHECK (salary >= 0),
            department_id INTEGER,
            skills TEXT[],
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            is_active BOOLEAN DEFAULT true
        );
        """
        
        cursor.execute(create_table_query)
        connection.commit()
        print("Table created successfully")
        
    except (Exception, Error) as error:
        print(f"Error creating table: {error}")
        connection.rollback()

# Example usage
connection, cursor = create_db_connection()
if connection:
    create_complex_table(cursor, connection)
    cursor.close()
    connection.close()
```

Slide 3: Data Insertion and Batch Processing

Efficient data insertion strategies are crucial for database performance. This implementation showcases both single-row and batch insertion methods, utilizing parameterized queries to prevent SQL injection and optimize database operations.

```python
def insert_employee_data(cursor, connection):
    try:
        # Single row insertion
        single_insert_query = """
        INSERT INTO employee_records 
        (first_name, last_name, email, birth_date, salary, department_id, skills)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        RETURNING id;
        """
        
        record = ('John', 'Doe', 'john@example.com', 
                 '1990-01-15', 75000.00, 1, 
                 ['Python', 'SQL', 'Data Analysis'])
        
        cursor.execute(single_insert_query, record)
        
        # Batch insertion
        batch_insert_query = """
        INSERT INTO employee_records 
        (first_name, last_name, email, birth_date, salary, department_id, skills)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        """
        
        batch_records = [
            ('Jane', 'Smith', 'jane@example.com', 
             '1992-03-20', 82000.00, 2, 
             ['Java', 'Spring', 'MySQL']),
            ('Mike', 'Johnson', 'mike@example.com', 
             '1988-07-10', 95000.00, 1, 
             ['Python', 'Django', 'PostgreSQL'])
        ]
        
        cursor.executemany(batch_insert_query, batch_records)
        connection.commit()
        
        print("Data inserted successfully")
        
    except (Exception, Error) as error:
        print(f"Error inserting data: {error}")
        connection.rollback()

# Example usage
connection, cursor = create_db_connection()
if connection:
    insert_employee_data(cursor, connection)
    cursor.close()
    connection.close()
```

Slide 4: Advanced Querying and Data Retrieval

PostgreSQL's powerful querying capabilities enable complex data retrieval operations. This implementation demonstrates advanced querying techniques including joins, aggregations, window functions, and complex filtering conditions.

```python
def perform_advanced_queries(cursor):
    try:
        # Complex query with JOIN, GROUP BY, and Window Functions
        advanced_query = """
        WITH dept_stats AS (
            SELECT 
                department_id,
                AVG(salary) as avg_dept_salary,
                COUNT(*) as employee_count
            FROM employee_records
            GROUP BY department_id
        )
        SELECT 
            e.first_name,
            e.last_name,
            e.salary,
            d.avg_dept_salary,
            d.employee_count,
            RANK() OVER (PARTITION BY e.department_id 
                        ORDER BY e.salary DESC) as salary_rank
        FROM employee_records e
        JOIN dept_stats d ON e.department_id = d.department_id
        WHERE e.is_active = true
        ORDER BY e.department_id, salary_rank;
        """
        
        cursor.execute(advanced_query)
        results = cursor.fetchall()
        
        # Process and display results
        for row in results:
            print(f"""
            Employee: {row[0]} {row[1]}
            Salary: ${row[2]:,.2f}
            Dept Avg: ${row[3]:,.2f}
            Dept Size: {row[4]}
            Salary Rank: {row[5]}
            """)
            
    except (Exception, Error) as error:
        print(f"Error executing query: {error}")

# Example usage
connection, cursor = create_db_connection()
if connection:
    perform_advanced_queries(cursor)
    cursor.close()
    connection.close()
```

Slide 5: Database Transactions and ACID Properties

Transaction management in PostgreSQL ensures data consistency and integrity through ACID properties. This implementation demonstrates proper transaction handling with commit, rollback, and savepoint operations for complex multi-step database operations.

```python
def handle_complex_transaction(cursor, connection):
    try:
        # Start transaction
        connection.autocommit = False
        
        # First operation: Update salaries
        cursor.execute("""
        UPDATE employee_records 
        SET salary = salary * 1.1 
        WHERE department_id = 1 
        RETURNING id, first_name, salary;
        """)
        
        # Create savepoint
        cursor.execute("SAVEPOINT salary_update;")
        
        # Second operation: Insert new department assignments
        try:
            cursor.execute("""
            INSERT INTO employee_records 
            (first_name, last_name, email, birth_date, salary, department_id)
            VALUES 
            ('Alex', 'Wilson', 'alex@example.com', '1995-05-15', 70000, 1);
            """)
            
            # Verify conditions before committing
            cursor.execute("""
            SELECT COUNT(*) 
            FROM employee_records 
            WHERE department_id = 1;
            """)
            
            if cursor.fetchone()[0] > 10:
                # Rollback to savepoint if department too large
                cursor.execute("ROLLBACK TO SAVEPOINT salary_update;")
                print("Rolled back to savepoint: department size limit reached")
            else:
                # Commit transaction
                connection.commit()
                print("Transaction completed successfully")
                
        except Exception as e:
            cursor.execute("ROLLBACK TO SAVEPOINT salary_update;")
            print(f"Error in second operation: {e}")
            
    except (Exception, Error) as error:
        connection.rollback()
        print(f"Transaction failed: {error}")
    
    finally:
        connection.autocommit = True

# Example usage
connection, cursor = create_db_connection()
if connection:
    handle_complex_transaction(cursor, connection)
    cursor.close()
    connection.close()
```

Slide 6: Full-Text Search Implementation

PostgreSQL's full-text search capabilities provide powerful document indexing and searching functionality. This implementation demonstrates the creation and usage of text search vectors, custom dictionaries, and ranking functions.

```python
def implement_full_text_search(cursor, connection):
    try:
        # Create text search configuration
        cursor.execute("""
        CREATE EXTENSION IF NOT EXISTS unaccent;
        
        CREATE TEXT SEARCH CONFIGURATION custom_search (
            COPY = pg_catalog.english
        );
        
        ALTER TEXT SEARCH CONFIGURATION custom_search
        ALTER MAPPING FOR hword, hword_part, word
        WITH unaccent, english_stem;
        """)
        
        # Create table with text search vectors
        cursor.execute("""
        CREATE TABLE documents (
            id SERIAL PRIMARY KEY,
            title TEXT,
            content TEXT,
            search_vector TSVECTOR GENERATED ALWAYS AS (
                setweight(to_tsvector('custom_search', coalesce(title, '')), 'A') ||
                setweight(to_tsvector('custom_search', coalesce(content, '')), 'B')
            ) STORED
        );
        
        CREATE INDEX idx_documents_search 
        ON documents USING GIN(search_vector);
        """)
        
        # Example search function
        def search_documents(query_text):
            search_query = """
            SELECT 
                id,
                title,
                ts_rank_cd(search_vector, query) AS rank
            FROM 
                documents,
                plainto_tsquery('custom_search', %s) query
            WHERE 
                search_vector @@ query
            ORDER BY rank DESC
            LIMIT 10;
            """
            cursor.execute(search_query, (query_text,))
            return cursor.fetchall()
        
        # Example usage
        cursor.execute("""
        INSERT INTO documents (title, content)
        VALUES 
        ('PostgreSQL Tutorial', 'Advanced database management system tutorial'),
        ('Database Design', 'Best practices for designing SQL databases');
        """)
        
        results = search_documents('database management')
        for doc_id, title, rank in results:
            print(f"Document ID: {doc_id}")
            print(f"Title: {title}")
            print(f"Rank: {rank}\n")
            
        connection.commit()
        
    except (Exception, Error) as error:
        connection.rollback()
        print(f"Error in full-text search implementation: {error}")

# Example usage
connection, cursor = create_db_connection()
if connection:
    implement_full_text_search(cursor, connection)
    cursor.close()
    connection.close()
```

Slide 7: Advanced Indexing Strategies

Proper indexing is crucial for query performance optimization. This implementation explores various indexing techniques including B-tree, Hash, GiST, and partial indexes, along with index maintenance and performance monitoring.

```python
def implement_advanced_indexing(cursor, connection):
    try:
        # Create table with various index types
        cursor.execute("""
        CREATE TABLE product_inventory (
            id SERIAL PRIMARY KEY,
            product_code VARCHAR(50),
            name VARCHAR(100),
            price DECIMAL(10,2),
            location POINT,
            tags TEXT[],
            last_updated TIMESTAMP
        );
        
        -- B-tree index for exact matches and range queries
        CREATE INDEX idx_product_price 
        ON product_inventory(price);
        
        -- Hash index for equality comparisons
        CREATE INDEX idx_product_code_hash 
        ON product_inventory USING HASH (product_code);
        
        -- GiST index for geometric data
        CREATE INDEX idx_product_location 
        ON product_inventory USING GIST (location);
        
        -- Partial index for active products
        CREATE INDEX idx_active_products 
        ON product_inventory(last_updated) 
        WHERE price > 0;
        
        -- Expression index for case-insensitive searches
        CREATE INDEX idx_product_name_lower 
        ON product_inventory(LOWER(name));
        """)
        
        # Function to analyze index usage
        def analyze_index_usage():
            cursor.execute("""
            SELECT 
                schemaname,
                tablename,
                indexname,
                idx_scan,
                idx_tup_read,
                idx_tup_fetch
            FROM 
                pg_stat_user_indexes
            WHERE 
                schemaname = 'public'
            ORDER BY 
                idx_scan DESC;
            """)
            return cursor.fetchall()
        
        # Insert sample data
        cursor.execute("""
        INSERT INTO product_inventory 
        (product_code, name, price, location, tags, last_updated)
        VALUES 
        ('P001', 'Laptop', 999.99, POINT(40.7128, -74.0060), 
         ARRAY['electronics', 'computers'], CURRENT_TIMESTAMP),
        ('P002', 'Smartphone', 699.99, POINT(34.0522, -118.2437), 
         ARRAY['electronics', 'mobile'], CURRENT_TIMESTAMP);
        """)
        
        connection.commit()
        
        # Analyze index performance
        index_stats = analyze_index_usage()
        for stat in index_stats:
            print(f"""
            Index: {stat[2]}
            Table: {stat[1]}
            Scans: {stat[3]}
            Tuples Read: {stat[4]}
            Tuples Fetched: {stat[5]}
            """)
            
    except (Exception, Error) as error:
        connection.rollback()
        print(f"Error in indexing implementation: {error}")

# Example usage
connection, cursor = create_db_connection()
if connection:
    implement_advanced_indexing(cursor, connection)
    cursor.close()
    connection.close()
```

Slide 8: Database Partitioning and Sharding

PostgreSQL's partitioning functionality enables efficient management of large datasets by breaking tables into smaller, more manageable pieces. This implementation demonstrates table partitioning strategies including range, list, and hash partitioning methods.

```python
def implement_table_partitioning(cursor, connection):
    try:
        # Create partitioned table
        cursor.execute("""
        CREATE TABLE sales_data (
            id SERIAL,
            sale_date DATE NOT NULL,
            amount DECIMAL(10,2),
            region VARCHAR(50),
            product_id INTEGER
        ) PARTITION BY RANGE (sale_date);
        
        -- Create partitions for different date ranges
        CREATE TABLE sales_2023 PARTITION OF sales_data
        FOR VALUES FROM ('2023-01-01') TO ('2024-01-01');
        
        CREATE TABLE sales_2024 PARTITION OF sales_data
        FOR VALUES FROM ('2024-01-01') TO ('2025-01-01');
        
        -- Create indexes on partitions
        CREATE INDEX idx_sales_2023_date ON sales_2023(sale_date);
        CREATE INDEX idx_sales_2024_date ON sales_2024(sale_date);
        """)
        
        # Function to insert test data
        def insert_test_data():
            cursor.execute("""
            INSERT INTO sales_data (sale_date, amount, region, product_id)
            SELECT 
                generate_series(
                    '2023-01-01'::date,
                    '2024-12-31'::date,
                    '1 day'::interval
                ) AS sale_date,
                random() * 1000 AS amount,
                (ARRAY['North', 'South', 'East', 'West'])[ceil(random() * 4)] AS region,
                ceil(random() * 100)::int AS product_id;
            """)
        
        # Insert test data and analyze partition usage
        insert_test_data()
        connection.commit()
        
        cursor.execute("""
        SELECT 
            tablename, 
            pg_size_pretty(pg_total_relation_size(schemaname || '.' || tablename)) as size
        FROM pg_tables
        WHERE tablename LIKE 'sales_%'
        ORDER BY tablename;
        """)
        
        partition_stats = cursor.fetchall()
        print("\nPartition Statistics:")
        for stat in partition_stats:
            print(f"Partition: {stat[0]}, Size: {stat[1]}")
            
    except (Exception, Error) as error:
        connection.rollback()
        print(f"Error in partitioning implementation: {error}")

# Example usage
connection, cursor = create_db_connection()
if connection:
    implement_table_partitioning(cursor, connection)
    cursor.close()
    connection.close()
```

Slide 9: Materialized Views and Performance Optimization

Materialized views provide enhanced query performance by storing the results of complex queries and supporting periodic refreshes. This implementation showcases the creation and management of materialized views with automatic refresh mechanisms.

```python
def implement_materialized_views(cursor, connection):
    try:
        # Create base tables
        cursor.execute("""
        CREATE TABLE sales_transactions (
            id SERIAL PRIMARY KEY,
            product_id INTEGER,
            sale_amount DECIMAL(10,2),
            sale_date DATE,
            customer_id INTEGER
        );
        
        -- Create materialized view for sales analytics
        CREATE MATERIALIZED VIEW sales_summary AS
        SELECT 
            date_trunc('month', sale_date) as month,
            COUNT(*) as total_transactions,
            SUM(sale_amount) as total_revenue,
            AVG(sale_amount) as avg_transaction_value,
            COUNT(DISTINCT customer_id) as unique_customers
        FROM sales_transactions
        GROUP BY date_trunc('month', sale_date)
        WITH NO DATA;
        
        -- Create unique index to support concurrent refresh
        CREATE UNIQUE INDEX idx_sales_summary_month 
        ON sales_summary(month);
        """)
        
        # Function to refresh materialized view
        def refresh_sales_summary(concurrent=True):
            if concurrent:
                cursor.execute("""
                REFRESH MATERIALIZED VIEW CONCURRENTLY sales_summary;
                """)
            else:
                cursor.execute("""
                REFRESH MATERIALIZED VIEW sales_summary;
                """)
            
        # Insert sample data and refresh view
        cursor.execute("""
        INSERT INTO sales_transactions 
        (product_id, sale_amount, sale_date, customer_id)
        SELECT 
            ceil(random() * 100),
            random() * 1000,
            current_date - (random() * 365)::integer,
            ceil(random() * 1000)
        FROM generate_series(1, 10000);
        """)
        
        refresh_sales_summary()
        
        # Query materialized view
        cursor.execute("""
        SELECT 
            to_char(month, 'YYYY-MM') as month,
            total_transactions,
            round(total_revenue::numeric, 2) as total_revenue,
            round(avg_transaction_value::numeric, 2) as avg_value,
            unique_customers
        FROM sales_summary
        ORDER BY month DESC;
        """)
        
        results = cursor.fetchall()
        print("\nSales Summary Report:")
        for row in results:
            print(f"""
            Month: {row[0]}
            Transactions: {row[1]}
            Revenue: ${row[2]:,.2f}
            Avg Value: ${row[3]:,.2f}
            Unique Customers: {row[4]}
            """)
            
        connection.commit()
        
    except (Exception, Error) as error:
        connection.rollback()
        print(f"Error in materialized view implementation: {error}")

# Example usage
connection, cursor = create_db_connection()
if connection:
    implement_materialized_views(cursor, connection)
    cursor.close()
    connection.close()
```

Slide 10: Database Monitoring and Performance Analysis

PostgreSQL's system catalogs and statistics collectors provide crucial insights into database performance. This implementation creates a comprehensive monitoring system that tracks query execution, index usage, and system resource utilization.

```python
def implement_performance_monitoring(cursor, connection):
    try:
        # Create monitoring functions
        cursor.execute("""
        CREATE OR REPLACE FUNCTION get_database_stats() 
        RETURNS TABLE (
            stat_name TEXT,
            stat_value BIGINT
        ) AS $$
        BEGIN
            RETURN QUERY
            SELECT 
                'total_connections'::TEXT,
                count(*)::BIGINT
            FROM 
                pg_stat_activity
            UNION ALL
            SELECT
                'active_queries'::TEXT,
                count(*)::BIGINT
            FROM 
                pg_stat_activity
            WHERE state = 'active';
        END;
        $$ LANGUAGE plpgsql;
        
        -- Create monitoring views
        CREATE OR REPLACE VIEW query_performance_stats AS
        SELECT
            queryid,
            calls,
            total_exec_time / 1000 as total_seconds,
            mean_exec_time / 1000 as mean_seconds,
            rows,
            shared_blks_hit + shared_blks_read as total_blocks
        FROM pg_stat_statements
        WHERE queryid IS NOT NULL
        ORDER BY total_exec_time DESC;
        """)
        
        def collect_performance_metrics():
            # Query execution statistics
            cursor.execute("""
            SELECT * FROM query_performance_stats LIMIT 5;
            """)
            
            query_stats = cursor.fetchall()
            print("\nTop 5 Time-Consuming Queries:")
            for stat in query_stats:
                print(f"""
                Query ID: {stat[0]}
                Calls: {stat[1]}
                Total Time: {stat[2]:.2f} seconds
                Mean Time: {stat[3]:.2f} seconds
                Rows Processed: {stat[4]}
                Blocks Accessed: {stat[5]}
                """)
            
            # Table statistics
            cursor.execute("""
            SELECT
                relname as table_name,
                seq_scan,
                idx_scan,
                n_live_tup as live_rows,
                n_dead_tup as dead_rows
            FROM pg_stat_user_tables
            ORDER BY n_live_tup DESC;
            """)
            
            table_stats = cursor.fetchall()
            print("\nTable Statistics:")
            for stat in table_stats:
                print(f"""
                Table: {stat[0]}
                Sequential Scans: {stat[1]}
                Index Scans: {stat[2]}
                Live Rows: {stat[3]}
                Dead Rows: {stat[4]}
                """)
        
        # Example monitoring loop
        collect_performance_metrics()
        
    except (Exception, Error) as error:
        print(f"Error in monitoring implementation: {error}")

# Example usage
connection, cursor = create_db_connection()
if connection:
    implement_performance_monitoring(cursor, connection)
    cursor.close()
    connection.close()
```

Slide 11: Async Database Operations with Python

Asynchronous database operations enable efficient handling of multiple concurrent database connections. This implementation demonstrates async patterns using asyncpg for high-performance PostgreSQL interaction.

```python
import asyncio
import asyncpg
from datetime import datetime

async def implement_async_operations():
    try:
        # Create connection pool
        pool = await asyncpg.create_pool(
            user='your_username',
            password='your_password',
            database='your_database',
            host='127.0.0.1',
            min_size=5,
            max_size=20
        )
        
        async def process_batch(batch_data):
            async with pool.acquire() as connection:
                async with connection.transaction():
                    # Prepare statement
                    stmt = await connection.prepare("""
                        INSERT INTO async_operations 
                        (data, processed_at) 
                        VALUES ($1, $2)
                        RETURNING id
                    """)
                    
                    return await stmt.fetch(
                        batch_data,
                        datetime.now()
                    )
        
        # Create test table
        async with pool.acquire() as connection:
            await connection.execute("""
                CREATE TABLE IF NOT EXISTS async_operations (
                    id SERIAL PRIMARY KEY,
                    data JSONB,
                    processed_at TIMESTAMP
                )
            """)
        
        # Process multiple batches concurrently
        test_data = [
            {'batch': i, 'items': list(range(5))}
            for i in range(10)
        ]
        
        tasks = [
            process_batch(data)
            for data in test_data
        ]
        
        results = await asyncio.gather(*tasks)
        
        print("\nProcessed Batches:")
        for i, result in enumerate(results):
            print(f"Batch {i}: {len(result)} records inserted")
        
        # Cleanup
        await pool.close()
        
    except Exception as error:
        print(f"Error in async operations: {error}")

# Example usage
asyncio.run(implement_async_operations())
```

Slide 12: Real-time Data Processing with PostgreSQL LISTEN/NOTIFY

PostgreSQL's LISTEN/NOTIFY mechanism enables real-time data processing and event-driven architectures. This implementation demonstrates building a reactive system that responds to database events instantly.

```python
import select
import json
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

def implement_realtime_processing(cursor, connection):
    try:
        # Set isolation level for NOTIFY
        connection.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        
        # Create trigger function
        cursor.execute("""
        CREATE OR REPLACE FUNCTION notify_data_change()
        RETURNS trigger AS $$
        BEGIN
            PERFORM pg_notify(
                'data_change',
                json_build_object(
                    'table', TG_TABLE_NAME,
                    'type', TG_OP,
                    'row_id', NEW.id,
                    'data', row_to_json(NEW)
                )::text
            );
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
        
        -- Create test table with trigger
        CREATE TABLE IF NOT EXISTS realtime_data (
            id SERIAL PRIMARY KEY,
            data_type VARCHAR(50),
            content JSONB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        -- Attach trigger
        CREATE TRIGGER realtime_data_trigger
            AFTER INSERT OR UPDATE
            ON realtime_data
            FOR EACH ROW
            EXECUTE FUNCTION notify_data_change();
        """)
        
        # Listen for notifications
        cursor.execute("LISTEN data_change;")
        
        def process_notification(notify):
            payload = json.loads(notify.payload)
            print(f"""
            Event Received:
            Operation: {payload['type']}
            Table: {payload['table']}
            Row ID: {payload['row_id']}
            Data: {json.dumps(payload['data'], indent=2)}
            """)
        
        # Insert test data
        cursor.execute("""
        INSERT INTO realtime_data (data_type, content)
        VALUES (
            'sensor_reading',
            '{"temperature": 25.6, "humidity": 65, "location": "Room A"}'::jsonb
        );
        """)
        
        # Check for notifications
        if select.select([connection], [], [], 5) != ([], [], []):
            connection.poll()
            while connection.notifies:
                process_notification(connection.notifies.pop())
                
    except (Exception, Error) as error:
        print(f"Error in realtime processing: {error}")

# Example usage
connection, cursor = create_db_connection()
if connection:
    implement_realtime_processing(cursor, connection)
    cursor.close()
    connection.close()
```

Slide 13: PostgreSQL Extensions and Custom Functions

PostgreSQL's extensibility enables creation of custom functions and data types. This implementation demonstrates building complex custom functions using PL/pgSQL and integrating with Python user-defined functions.

```python
def implement_custom_extensions(cursor, connection):
    try:
        # Create custom aggregate function
        cursor.execute("""
        CREATE OR REPLACE FUNCTION array_to_histogram(numeric[])
        RETURNS TABLE (
            bucket_range text,
            count bigint
        ) AS $$
        BEGIN
            RETURN QUERY
            WITH bucket_edges AS (
                SELECT
                    width_bucket(
                        unnest($1),
                        array_min($1),
                        array_max($1),
                        10
                    ) as bucket_num,
                    array_min($1) + (array_max($1) - array_min($1))/10 * generate_series(0, 10) as edge
            )
            SELECT
                '[' || round(edge::numeric, 2)::text || ' - ' ||
                round((edge + (array_max($1) - array_min($1))/10)::numeric, 2)::text || ')',
                count(*)
            FROM bucket_edges
            GROUP BY edge
            ORDER BY edge;
        END;
        $$ LANGUAGE plpgsql;
        
        -- Create custom type
        cursor.execute("""
        CREATE TYPE geo_location AS (
            latitude decimal,
            longitude decimal,
            altitude decimal
        );
        
        -- Create function using custom type
        CREATE OR REPLACE FUNCTION calculate_distance(
            point1 geo_location,
            point2 geo_location
        )
        RETURNS decimal AS $$
        DECLARE
            R constant decimal := 6371000; -- Earth radius in meters
        BEGIN
            RETURN (2 * R * asin(sqrt(
                power(sin((radians(point2.latitude) - radians(point1.latitude))/2), 2) +
                cos(radians(point1.latitude)) * cos(radians(point2.latitude)) *
                power(sin((radians(point2.longitude) - radians(point1.longitude))/2), 2)
            )));
        END;
        $$ LANGUAGE plpgsql;
        """)
        
        # Test custom functions
        cursor.execute("""
        SELECT * FROM array_to_histogram(ARRAY[1,2,2,3,3,3,4,4,5,6,7,8,9,10]);
        """)
        
        histogram_results = cursor.fetchall()
        print("\nHistogram Results:")
        for bucket in histogram_results:
            print(f"Range {bucket[0]}: {bucket[1]} items")
        
        # Test distance calculation
        cursor.execute("""
        SELECT calculate_distance(
            ROW(40.7128, -74.0060, 0)::geo_location,  -- New York
            ROW(51.5074, -0.1278, 0)::geo_location    -- London
        );
        """)
        
        distance = cursor.fetchone()[0]
        print(f"\nDistance between points: {distance/1000:.2f} km")
        
        connection.commit()
        
    except (Exception, Error) as error:
        connection.rollback()
        print(f"Error in custom extensions: {error}")

# Example usage
connection, cursor = create_db_connection()
if connection:
    implement_custom_extensions(cursor, connection)
    cursor.close()
    connection.close()
```

Slide 14: Additional Resources

*   Introduction to PostgreSQL Query Performance - [https://www.google.com/search?q=postgresql+query+performance+optimization](https://www.google.com/search?q=postgresql+query+performance+optimization)
*   PostgreSQL Indexing Deep Dive - [https://www.google.com/search?q=postgresql+indexing+strategies](https://www.google.com/search?q=postgresql+indexing+strategies)
*   Advanced PostgreSQL Replication Patterns - [https://www.google.com/search?q=postgresql+replication+patterns](https://www.google.com/search?q=postgresql+replication+patterns)
*   PostgreSQL Security Best Practices - [https://www.google.com/search?q=postgresql+security+best+practices](https://www.google.com/search?q=postgresql+security+best+practices)
*   High-Performance PostgreSQL Extensions - [https://www.google.com/search?q=postgresql+extensions+development](https://www.google.com/search?q=postgresql+extensions+development)

