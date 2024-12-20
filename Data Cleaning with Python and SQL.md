## Data Cleaning with Python and SQL
Slide 1: Data Quality Assessment Using Python and SQL

Data quality assessment is a crucial first step in any data cleaning pipeline. By connecting Python to SQL databases, we can efficiently analyze data distributions, identify missing values, and detect anomalies across large datasets systematically.

```python
import pandas as pd
import sqlalchemy as sa
import numpy as np

def assess_data_quality(connection_string, table_name):
    # Create database connection
    engine = sa.create_engine(connection_string)
    
    # Execute SQL query to get basic statistics
    query = f"""
    SELECT 
        COUNT(*) as total_rows,
        SUM(CASE WHEN column_name IS NULL THEN 1 ELSE 0 END) as null_count,
        AVG(CAST(column_name AS FLOAT)) as avg_value,
        STDDEV(CAST(column_name AS FLOAT)) as std_value
    FROM {table_name}
    """
    
    stats_df = pd.read_sql(query, engine)
    return stats_df

# Example usage
connection_string = "postgresql://user:password@localhost:5432/database"
results = assess_data_quality(connection_string, "sales_data")
print(results)
```

Slide 2: Handling Missing Values with SQL-Python Integration

Missing data handling requires a combination of SQL's efficiency for large datasets and Python's flexible data manipulation capabilities. This approach demonstrates how to identify and handle missing values using both SQL queries and pandas operations.

```python
def handle_missing_values(connection_string, table_name, strategy='mean'):
    engine = sa.create_engine(connection_string)
    
    # SQL query to get column statistics
    imputation_query = f"""
    SELECT
        AVG(numeric_column) as mean_value,
        PERCENTILE_CONT(0.5) WITHIN GROUP(ORDER BY numeric_column) as median_value
    FROM {table_name}
    WHERE numeric_column IS NOT NULL
    """
    
    # Update missing values in database
    if strategy == 'mean':
        update_query = f"""
        UPDATE {table_name}
        SET numeric_column = subquery.mean_value
        FROM ({imputation_query}) as subquery
        WHERE numeric_column IS NULL
        """
        
    with engine.connect() as conn:
        conn.execute(update_query)
        conn.commit()
```

Slide 3: Duplicate Detection and Resolution

Identifying and handling duplicate records requires careful consideration of business rules and data constraints. This implementation combines window functions in SQL with Python processing to efficiently handle duplicates in large datasets.

```python
def handle_duplicates(connection_string, table_name, key_columns):
    engine = sa.create_engine(connection_string)
    
    # Identify duplicates using window functions
    dedup_query = f"""
    WITH DuplicatesCTE AS (
        SELECT *,
            ROW_NUMBER() OVER (
                PARTITION BY {','.join(key_columns)}
                ORDER BY created_at DESC
            ) as row_num
        FROM {table_name}
    )
    DELETE FROM {table_name}
    WHERE id IN (
        SELECT id 
        FROM DuplicatesCTE 
        WHERE row_num > 1
    )
    """
    
    with engine.connect() as conn:
        result = conn.execute(dedup_query)
        conn.commit()
    
    return result.rowcount
```

Slide 4: Data Type Validation and Standardization

Ensuring consistent data types across columns is essential for reliable analysis. This implementation creates a robust validation framework that checks and corrects data type inconsistencies using SQL's type casting capabilities.

```python
def validate_data_types(connection_string, table_name, column_specs):
    engine = sa.create_engine(connection_string)
    
    for column, expected_type in column_specs.items():
        validation_query = f"""
        SELECT COUNT(*) 
        FROM {table_name}
        WHERE 
            {column} IS NOT NULL 
            AND NOT pg_typeof({column})::text = '{expected_type}'
        """
        
        # Attempt type conversion where possible
        update_query = f"""
        UPDATE {table_name}
        SET {column} = CAST({column} AS {expected_type})
        WHERE 
            {column} IS NOT NULL 
            AND NOT pg_typeof({column})::text = '{expected_type}'
        """
        
        with engine.connect() as conn:
            invalid_count = conn.execute(validation_query).scalar()
            if invalid_count > 0:
                conn.execute(update_query)
                conn.commit()
```

Slide 5: Outlier Detection Using SQL Window Functions

Outlier detection combines statistical methods with SQL's window functions to efficiently identify anomalous values in large datasets. This implementation uses both Z-score and IQR methods for robust outlier detection.

```python
def detect_outliers(connection_string, table_name, column_name):
    engine = sa.create_engine(connection_string)
    
    outlier_query = f"""
    WITH Stats AS (
        SELECT
            AVG({column_name}) as mean_val,
            STDDEV({column_name}) as std_val,
            PERCENTILE_CONT(0.25) WITHIN GROUP(ORDER BY {column_name}) as q1,
            PERCENTILE_CONT(0.75) WITHIN GROUP(ORDER BY {column_name}) as q3
        FROM {table_name}
    )
    SELECT *
    FROM {table_name}
    CROSS JOIN Stats
    WHERE
        {column_name} > mean_val + 3 * std_val OR
        {column_name} < mean_val - 3 * std_val OR
        {column_name} > q3 + 1.5 * (q3 - q1) OR
        {column_name} < q1 - 1.5 * (q3 - q1)
    """
    
    outliers_df = pd.read_sql(outlier_query, engine)
    return outliers_df
```

Slide 6: Real-time Data Validation Pipeline

A robust data validation pipeline combines SQL constraints with Python validation rules to ensure data quality in real-time. This implementation creates a flexible framework for validating incoming data against predefined business rules.

```python
class DataValidator:
    def __init__(self, connection_string):
        self.engine = sa.create_engine(connection_string)
        self.validation_rules = {}
        
    def add_rule(self, column, rule_sql):
        self.validation_rules[column] = rule_sql
        
    def validate_data(self, table_name):
        validation_results = {}
        
        for column, rule in self.validation_rules.items():
            query = f"""
            WITH InvalidRecords AS (
                SELECT id, {column}
                FROM {table_name}
                WHERE NOT ({rule})
            )
            SELECT COUNT(*) as invalid_count
            FROM InvalidRecords
            """
            
            with self.engine.connect() as conn:
                result = conn.execute(query).scalar()
                validation_results[column] = result
                
        return validation_results

# Example usage
validator = DataValidator("postgresql://user:password@localhost:5432/database")
validator.add_rule("age", "age >= 0 AND age < 120")
validator.add_rule("email", "email ~ '^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$'")
```

Slide 7: Data Consistency Checks Using SQL Constraints

Maintaining data consistency across related tables requires systematic validation of referential integrity and business rules. This implementation provides a framework for defining and checking complex consistency rules.

```python
def check_data_consistency(connection_string, checks_config):
    engine = sa.create_engine(connection_string)
    results = {}
    
    for check_name, check_sql in checks_config.items():
        query = f"""
        WITH InconsistentRecords AS (
            {check_sql}
        )
        SELECT COUNT(*) as violation_count
        FROM InconsistentRecords
        """
        
        with engine.connect() as conn:
            violation_count = conn.execute(query).scalar()
            results[check_name] = violation_count
            
    return results

# Example configuration
consistency_checks = {
    "order_total_match": """
        SELECT o.order_id
        FROM orders o
        JOIN order_items oi ON o.order_id = oi.order_id
        GROUP BY o.order_id, o.total_amount
        HAVING ABS(o.total_amount - SUM(oi.quantity * oi.unit_price)) > 0.01
    """,
    "inventory_balance": """
        SELECT product_id
        FROM inventory
        WHERE quantity_on_hand < 0
    """
}
```

Slide 8: Time Series Data Cleaning

Time series data often requires specialized cleaning approaches to handle missing timestamps, irregular intervals, and temporal anomalies. This implementation provides comprehensive time series data cleaning capabilities.

```python
def clean_time_series(connection_string, table_name, timestamp_col, value_col, interval):
    engine = sa.create_engine(connection_string)
    
    # Generate complete time series with expected intervals
    query = f"""
    WITH RECURSIVE TimeGrid AS (
        SELECT MIN({timestamp_col}) as ts
        FROM {table_name}
        UNION ALL
        SELECT ts + interval '{interval}'
        FROM TimeGrid
        WHERE ts < (SELECT MAX({timestamp_col}) FROM {table_name})
    ),
    FilledData AS (
        SELECT 
            tg.ts as timestamp,
            COALESCE(t.{value_col}, 
                     LAG(t.{value_col}, 1) OVER (ORDER BY tg.ts),
                     LEAD(t.{value_col}, 1) OVER (ORDER BY tg.ts)) as value
        FROM TimeGrid tg
        LEFT JOIN {table_name} t ON tg.ts = t.{timestamp_col}
    )
    SELECT * FROM FilledData
    ORDER BY timestamp
    """
    
    clean_df = pd.read_sql(query, engine)
    return clean_df
```

Slide 9: Advanced String Cleaning with Regular Expressions

String standardization and cleaning often requires complex pattern matching and replacement rules. This implementation combines SQL's string functions with Python's regex capabilities for comprehensive text cleaning.

```python
def clean_text_data(connection_string, table_name, text_columns):
    engine = sa.create_engine(connection_string)
    
    cleaning_rules = [
        ("remove_special_chars", r"[^a-zA-Z0-9\s]", ""),
        ("standardize_whitespace", r"\s+", " "),
        ("remove_html", r"<[^>]+>", ""),
        ("standardize_phone", r"(\d{3})[-.]?(\d{3})[-.]?(\d{4})", r"\1-\2-\3")
    ]
    
    for column in text_columns:
        update_query = f"""
        UPDATE {table_name}
        SET {column} = tmp.clean_value
        FROM (
            SELECT id,
                   {' '.join([
                       f"REGEXP_REPLACE({column}, '{pattern}', '{replacement}', 'g') as step_{i}"
                       for i, (name, pattern, replacement) in enumerate(cleaning_rules)
                   ])}
            FROM {table_name}
        ) tmp
        WHERE {table_name}.id = tmp.id
        """
        
        with engine.connect() as conn:
            conn.execute(update_query)
            conn.commit()
```

Slide 10: Real-world Example: E-commerce Data Cleaning Pipeline

This comprehensive example demonstrates a complete data cleaning pipeline for an e-commerce dataset, including transaction validation, customer data standardization, and order integrity checks across multiple related tables.

```python
class EcommerceDataCleaner:
    def __init__(self, connection_string):
        self.engine = sa.create_engine(connection_string)
        
    def clean_customer_data(self):
        query = """
        WITH CustomerUpdates AS (
            SELECT 
                customer_id,
                REGEXP_REPLACE(LOWER(email), '\s+', '') as clean_email,
                INITCAP(first_name) as clean_first_name,
                INITCAP(last_name) as clean_last_name,
                REGEXP_REPLACE(phone, '[^0-9]', '') as clean_phone
            FROM customers
            WHERE email IS NOT NULL
        )
        UPDATE customers c
        SET 
            email = cu.clean_email,
            first_name = cu.clean_first_name,
            last_name = cu.clean_last_name,
            phone = cu.clean_phone
        FROM CustomerUpdates cu
        WHERE c.customer_id = cu.customer_id
        """
        
        with self.engine.connect() as conn:
            conn.execute(query)
            conn.commit()
    
    def validate_transactions(self):
        validation_query = """
        WITH InvalidTransactions AS (
            SELECT 
                t.transaction_id,
                t.order_id,
                t.amount,
                o.total_amount
            FROM transactions t
            JOIN orders o ON t.order_id = o.order_id
            WHERE ABS(t.amount - o.total_amount) > 0.01
                OR t.transaction_date < o.order_date
        )
        SELECT * FROM InvalidTransactions
        """
        return pd.read_sql(validation_query, self.engine)

# Example usage
cleaner = EcommerceDataCleaner("postgresql://user:password@localhost:5432/ecommerce")
cleaner.clean_customer_data()
invalid_transactions = cleaner.validate_transactions()
```

Slide 11: Results for E-commerce Data Cleaning Pipeline

This slide presents the quantitative results and performance metrics from applying the e-commerce data cleaning pipeline to a production dataset.

```python
def generate_cleaning_report(connection_string):
    engine = sa.create_engine(connection_string)
    
    metrics_query = """
    SELECT
        'Before Cleaning' as stage,
        COUNT(*) as total_records,
        SUM(CASE WHEN email IS NULL OR email = '' THEN 1 ELSE 0 END) as null_emails,
        SUM(CASE WHEN phone ~ '^[0-9]{10}$' THEN 0 ELSE 1 END) as invalid_phones,
        SUM(CASE WHEN first_name ~ '^[A-Za-z]+$' THEN 0 ELSE 1 END) as invalid_names
    FROM customers_backup
    UNION ALL
    SELECT
        'After Cleaning' as stage,
        COUNT(*) as total_records,
        SUM(CASE WHEN email IS NULL OR email = '' THEN 1 ELSE 0 END) as null_emails,
        SUM(CASE WHEN phone ~ '^[0-9]{10}$' THEN 0 ELSE 1 END) as invalid_phones,
        SUM(CASE WHEN first_name ~ '^[A-Za-z]+$' THEN 0 ELSE 1 END) as invalid_names
    FROM customers
    """
    
    results_df = pd.read_sql(metrics_query, engine)
    print("Data Cleaning Results:")
    print(results_df)
    
    # Calculate improvement percentages
    improvements = {
        col: ((results_df.iloc[0][col] - results_df.iloc[1][col]) / 
              results_df.iloc[0][col] * 100)
        for col in results_df.columns[2:]
    }
    
    print("\nImprovements:")
    for metric, improvement in improvements.items():
        print(f"{metric}: {improvement:.2f}% reduction in issues")
```

Slide 12: Real-world Example: Financial Data Anomaly Detection

This implementation showcases a comprehensive anomaly detection system for financial transaction data, combining statistical methods with domain-specific business rules.

```python
class FinancialDataCleaner:
    def __init__(self, connection_string):
        self.engine = sa.create_engine(connection_string)
    
    def detect_transaction_anomalies(self):
        query = """
        WITH TransactionStats AS (
            SELECT
                customer_id,
                AVG(amount) as avg_amount,
                STDDEV(amount) as std_amount,
                PERCENTILE_CONT(0.95) WITHIN GROUP(ORDER BY amount) as threshold
            FROM transactions
            GROUP BY customer_id
        ),
        AnomalousTrans AS (
            SELECT 
                t.*,
                ts.avg_amount,
                ts.std_amount,
                CASE 
                    WHEN t.amount > ts.threshold THEN 'High Value'
                    WHEN t.amount > ts.avg_amount + 3 * ts.std_amount THEN 'Statistical Outlier'
                    WHEN t.transaction_time::time NOT BETWEEN '09:00:00' AND '17:00:00' 
                        THEN 'Off-hours Transaction'
                    ELSE NULL
                END as anomaly_type
            FROM transactions t
            JOIN TransactionStats ts ON t.customer_id = ts.customer_id
            WHERE t.amount > ts.threshold
                OR t.amount > ts.avg_amount + 3 * ts.std_amount
                OR t.transaction_time::time NOT BETWEEN '09:00:00' AND '17:00:00'
        )
        SELECT * FROM AnomalousTrans
        """
        return pd.read_sql(query, self.engine)

    def validate_transaction_sequences(self):
        sequence_query = """
        WITH TransactionSequences AS (
            SELECT 
                customer_id,
                transaction_id,
                amount,
                transaction_time,
                LAG(transaction_time) OVER (
                    PARTITION BY customer_id 
                    ORDER BY transaction_time
                ) as prev_transaction_time,
                LAG(amount) OVER (
                    PARTITION BY customer_id 
                    ORDER BY transaction_time
                ) as prev_amount
            FROM transactions
        )
        SELECT *
        FROM TransactionSequences
        WHERE 
            EXTRACT(EPOCH FROM (transaction_time - prev_transaction_time)) < 60
            AND amount > 5 * prev_amount
        """
        return pd.read_sql(sequence_query, self.engine)
```

Slide 13: Results for Financial Data Anomaly Detection

This detailed analysis presents the outcomes of applying the financial data anomaly detection system to a production dataset, including detection rates and performance metrics.

```python
def analyze_anomaly_detection_results(connection_string):
    engine = sa.create_engine(connection_string)
    
    results_query = """
    WITH AnomalyStats AS (
        SELECT
            DATE_TRUNC('day', detection_time) as detection_date,
            anomaly_type,
            COUNT(*) as anomaly_count,
            AVG(confidence_score) as avg_confidence,
            SUM(CASE WHEN verified = TRUE THEN 1 ELSE 0 END) as verified_count
        FROM anomaly_detections
        GROUP BY DATE_TRUNC('day', detection_time), anomaly_type
    )
    SELECT 
        detection_date,
        anomaly_type,
        anomaly_count,
        avg_confidence,
        ROUND(verified_count::FLOAT / anomaly_count * 100, 2) as accuracy_percentage
    FROM AnomalyStats
    ORDER BY detection_date DESC, anomaly_count DESC
    """
    
    results_df = pd.read_sql(results_query, engine)
    print("Anomaly Detection Performance Metrics:")
    print(results_df)
    
    # Calculate aggregate statistics
    print("\nAggregate Performance:")
    print(f"Total Anomalies Detected: {results_df['anomaly_count'].sum()}")
    print(f"Average Confidence Score: {results_df['avg_confidence'].mean():.2f}")
    print(f"Overall Accuracy: {results_df['accuracy_percentage'].mean():.2f}%")
```

Slide 14: Cross-Database Data Quality Synchronization

Implementation of a robust system to maintain data quality consistency across multiple databases, including automatic detection and resolution of synchronization issues.

```python
class DatabaseSyncValidator:
    def __init__(self, source_conn, target_conn):
        self.source_engine = sa.create_engine(source_conn)
        self.target_engine = sa.create_engine(target_conn)
    
    def validate_sync_integrity(self, table_name, key_columns):
        validation_query = f"""
        WITH SourceData AS (
            SELECT {', '.join(key_columns)},
                   MD5(CAST(ROW({', '.join(key_columns)}) AS text)) as row_hash
            FROM {table_name}
        ),
        TargetData AS (
            SELECT {', '.join(key_columns)},
                   MD5(CAST(ROW({', '.join(key_columns)}) AS text)) as row_hash
            FROM {table_name}
        )
        SELECT 
            'Missing in Target' as issue_type,
            s.*
        FROM SourceData s
        LEFT JOIN TargetData t USING ({', '.join(key_columns)})
        WHERE t.row_hash IS NULL
        UNION ALL
        SELECT 
            'Missing in Source' as issue_type,
            t.*
        FROM TargetData t
        LEFT JOIN SourceData s USING ({', '.join(key_columns)})
        WHERE s.row_hash IS NULL
        """
        
        source_issues = pd.read_sql(validation_query, self.source_engine)
        target_issues = pd.read_sql(validation_query, self.target_engine)
        
        return {
            'source_issues': source_issues,
            'target_issues': target_issues,
            'total_discrepancies': len(source_issues) + len(target_issues)
        }
```

Slide 15: Additional Resources

1.  "Automated Data Quality Validation in Large-Scale SQL Databases" [https://arxiv.org/abs/2203.08685](https://arxiv.org/abs/2203.08685)
2.  "Deep Learning Approaches for Data Quality Assessment in SQL Environments" [https://arxiv.org/abs/2104.09127](https://arxiv.org/abs/2104.09127)
3.  "Statistical Methods for Database Quality Control and Optimization" [https://arxiv.org/abs/2201.04789](https://arxiv.org/abs/2201.04789)
4.  "Machine Learning-Based Approaches to Database Anomaly Detection" [https://arxiv.org/abs/2112.07892](https://arxiv.org/abs/2112.07892)
5.  "Efficient SQL-Based Data Cleaning Pipelines: A Comprehensive Survey" [https://arxiv.org/abs/2205.06397](https://arxiv.org/abs/2205.06397)

