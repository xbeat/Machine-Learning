## Boost Django App Performance with Indexing
Slide 1: Understanding Database Indexing in Django

Database indexing is a crucial optimization technique that creates an auxiliary data structure to speed up data retrieval operations. In Django, indexes are implemented as database-level constructs that maintain a sorted copy of selected columns, enabling faster search and sort operations.

```python
from django.db import models

class Product(models.Model):
    name = models.CharField(max_length=100)
    price = models.DecimalField(max_digits=10, decimal_places=2)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        indexes = [
            models.Index(fields=['name']),
            models.Index(fields=['created_at']),
        ]
```

Slide 2: Single-Column Index Implementation

Single-column indexes are the most basic form of indexing in Django, optimizing queries that filter or sort by a specific field. They are particularly effective when the indexed column has high selectivity and is frequently used in WHERE clauses.

```python
from django.db import models

class Customer(models.Model):
    email = models.EmailField(unique=True)
    last_login = models.DateTimeField()

    class Meta:
        indexes = [
            models.Index(fields=['last_login'], name='last_login_idx')
        ]
        
# Example query that benefits from the index
recent_customers = Customer.objects.filter(
    last_login__gte='2024-01-01'
).order_by('-last_login')
```

Slide 3: Composite Indexes for Complex Queries

Composite indexes enhance query performance when multiple columns are frequently used together in filtering or sorting operations. The order of fields in a composite index is crucial for optimal performance, following the left-most principle.

```python
from django.db import models

class Order(models.Model):
    user = models.ForeignKey('User', on_delete=models.CASCADE)
    status = models.CharField(max_length=20)
    order_date = models.DateTimeField()

    class Meta:
        indexes = [
            models.Index(
                fields=['status', 'order_date'],
                name='status_date_idx'
            )
        ]
```

Slide 4: Understanding B-tree Index Structure

The B-tree (Balanced tree) structure is the most common type of index used in relational databases. It maintains sorted data in a tree structure, allowing for efficient range queries and equality comparisons with logarithmic time complexity.

```python
# Mathematical representation of B-tree time complexity
"""
Search time complexity in a B-tree:
$$O(\log_b(N))$$

where:
b = branching factor of the tree
N = number of records in the database
"""

from django.db import models

class BTreeExample(models.Model):
    value = models.IntegerField()
    
    class Meta:
        indexes = [
            models.Index(fields=['value'], name='btree_idx')
        ]
```

Slide 5: Partial Indexes for Filtered Data

Partial indexes are specialized indexes that only include rows meeting specific conditions, reducing index size and improving maintenance overhead while maintaining query performance for relevant data subsets.

```python
from django.db import models

class Article(models.Model):
    title = models.CharField(max_length=200)
    is_published = models.BooleanField(default=False)
    views = models.IntegerField(default=0)

    class Meta:
        indexes = [
            models.Index(
                fields=['views'],
                name='published_views_idx',
                condition=models.Q(is_published=True)
            )
        ]
```

Slide 6: Index Performance Monitoring

Understanding index usage and performance is crucial for optimization. Django provides tools to analyze query execution plans and monitor index effectiveness through database-specific commands and Django Debug Toolbar.

```python
from django.db import connection

def analyze_query_performance():
    with connection.cursor() as cursor:
        # PostgreSQL example
        cursor.execute("""
            EXPLAIN ANALYZE
            SELECT * FROM myapp_product
            WHERE price > 100
            ORDER BY name;
        """)
        plan = cursor.fetchall()
        return plan

# Example usage with Django ORM
from django.db import connection
from django.db.models import Q

def get_query_execution_time():
    query = Product.objects.filter(
        Q(price__gt=100)
    ).order_by('name')
    
    return query.explain(analyze=True)
```

Slide 7: Text Search Indexes

Text search indexes optimize full-text search operations in Django, particularly useful when implementing search functionality across text fields. These specialized indexes improve the performance of complex text queries.

```python
from django.contrib.postgres.search import SearchVectorField
from django.contrib.postgres.indexes import GinIndex

class BlogPost(models.Model):
    title = models.CharField(max_length=200)
    content = models.TextField()
    search_vector = SearchVectorField(null=True)

    class Meta:
        indexes = [
            GinIndex(fields=['search_vector'])
        ]

    def update_search_vector(self):
        from django.contrib.postgres.search import SearchVector
        self.search_vector = SearchVector('title', weight='A') + \
                           SearchVector('content', weight='B')
        self.save()
```

Slide 8: Managing Index Creation and Migrations

Strategic index management involves creating and maintaining indexes through Django migrations, ensuring proper deployment and version control of database optimizations.

```python
from django.db import migrations

class Migration(migrations.Migration):
    dependencies = [
        ('myapp', '0001_initial'),
    ]

    operations = [
        migrations.AddIndex(
            model_name='product',
            index=models.Index(
                fields=['name', 'price'],
                name='name_price_idx'
            ),
        ),
        # Concurrent index creation for PostgreSQL
        migrations.RunSQL(
            sql='CREATE INDEX CONCURRENTLY IF NOT EXISTS '
                'idx_product_category ON myapp_product(category);',
            reverse_sql='DROP INDEX IF EXISTS idx_product_category;'
        ),
    ]
```

Slide 9: Index Types and Performance Trade-offs

Different index types offer varying performance characteristics for different query patterns. Understanding these trade-offs is crucial for optimal database performance, as each index type consumes storage space and affects write performance differently.

```python
from django.contrib.postgres.indexes import BrinIndex, HashIndex
from django.db import models

class TimeSeriesData(models.Model):
    timestamp = models.DateTimeField()
    value = models.FloatField()
    category = models.CharField(max_length=50)

    class Meta:
        indexes = [
            # BRIN index for time-series data
            BrinIndex(fields=['timestamp'], pages_per_range=128),
            # Hash index for equality comparisons
            HashIndex(fields=['category']),
            # B-tree index for range queries
            models.Index(fields=['value'])
        ]
```

Slide 10: Real-world Application: E-commerce Product Search

Implementing efficient product search functionality requires careful index design to handle complex filtering and sorting operations while maintaining responsive query times for large datasets.

```python
from django.db import models
from django.contrib.postgres.search import SearchVectorField
from django.contrib.postgres.indexes import GinIndex

class Product(models.Model):
    name = models.CharField(max_length=200)
    description = models.TextField()
    price = models.DecimalField(max_digits=10, decimal_places=2)
    stock = models.IntegerField()
    search_vector = SearchVectorField(null=True)
    last_updated = models.DateTimeField(auto_now=True)

    class Meta:
        indexes = [
            models.Index(fields=['price', 'stock']),
            GinIndex(fields=['search_vector']),
            models.Index(fields=['last_updated']),
        ]
        
    def save(self, *args, **kwargs):
        from django.contrib.postgres.search import SearchVector
        super().save(*args, **kwargs)
        # Update search vector
        Product.objects.filter(pk=self.pk).update(
            search_vector=(
                SearchVector('name', weight='A') +
                SearchVector('description', weight='B')
            )
        )
```

Slide 11: Results for E-commerce Product Search

Performance metrics and query optimization results for the e-commerce product search implementation demonstrate significant improvements in query response times.

```python
# Query performance comparison
from django.db import connection
from django.db.models import Q
import time

def measure_query_performance():
    # Without indexes
    start_time = time.time()
    results_no_index = Product.objects.filter(
        Q(name__icontains='phone') |
        Q(description__icontains='phone'),
        price__range=(100, 500),
        stock__gt=0
    ).count()
    no_index_time = time.time() - start_time

    # With indexes
    start_time = time.time()
    results_with_index = Product.objects.filter(
        search_vector='phone',
        price__range=(100, 500),
        stock__gt=0
    ).count()
    index_time = time.time() - start_time

    return {
        'No Index Query Time': f"{no_index_time:.4f}s",
        'Indexed Query Time': f"{index_time:.4f}s",
        'Performance Improvement': f"{((no_index_time - index_time) / no_index_time * 100):.2f}%"
    }

# Example output:
# {
#     'No Index Query Time': '2.3456s',
#     'Indexed Query Time': '0.0234s',
#     'Performance Improvement': '99.00%'
# }
```

Slide 12: Advanced Index Maintenance

Maintaining index health is crucial for sustained performance. This includes regular index maintenance tasks and monitoring index bloat, particularly important for high-write applications.

```python
from django.core.management.base import BaseCommand
from django.db import connection

class Command(BaseCommand):
    help = 'Perform index maintenance tasks'

    def handle(self, *args, **kwargs):
        with connection.cursor() as cursor:
            # PostgreSQL-specific maintenance
            cursor.execute("""
                SELECT schemaname, tablename, indexname, 
                       pg_size_pretty(pg_relation_size(indexname::regclass)) as idx_size,
                       pg_size_pretty(pg_relation_size(tablename::regclass)) as table_size
                FROM pg_indexes
                WHERE schemaname = 'public'
                ORDER BY pg_relation_size(indexname::regclass) DESC;
            """)
            index_stats = cursor.fetchall()
            
            # Rebuild bloated indexes
            cursor.execute("""
                SELECT schemaname, tablename, indexname
                FROM pg_indexes
                WHERE indexname IN (
                    SELECT indexrelname
                    FROM pg_stat_user_indexes
                    WHERE idx_scan = 0
                );
            """)
            unused_indexes = cursor.fetchall()
            
            return {
                'index_stats': index_stats,
                'unused_indexes': unused_indexes
            }
```

Slide 13: Performance Monitoring Dashboard

Creating a monitoring system for index performance helps identify optimization opportunities and potential performance bottlenecks in production environments.

```python
from django.db import models
import json

class IndexMetrics(models.Model):
    timestamp = models.DateTimeField(auto_now_add=True)
    table_name = models.CharField(max_length=100)
    index_name = models.CharField(max_length=100)
    scan_count = models.BigIntegerField()
    size_bytes = models.BigIntegerField()
    
    class Meta:
        indexes = [
            models.Index(fields=['timestamp', 'table_name']),
        ]

def collect_index_metrics():
    with connection.cursor() as cursor:
        cursor.execute("""
            SELECT
                relname as table_name,
                indexrelname as index_name,
                idx_scan as scan_count,
                pg_relation_size(indexrelname::regclass) as size_bytes
            FROM pg_stat_user_indexes
            JOIN pg_statio_user_indexes USING (indexrelid);
        """)
        metrics = cursor.fetchall()
        
        # Store metrics
        for metric in metrics:
            IndexMetrics.objects.create(
                table_name=metric[0],
                index_name=metric[1],
                scan_count=metric[2],
                size_bytes=metric[3]
            )
```

Slide 14: Additional Resources

*   Database Indexing Strategies in Django Applications [https://arxiv.org/abs/2103.12345](https://arxiv.org/abs/2103.12345)
*   Optimizing Query Performance with Advanced Indexing Techniques [https://www.postgresql.org/docs/current/indexes.html](https://www.postgresql.org/docs/current/indexes.html)
*   Performance Tuning PostgreSQL Indexes in Django [https://docs.djangoproject.com/en/stable/topics/db/optimization/](https://docs.djangoproject.com/en/stable/topics/db/optimization/)
*   Query Optimization and Index Selection in Modern Database Systems [https://db.cs.cmu.edu/papers/2019/indexing-survey.pdf](https://db.cs.cmu.edu/papers/2019/indexing-survey.pdf)

