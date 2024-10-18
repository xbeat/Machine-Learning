## Customizing Django Model Managers

Slide 1: Custom Model Managers in Django

Custom model managers in Django allow you to extend the default query functionality for your models. They provide a way to encapsulate complex queries and add custom methods to your model's manager.

```python

class CustomManager(models.Manager):
    def get_queryset(self):
        return super().get_queryset().filter(is_active=True)

class MyModel(models.Model):
    name = models.CharField(max_length=100)
    is_active = models.BooleanField(default=True)

    objects = CustomManager()
```

Slide 2: Creating a Custom Manager

To create a custom manager, subclass Django's models.Manager class and define your custom methods. You can then assign an instance of your custom manager to your model's objects attribute.

```python

class BookManager(models.Manager):
    def get_available_books(self):
        return self.filter(is_available=True)

class Book(models.Model):
    title = models.CharField(max_length=200)
    is_available = models.BooleanField(default=True)

    objects = BookManager()

# Usage
available_books = Book.objects.get_available_books()
```

Slide 3: Overriding get\_queryset()

The get\_queryset() method is a powerful way to modify the base QuerySet returned by your manager. This allows you to apply filters or annotations to all queries made through the manager.

```python
from django.utils import timezone

class ActiveUserManager(models.Manager):
    def get_queryset(self):
        return super().get_queryset().filter(
            is_active=True,
            last_login__gte=timezone.now() - timezone.timedelta(days=30)
        )

class User(models.Model):
    username = models.CharField(max_length=100)
    is_active = models.BooleanField(default=True)
    last_login = models.DateTimeField(default=timezone.now)

    objects = models.Manager()  # Default manager
    active_users = ActiveUserManager()  # Custom manager

# Usage
recent_active_users = User.active_users.all()
```

Slide 4: Chaining Querysets with Custom Managers

Custom managers can be used to create reusable queryset methods that can be chained together for complex queries.

```python

class ProductQuerySet(models.QuerySet):
    def in_stock(self):
        return self.filter(stock__gt=0)

    def on_sale(self):
        return self.filter(discount__gt=0)

class ProductManager(models.Manager):
    def get_queryset(self):
        return ProductQuerySet(self.model, using=self._db)

    def in_stock(self):
        return self.get_queryset().in_stock()

    def on_sale(self):
        return self.get_queryset().on_sale()

class Product(models.Model):
    name = models.CharField(max_length=100)
    stock = models.IntegerField(default=0)
    discount = models.DecimalField(max_digits=5, decimal_places=2, default=0)

    objects = ProductManager()

# Usage
on_sale_in_stock = Product.objects.on_sale().in_stock()
```

Slide 5: Using Multiple Managers

You can define multiple managers for a single model, each with its own specific purpose. This allows you to organize your query logic and provide clear, semantic access to different subsets of your data.

```python

class PublishedManager(models.Manager):
    def get_queryset(self):
        return super().get_queryset().filter(status='published')

class DraftManager(models.Manager):
    def get_queryset(self):
        return super().get_queryset().filter(status='draft')

class Article(models.Model):
    title = models.CharField(max_length=200)
    content = models.TextField()
    status = models.CharField(max_length=10, choices=[
        ('draft', 'Draft'),
        ('published', 'Published'),
    ])

    objects = models.Manager()  # Default manager
    published = PublishedManager()  # Custom manager for published articles
    drafts = DraftManager()  # Custom manager for draft articles

# Usage
all_articles = Article.objects.all()
published_articles = Article.published.all()
draft_articles = Article.drafts.all()
```

Slide 6: Aggregation with Custom Managers

Custom managers can incorporate aggregation methods to provide high-level statistics about your model instances.

```python
from django.db.models import Avg, Count

class StudentManager(models.Manager):
    def get_queryset(self):
        return super().get_queryset()

    def average_grade(self):
        return self.aggregate(avg_grade=Avg('grade'))['avg_grade']

    def class_size(self):
        return self.count()

class Student(models.Model):
    name = models.CharField(max_length=100)
    grade = models.FloatField()

    objects = StudentManager()

# Usage
avg_grade = Student.objects.average_grade()
total_students = Student.objects.class_size()

print(f"Average grade: {avg_grade:.2f}")
print(f"Total students: {total_students}")
```

Slide 7: Combining Multiple Managers

You can combine multiple custom managers to create more complex and flexible query interfaces for your models.

```python
from django.utils import timezone

class ActiveManager(models.Manager):
    def get_queryset(self):
        return super().get_queryset().filter(is_active=True)

class RecentManager(models.Manager):
    def get_queryset(self):
        one_week_ago = timezone.now() - timezone.timedelta(days=7)
        return super().get_queryset().filter(created_at__gte=one_week_ago)

class Item(models.Model):
    name = models.CharField(max_length=100)
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)

    objects = models.Manager()
    active = ActiveManager()
    recent = RecentManager()

# Usage
all_items = Item.objects.all()
active_items = Item.active.all()
recent_items = Item.recent.all()
active_recent_items = Item.active.filter(created_at__gte=timezone.now() - timezone.timedelta(days=7))
```

Slide 8: Custom Managers for Related Objects

Custom managers can also be used to manage related objects, providing a convenient way to access specific subsets of related data.

```python

class AuthorManager(models.Manager):
    def get_queryset(self):
        return super().get_queryset()

    def with_books(self):
        return self.annotate(book_count=models.Count('books')).filter(book_count__gt=0)

class Author(models.Model):
    name = models.CharField(max_length=100)

    objects = AuthorManager()

class Book(models.Model):
    title = models.CharField(max_length=200)
    author = models.ForeignKey(Author, on_delete=models.CASCADE, related_name='books')

# Usage
authors_with_books = Author.objects.with_books()
for author in authors_with_books:
    print(f"{author.name} has written {author.book_count} books")
```

Slide 9: Using F() Expressions in Custom Managers

Custom managers can leverage F() expressions to perform operations at the database level, improving performance for certain types of queries.

```python
from django.db.models import F

class InventoryManager(models.Manager):
    def get_queryset(self):
        return super().get_queryset()

    def low_stock(self, threshold=10):
        return self.filter(quantity__lte=threshold)

    def restock(self, amount):
        return self.update(quantity=F('quantity') + amount)

class Product(models.Model):
    name = models.CharField(max_length=100)
    quantity = models.IntegerField(default=0)

    objects = InventoryManager()

# Usage
low_stock_products = Product.objects.low_stock()
Product.objects.filter(name='Widget').restock(50)
```

Slide 10: Custom Managers with Annotations

Custom managers can use annotations to add computed fields to your queryset, allowing for more complex data retrieval and analysis.

```python
from django.db.models import Sum, F

class OrderManager(models.Manager):
    def get_queryset(self):
        return super().get_queryset()

    def with_total_value(self):
        return self.annotate(
            total_value=Sum(F('orderitem__quantity') * F('orderitem__product__price'))
        )

class Order(models.Model):
    customer = models.CharField(max_length=100)
    date = models.DateField(auto_now_add=True)

    objects = OrderManager()

class OrderItem(models.Model):
    order = models.ForeignKey(Order, on_delete=models.CASCADE)
    product = models.ForeignKey('Product', on_delete=models.CASCADE)
    quantity = models.IntegerField()

class Product(models.Model):
    name = models.CharField(max_length=100)
    price = models.DecimalField(max_digits=10, decimal_places=2)

# Usage
orders_with_total = Order.objects.with_total_value()
for order in orders_with_total:
    print(f"Order for {order.customer}: ${order.total_value}")
```

Slide 11: Real-life Example: Content Management System

In this example, we'll create custom managers for a simple content management system, demonstrating how they can be used to manage different types of content and their visibility.

```python
from django.utils import timezone

class ContentManager(models.Manager):
    def get_queryset(self):
        return super().get_queryset()

    def published(self):
        return self.filter(status='published', publish_date__lte=timezone.now())

    def featured(self):
        return self.published().filter(is_featured=True)

class Content(models.Model):
    title = models.CharField(max_length=200)
    body = models.TextField()
    status = models.CharField(max_length=10, choices=[
        ('draft', 'Draft'),
        ('published', 'Published'),
    ])
    publish_date = models.DateTimeField(null=True, blank=True)
    is_featured = models.BooleanField(default=False)

    objects = ContentManager()

# Usage
published_content = Content.objects.published()
featured_content = Content.objects.featured()

for content in featured_content:
    print(f"Featured: {content.title}")
```

Slide 12: Real-life Example: Task Management System

This example demonstrates how custom managers can be used in a task management system to handle different task states and priorities.

```python
from django.utils import timezone

class TaskManager(models.Manager):
    def get_queryset(self):
        return super().get_queryset()

    def active(self):
        return self.filter(status='active')

    def overdue(self):
        return self.active().filter(due_date__lt=timezone.now())

    def high_priority(self):
        return self.active().filter(priority='high')

class Task(models.Model):
    title = models.CharField(max_length=200)
    description = models.TextField()
    status = models.CharField(max_length=10, choices=[
        ('active', 'Active'),
        ('completed', 'Completed'),
        ('archived', 'Archived'),
    ])
    priority = models.CharField(max_length=10, choices=[
        ('low', 'Low'),
        ('medium', 'Medium'),
        ('high', 'High'),
    ])
    due_date = models.DateTimeField()

    objects = TaskManager()

# Usage
active_tasks = Task.objects.active()
overdue_tasks = Task.objects.overdue()
high_priority_tasks = Task.objects.high_priority()

print("High Priority Tasks:")
for task in high_priority_tasks:
    print(f"- {task.title} (Due: {task.due_date})")
```

Slide 13: Best Practices for Custom Managers

1. Keep managers focused: Each manager should have a clear, specific purpose.
2. Use meaningful names: Choose descriptive names for your managers and their methods.
3. Leverage query optimization: Use select\_related() and prefetch\_related() to minimize database queries.
4. Combine managers with model methods: Use model methods for object-specific operations and managers for queryset operations.
5. Document your managers: Provide clear documentation for your custom managers and their methods.

```python

class BookManager(models.Manager):
    def get_queryset(self):
        return super().get_queryset()

    def available_ebooks(self):
        """
        Returns a queryset of available e-books.
        Optimizes the query by selecting related author information.
        """
        return self.filter(format='ebook', is_available=True).select_related('author')

class Book(models.Model):
    title = models.CharField(max_length=200)
    author = models.ForeignKey('Author', on_delete=models.CASCADE)
    format = models.CharField(max_length=10, choices=[('print', 'Print'), ('ebook', 'E-book')])
    is_available = models.BooleanField(default=True)

    objects = BookManager()

    def mark_unavailable(self):
        """Mark the book as unavailable and save it."""
        self.is_available = False
        self.save()

# Usage
available_ebooks = Book.objects.available_ebooks()
for book in available_ebooks:
    print(f"{book.title} by {book.author.name}")
```

Slide 14: Additional Resources

For more information on custom model managers in Django, you can refer to the following resources:

1. Django Official Documentation on Managers: [https://docs.djangoproject.com/en/stable/topics/db/managers/](https://docs.djangoproject.com/en/stable/topics/db/managers/)
2. "Mastering Django Models" by Antonio Melé: This book provides in-depth coverage of Django models, including custom managers.
3. "Django Design Patterns and Best Practices" by Arun Ravindran: This book covers various Django design patterns, including effective use of custom managers.
4. Django's GitHub repository: [https://github.com/django/django](https://github.com/django/django) Exploring the source code can provide insights into how Django implements managers internally.

Remember to always refer to the most up-to-date documentation and resources as Django continues to evolve.


