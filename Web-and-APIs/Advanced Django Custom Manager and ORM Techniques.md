## Advanced Django Custom Manager and ORM Techniques

Slide 1: Custom Managers in Django

Custom managers in Django allow you to extend the default query functionality of models. They provide a way to encapsulate common query patterns and add custom methods to your model's interface.

```python

class ProductManager(models.Manager):
    def expensive_products(self):
        return self.filter(price__gt=100)

class Product(models.Model):
    name = models.CharField(max_length=100)
    price = models.DecimalField(max_digits=10, decimal_places=2)
    category = models.CharField(max_length=50)

    objects = ProductManager()
```

Slide 2: Chaining Custom Manager Methods

Chaining custom manager methods allows you to build complex queries in a readable and expressive way. You can combine multiple methods to create sophisticated filters and retrieve specific data sets.

```python
    def expensive_products(self):
        return self.filter(price__gt=100)

    def get_filtered_products(self, category):
        return self.filter(category=category)

# Usage
expensive_electronics = Product.objects.expensive_products().get_filtered_products('Electronics')
```

Slide 3: Enhancing Readability with Method Chaining

Method chaining improves code readability by breaking down complex queries into smaller, more manageable pieces. This approach makes your code more maintainable and easier to understand.

```python
    def in_stock(self):
        return self.filter(stock__gt=0)

    def on_sale(self):
        return self.filter(discount__gt=0)

    def by_category(self, category):
        return self.filter(category=category)

# Usage
in_stock_electronics_on_sale = (
    Product.objects
    .in_stock()
    .on_sale()
    .by_category('Electronics')
)
```

Slide 4: Extending QuerySet Functionality

Custom managers allow you to extend the functionality of QuerySets, providing additional methods that can be used in queries.

```python
from django.db.models import Avg

class ProductManager(models.Manager):
    def with_average_rating(self):
        return self.annotate(avg_rating=Avg('review__rating'))

    def top_rated(self, threshold=4.0):
        return self.with_average_rating().filter(avg_rating__gte=threshold)

class Product(models.Model):
    name = models.CharField(max_length=100)
    category = models.CharField(max_length=50)
    objects = ProductManager()

class Review(models.Model):
    product = models.ForeignKey(Product, on_delete=models.CASCADE)
    rating = models.IntegerField()

# Usage
top_rated_products = Product.objects.top_rated()
```

Slide 5: Combining Custom Managers with F() Expressions

F() expressions in Django allow you to refer to model field values within queries. By combining custom managers with F() expressions, you can create powerful and efficient queries.

```python

class ProductManager(models.Manager):
    def discounted_products(self):
        return self.filter(sale_price__lt=F('regular_price'))

    def significant_discount(self, percentage):
        return self.annotate(
            discount_percentage=(F('regular_price') - F('sale_price')) / F('regular_price') * 100
        ).filter(discount_percentage__gte=percentage)

class Product(models.Model):
    name = models.CharField(max_length=100)
    regular_price = models.DecimalField(max_digits=10, decimal_places=2)
    sale_price = models.DecimalField(max_digits=10, decimal_places=2)
    objects = ProductManager()

# Usage
heavily_discounted = Product.objects.significant_discount(30)
```

Slide 6: Implementing Complex Filtering Logic

Custom managers enable you to implement complex filtering logic that might be cumbersome to express in a single query. This approach improves code organization and reusability.

```python

class BookManager(models.Manager):
    def search(self, query):
        return self.filter(
            Q(title__icontains=query) |
            Q(author__name__icontains=query) |
            Q(genre__name__icontains=query)
        )

    def available(self):
        return self.filter(is_available=True)

    def recent(self, days=30):
        from datetime import timedelta
        from django.utils import timezone
        recent_date = timezone.now() - timedelta(days=days)
        return self.filter(publication_date__gte=recent_date)

class Book(models.Model):
    title = models.CharField(max_length=200)
    author = models.ForeignKey('Author', on_delete=models.CASCADE)
    genre = models.ForeignKey('Genre', on_delete=models.CASCADE)
    is_available = models.BooleanField(default=True)
    publication_date = models.DateField()
    objects = BookManager()

# Usage
recent_available_books = Book.objects.recent().available()
search_results = Book.objects.search('Python').available()
```

Slide 7: Optimizing Queries with Prefetch Related

Custom managers can be used to optimize queries by incorporating prefetch\_related() calls. This technique can significantly reduce the number of database queries for related objects.

```python
    def with_books_and_reviews(self):
        return self.prefetch_related(
            'books',
            'books__reviews'
        )

class Author(models.Model):
    name = models.CharField(max_length=100)
    objects = AuthorManager()

class Book(models.Model):
    title = models.CharField(max_length=200)
    author = models.ForeignKey(Author, related_name='books', on_delete=models.CASCADE)

class Review(models.Model):
    book = models.ForeignKey(Book, related_name='reviews', on_delete=models.CASCADE)
    content = models.TextField()

# Usage
authors_with_data = Author.objects.with_books_and_reviews()

for author in authors_with_data:
    print(f"Author: {author.name}")
    for book in author.books.all():
        print(f"  Book: {book.title}")
        for review in book.reviews.all():
            print(f"    Review: {review.content[:50]}...")
```

Slide 8: Implementing Custom Aggregations

Custom managers can implement complex aggregations that go beyond simple counts or averages. This example shows how to calculate a weighted average using a custom manager method.

```python
from django.db.models.functions import Cast

class ProductManager(models.Manager):
    def weighted_average_rating(self):
        return self.annotate(
            weighted_avg=Sum(F('reviews__rating') * F('reviews__weight')) / Cast(Sum('reviews__weight'), FloatField())
        )

class Product(models.Model):
    name = models.CharField(max_length=100)
    objects = ProductManager()

class Review(models.Model):
    product = models.ForeignKey(Product, related_name='reviews', on_delete=models.CASCADE)
    rating = models.IntegerField()
    weight = models.IntegerField(default=1)

# Usage
products_with_weighted_ratings = Product.objects.weighted_average_rating()

for product in products_with_weighted_ratings:
    print(f"{product.name}: Weighted Average Rating = {product.weighted_avg:.2f}")
```

Slide 9: Implementing Time-based Queries

Custom managers can simplify time-based queries, making it easy to filter objects based on various time criteria.

```python
from datetime import timedelta

class EventManager(models.Manager):
    def upcoming(self):
        return self.filter(start_time__gte=timezone.now())

    def past(self):
        return self.filter(end_time__lt=timezone.now())

    def this_week(self):
        today = timezone.now().date()
        week_start = today - timedelta(days=today.weekday())
        week_end = week_start + timedelta(days=7)
        return self.filter(start_time__date__range=[week_start, week_end])

class Event(models.Model):
    title = models.CharField(max_length=200)
    start_time = models.DateTimeField()
    end_time = models.DateTimeField()
    objects = EventManager()

# Usage
upcoming_events = Event.objects.upcoming()
past_events = Event.objects.past()
this_week_events = Event.objects.this_week()

print("Upcoming events:", upcoming_events.count())
print("Past events:", past_events.count())
print("Events this week:", this_week_events.count())
```

Slide 10: Implementing Geo-spatial Queries

Custom managers can be used to implement geo-spatial queries, allowing you to find objects based on their geographical location.

```python
from django.contrib.gis.measure import D
from django.contrib.gis.geos import Point

class LocationManager(models.Manager):
    def nearby(self, latitude, longitude, km):
        user_location = Point(longitude, latitude, srid=4326)
        return self.filter(point__distance_lte=(user_location, D(km=km)))

class Location(models.Model):
    name = models.CharField(max_length=100)
    point = models.PointField()
    objects = LocationManager()

# Usage
nearby_locations = Location.objects.nearby(latitude=40.7128, longitude=-74.0060, km=5)

for location in nearby_locations:
    distance = location.point.distance(Point(-74.0060, 40.7128, srid=4326)) * 100  # approximate km
    print(f"{location.name}: {distance:.2f} km away")
```

Slide 11: Implementing Full-Text Search

Custom managers can be used to implement full-text search functionality, providing more advanced search capabilities than simple LIKE queries.

```python

class ArticleManager(models.Manager):
    def search(self, query):
        search_vector = SearchVector('title', weight='A') + SearchVector('content', weight='B')
        search_query = SearchQuery(query)
        return self.annotate(
            rank=SearchRank(search_vector, search_query)
        ).filter(rank__gte=0.3).order_by('-rank')

class Article(models.Model):
    title = models.CharField(max_length=200)
    content = models.TextField()
    objects = ArticleManager()

# Usage
search_results = Article.objects.search('Django ORM')

for article in search_results:
    print(f"{article.title}: Relevance = {article.rank:.2f}")
```

Slide 12: Implementing Caching with Custom Managers

Custom managers can be used to implement caching strategies, improving performance for frequently accessed data.

```python
from django.db import models

class CachingManager(models.Manager):
    def get_cached(self, **kwargs):
        cache_key = f"product_{kwargs['id']}"
        cached_product = cache.get(cache_key)
        
        if not cached_product:
            cached_product = super().get(**kwargs)
            cache.set(cache_key, cached_product, timeout=3600)  # Cache for 1 hour
        
        return cached_product

class Product(models.Model):
    name = models.CharField(max_length=100)
    price = models.DecimalField(max_digits=10, decimal_places=2)
    objects = CachingManager()

# Usage
product = Product.objects.get_cached(id=1)
print(f"Product: {product.name}, Price: ${product.price}")
```

Slide 13: Real-Life Example: Content Management System

In this example, we'll create a custom manager for a blog post model in a content management system. The manager will include methods for retrieving published posts, featured posts, and posts by category.

```python
from django.utils import timezone

class PostManager(models.Manager):
    def published(self):
        return self.filter(status='published', publish_date__lte=timezone.now())

    def featured(self):
        return self.published().filter(featured=True)

    def by_category(self, category_slug):
        return self.published().filter(category__slug=category_slug)

class Category(models.Model):
    name = models.CharField(max_length=100)
    slug = models.SlugField(unique=True)

class Post(models.Model):
    STATUS_CHOICES = (
        ('draft', 'Draft'),
        ('published', 'Published'),
    )
    title = models.CharField(max_length=200)
    content = models.TextField()
    status = models.CharField(max_length=10, choices=STATUS_CHOICES, default='draft')
    publish_date = models.DateTimeField(default=timezone.now)
    category = models.ForeignKey(Category, on_delete=models.CASCADE)
    featured = models.BooleanField(default=False)
    objects = PostManager()

# Usage
published_posts = Post.objects.published()
featured_posts = Post.objects.featured()
tech_posts = Post.objects.by_category('technology')

print(f"Published posts: {published_posts.count()}")
print(f"Featured posts: {featured_posts.count()}")
print(f"Technology posts: {tech_posts.count()}")
```

Slide 14: Real-Life Example: E-commerce Product Recommendations

In this example, we'll create a custom manager for an e-commerce product model that includes methods for retrieving related products and generating personalized recommendations.

```python
from django.db.models import Count, F

class ProductManager(models.Manager):
    def related_products(self, product, limit=5):
        return self.filter(category=product.category).exclude(id=product.id)[:limit]

    def personalized_recommendations(self, user, limit=10):
        user_categories = user.orders.values('products__category').annotate(
            count=Count('products__category')
        ).order_by('-count').values_list('products__category', flat=True)

        return self.filter(category__in=user_categories).annotate(
            relevance=Count('category', filter=F('category__in=user_categories'))
        ).order_by('-relevance', '-average_rating')[:limit]

class Category(models.Model):
    name = models.CharField(max_length=100)

class Product(models.Model):
    name = models.CharField(max_length=200)
    category = models.ForeignKey(Category, on_delete=models.CASCADE)
    price = models.DecimalField(max_digits=10, decimal_places=2)
    average_rating = models.FloatField(default=0.0)
    objects = ProductManager()

class Order(models.Model):
    user = models.ForeignKey('auth.User', related_name='orders', on_delete=models.CASCADE)
    products = models.ManyToManyField(Product)

# Usage
product = Product.objects.get(id=1)
related_products = Product.objects.related_products(product)

user = User.objects.get(id=1)
recommendations = Product.objects.personalized_recommendations(user)

print("Related Products:")
for related_product in related_products:
    print(f"- {related_product.name}")

print("\nPersonalized Recommendations:")
for recommendation in recommendations:
    print(f"- {recommendation.name} (Relevance: {recommendation.relevance})")
```

Slide 15: Additional Resources

For more information on Django custom managers and advanced ORM techniques, consider exploring the following resources:

1. Django Documentation on Managers: [https://docs.djangoproject.com/en/stable/topics/db/managers/](https://docs.djangoproject.com/en/stable/topics/db/managers/)
2. Django ORM Cookbook: [https://books.agiliq.com/projects/django-orm-cookbook/en/latest/](https://books.agiliq.com/projects/django-orm-cookbook/en/latest/)
3. Django QuerySet API reference: [https://docs.djangoproject.com/en/stable/ref/models](https://docs.djangoproject.com/en/stable/ref/models)


