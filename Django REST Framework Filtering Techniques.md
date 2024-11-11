## Django REST Framework Filtering Techniques
Slide 1: Basic DRF Filtering Setup

Django Rest Framework filtering requires initial configuration and package installation. The django-filter package extends DRF's filtering capabilities by providing a comprehensive set of filter types and customization options for building flexible API endpoints.

```python
# settings.py
INSTALLED_APPS = [
    'django_filters',
    'rest_framework',
]

REST_FRAMEWORK = {
    'DEFAULT_FILTER_BACKENDS': [
        'django_filters.rest_framework.DjangoFilterBackend',
    ]
}
```

Slide 2: Book Model Definition

The Book model serves as our data structure foundation, incorporating essential fields for demonstrating various filtering techniques. This implementation includes fields commonly used in real-world applications with appropriate field types.

```python
# models.py
from django.db import models

class Book(models.Model):
    title = models.CharField(max_length=200)
    author = models.CharField(max_length=100)
    publication_date = models.DateField()
    price = models.DecimalField(max_digits=6, decimal_places=2)
    genre = models.CharField(max_length=50)
    is_available = models.BooleanField(default=True)
    rating = models.FloatField()

    def __str__(self):
        return self.title
```

Slide 3: Custom Filter Class

A custom filter class provides granular control over filtering behavior, allowing for complex queries and field-specific filtering logic. This implementation showcases common filtering patterns used in production environments.

```python
# filters.py
import django_filters
from .models import Book

class BookFilter(django_filters.FilterSet):
    title = django_filters.CharFilter(lookup_expr='icontains')
    min_price = django_filters.NumberFilter(field_name='price', lookup_expr='gte')
    max_price = django_filters.NumberFilter(field_name='price', lookup_expr='lte')
    publication_date = django_filters.DateFromToRangeFilter()
    
    class Meta:
        model = Book
        fields = ['genre', 'author', 'is_available', 'rating']
```

Slide 4: ViewSet Implementation

The ViewSet implementation incorporates the custom filter class and provides the foundation for API endpoints. This setup enables automatic filtering based on query parameters while maintaining clean code structure.

```python
# views.py
from rest_framework import viewsets
from .models import Book
from .serializers import BookSerializer
from .filters import BookFilter

class BookViewSet(viewsets.ModelViewSet):
    queryset = Book.objects.all()
    serializer_class = BookSerializer
    filterset_class = BookFilter
    
    def get_queryset(self):
        return Book.objects.all().order_by('-publication_date')
```

Slide 5: Advanced Filter Methods

Advanced filtering methods enable complex query constructions and custom filtering logic. This implementation demonstrates how to create dynamic filters based on multiple conditions and custom business rules.

```python
# filters.py
class AdvancedBookFilter(django_filters.FilterSet):
    price_range = django_filters.CharFilter(method='filter_price_range')
    rating_above = django_filters.NumberFilter(method='filter_rating_above')
    
    def filter_price_range(self, queryset, name, value):
        try:
            min_price, max_price = map(float, value.split(','))
            return queryset.filter(price__gte=min_price, price__lte=max_price)
        except (ValueError, AttributeError):
            return queryset
            
    def filter_rating_above(self, queryset, name, value):
        return queryset.filter(rating__gte=value)
```

Slide 6: Custom Filter Backend

Creating a custom filter backend allows for specialized filtering logic that can be reused across multiple views. This implementation shows how to extend DRF's filter backend capabilities.

```python
# backends.py
from rest_framework import filters

class CustomBookFilterBackend(filters.BaseFilterBackend):
    def filter_queryset(self, request, queryset, view):
        genre_priority = request.query_params.get('genre_priority', None)
        
        if genre_priority:
            return queryset.filter(genre=genre_priority).order_by('-rating')
            
        return queryset.order_by('-publication_date')
```

Slide 7: Implementing Search Functionality

Search functionality extends filtering capabilities by allowing text-based queries across multiple fields. This implementation demonstrates how to integrate search with existing filters for comprehensive data retrieval.

```python
# views.py
from rest_framework import filters

class BookViewSet(viewsets.ModelViewSet):
    queryset = Book.objects.all()
    serializer_class = BookSerializer
    filterset_class = BookFilter
    search_fields = ['title', 'author', 'genre']
    filter_backends = [
        django_filters.rest_framework.DjangoFilterBackend,
        filters.SearchFilter,
        filters.OrderingFilter
    ]
```

Slide 8: Complex Query Filtering

Complex query filtering enables advanced data retrieval patterns using Q objects and custom filter methods. This implementation showcases how to handle intricate filtering requirements in real-world scenarios.

```python
# filters.py
from django.db.models import Q

class ComplexBookFilter(django_filters.FilterSet):
    keyword = django_filters.CharFilter(method='filter_by_keyword')
    date_range = django_filters.DateFromToRangeFilter(field_name='publication_date')
    
    def filter_by_keyword(self, queryset, name, value):
        return queryset.filter(
            Q(title__icontains=value) |
            Q(author__icontains=value) |
            Q(genre__icontains=value)
        )
```

Slide 9: Performance Optimization

Performance optimization for filtered queries involves careful consideration of database operations and query optimization. This implementation demonstrates techniques to maintain efficiency with large datasets.

```python
# views.py
from django.db.models import Prefetch
from django.utils.decorators import method_decorator
from django.views.decorators.cache import cache_page

class OptimizedBookViewSet(viewsets.ModelViewSet):
    queryset = Book.objects.select_related('author').prefetch_related('genre')
    serializer_class = BookSerializer
    filterset_class = BookFilter
    
    @method_decorator(cache_page(60 * 15))  # Cache for 15 minutes
    def list(self, request, *args, **kwargs):
        return super().list(request, *args, **kwargs)
```

Slide 10: Custom Pagination with Filtering

Implementing custom pagination alongside filtering ensures efficient data delivery and resource utilization. This setup demonstrates how to handle large datasets with proper pagination controls.

```python
# pagination.py
from rest_framework.pagination import PageNumberPagination
from rest_framework.response import Response

class CustomBookPagination(PageNumberPagination):
    page_size = 20
    page_size_query_param = 'page_size'
    max_page_size = 100
    
    def get_paginated_response(self, data):
        return Response({
            'links': {
                'next': self.get_next_link(),
                'previous': self.get_previous_link()
            },
            'count': self.page.paginator.count,
            'total_pages': self.page.paginator.num_pages,
            'results': data
        })
```

Slide 11: Real-world Example - Book Recommendation System

This implementation showcases a practical application of filtering in a book recommendation system, combining multiple filtering techniques to deliver personalized results.

```python
# views.py
class BookRecommendationViewSet(viewsets.ModelViewSet):
    serializer_class = BookSerializer
    
    def get_queryset(self):
        user_preferences = self.request.user.preferences
        base_queryset = Book.objects.all()
        
        return base_queryset.filter(
            genre__in=user_preferences.favorite_genres,
            rating__gte=4.0
        ).exclude(
            id__in=user_preferences.read_books.all()
        ).order_by('-rating')[:10]
```

Slide 12: Results for Book Recommendation System

```python
# Example Output
{
    "recommendations": [
        {
            "id": 1,
            "title": "The Design of Everyday Things",
            "author": "Don Norman",
            "rating": 4.8,
            "genre": "Technology",
            "match_score": 0.95
        },
        {
            "id": 15,
            "title": "Clean Code",
            "author": "Robert C. Martin",
            "rating": 4.7,
            "genre": "Technology",
            "match_score": 0.92
        }
    ],
    "metadata": {
        "total_matches": 10,
        "processing_time": "0.123s"
    }
}
```

Slide 13: Advanced Filter Chaining

Advanced filter chaining allows for complex filtering scenarios by combining multiple filters dynamically. This implementation demonstrates how to create modular and reusable filter chains.

```python
# filters.py
class ChainableBookFilter(django_filters.FilterSet):
    price_range = django_filters.RangeFilter(field_name='price')
    publication_year = django_filters.NumberFilter(
        field_name='publication_date',
        lookup_expr='year'
    )
    
    @property
    def qs(self):
        base_qs = super().qs
        if self.request.query_params.get('bestseller'):
            base_qs = base_qs.filter(rating__gte=4.5)
        return base_qs
```

