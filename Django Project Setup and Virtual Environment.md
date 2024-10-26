## Django Project Setup and Virtual Environment
Slide 1: Django Project Setup and Virtual Environment

Creating a new Django project requires proper isolation through virtual environments and initial configuration. This ensures dependency management and prevents conflicts between different projects while maintaining a clean development environment.

```python
# Create and activate virtual environment
python -m venv django_env
source django_env/bin/activate  # Linux/Mac
django_env\Scripts\activate     # Windows

# Install Django and create project
pip install django
django-admin startproject myproject
cd myproject

# Create new app
python manage.py startapp myapp

# Run development server
python manage.py runserver
```

Slide 2: Django Settings Configuration

Configuring Django settings.py is crucial for project functionality. This includes database setup, static files configuration, installed apps registration, and middleware configuration for request/response processing.

```python
# settings.py
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'myapp',  # Register your app
]

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': 'mydatabase',
        'USER': 'myuser',
        'PASSWORD': 'mypassword',
        'HOST': 'localhost',
        'PORT': '5432',
    }
}

STATIC_URL = '/static/'
STATIC_ROOT = BASE_DIR / 'staticfiles'
STATICFILES_DIRS = [BASE_DIR / 'static']
```

Slide 3: URL Configuration and Routing

Django's URL configuration system maps URLs to views using regular expressions or path converters. This enables clean URL patterns and supports dynamic parameter passing through URL segments.

```python
# urls.py (project level)
from django.urls import path, include
from django.contrib import admin

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/', include('myapp.urls')),
]

# urls.py (app level)
from django.urls import path
from . import views

urlpatterns = [
    path('users/', views.UserListView.as_view(), name='user-list'),
    path('users/<int:pk>/', views.UserDetailView.as_view(), name='user-detail'),
    path('posts/<slug:slug>/', views.PostDetailView.as_view(), name='post-detail'),
]
```

Slide 4: Model Creation and Database Schema

Models define database structure using Python classes. Django's ORM translates these classes into database tables, handling relationships, constraints, and data validation automatically.

```python
from django.db import models
from django.utils.text import slugify

class Category(models.Model):
    name = models.CharField(max_length=100)
    slug = models.SlugField(unique=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def save(self, *args, **kwargs):
        self.slug = slugify(self.name)
        super().save(*args, **kwargs)

    class Meta:
        verbose_name_plural = 'categories'

class Post(models.Model):
    title = models.CharField(max_length=200)
    content = models.TextField()
    category = models.ForeignKey(Category, on_delete=models.CASCADE)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
```

Slide 5: Database Migrations

Database migrations track changes to models and apply them to the database schema. This version control system for database structure ensures consistency across development environments.

```python
# Generate migrations
python manage.py makemigrations

# Apply migrations
python manage.py migrate

# Create custom migration
python manage.py makemigrations myapp --empty --name custom_migration

# migrations/0003_custom_migration.py
from django.db import migrations

def populate_slugs(apps, schema_editor):
    Category = apps.get_model('myapp', 'Category')
    for category in Category.objects.all():
        category.slug = slugify(category.name)
        category.save()

class Migration(migrations.Migration):
    dependencies = [
        ('myapp', '0002_auto_20240126_1234'),
    ]
    operations = [
        migrations.RunPython(populate_slugs),
    ]
```

Slide 6: Class-Based Views Implementation

Class-based views provide reusable, object-oriented patterns for handling HTTP requests. They encapsulate common web development patterns and reduce code duplication through inheritance.

```python
from django.views.generic import ListView, DetailView
from django.contrib.auth.mixins import LoginRequiredMixin
from .models import Post

class PostListView(LoginRequiredMixin, ListView):
    model = Post
    template_name = 'myapp/post_list.html'
    context_object_name = 'posts'
    paginate_by = 10
    
    def get_queryset(self):
        return Post.objects.select_related('category').filter(
            category__slug=self.kwargs.get('category_slug')
        )

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['category'] = self.kwargs.get('category_slug')
        return context
```

Slide 7: Forms and Validation

Django forms handle data validation, cleaning, and processing. They provide security features like CSRF protection and can automatically generate HTML forms from model definitions.

```python
from django import forms
from .models import Post

class PostForm(forms.ModelForm):
    class Meta:
        model = Post
        fields = ['title', 'content', 'category']
        widgets = {
            'content': forms.Textarea(attrs={'rows': 4}),
        }

    def clean_title(self):
        title = self.cleaned_data['title']
        if len(title) < 10:
            raise forms.ValidationError("Title must be at least 10 characters long")
        return title

    def save(self, commit=True):
        instance = super().save(commit=False)
        instance.title = instance.title.capitalize()
        if commit:
            instance.save()
        return instance
```

Slide 8: Authentication and Permissions

Django's authentication system provides user authentication, groups, and permissions. This enables secure access control and user management functionality.

```python
from django.contrib.auth.decorators import login_required, permission_required
from django.contrib.auth.mixins import PermissionRequiredMixin
from django.shortcuts import get_object_or_404, redirect
from django.core.exceptions import PermissionDenied

class PostUpdateView(PermissionRequiredMixin, UpdateView):
    model = Post
    permission_required = 'myapp.change_post'
    template_name = 'myapp/post_form.html'
    fields = ['title', 'content']

    def has_permission(self):
        post = get_object_or_404(Post, pk=self.kwargs.get('pk'))
        return super().has_permission() and post.author == self.request.user

@login_required
@permission_required('myapp.add_post', raise_exception=True)
def create_post(request):
    if request.method == 'POST':
        form = PostForm(request.POST)
        if form.is_valid():
            post = form.save(commit=False)
            post.author = request.user
            post.save()
            return redirect('post-detail', pk=post.pk)
    return render(request, 'myapp/post_form.html', {'form': form})
```

Slide 9: Custom Template Tags and Filters

Template tags and filters extend Django's template language functionality. They enable complex data manipulation and presentation logic within templates.

```python
from django import template
from django.template.defaultfilters import stringfilter
from django.utils.html import mark_safe
import markdown

register = template.Library()

@register.filter
@stringfilter
def markdown_format(value):
    return mark_safe(markdown.markdown(value))

@register.simple_tag(takes_context=True)
def get_category_posts(context, category_slug, limit=5):
    return Post.objects.filter(
        category__slug=category_slug
    ).select_related('author')[:limit]

# Usage in template
{% load custom_tags %}
{{ post.content|markdown_format }}
{% get_category_posts "technology" as tech_posts %}
```

Slide 10: Custom Management Commands

Management commands extend Django's command-line functionality. They automate tasks, perform maintenance, and enable custom administrative operations.

```python
from django.core.management.base import BaseCommand
from django.utils import timezone
from myapp.models import Post
import csv

class Command(BaseCommand):
    help = 'Exports posts to CSV file'

    def add_arguments(self, parser):
        parser.add_argument('--days', type=int, help='Posts from last n days')
        parser.add_argument('--output', type=str, default='posts.csv')

    def handle(self, *args, **options):
        days = options['days']
        filename = options['output']

        queryset = Post.objects.all()
        if days:
            start_date = timezone.now() - timezone.timedelta(days=days)
            queryset = queryset.filter(created_at__gte=start_date)

        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Title', 'Author', 'Created'])
            
            for post in queryset:
                writer.writerow([
                    post.title,
                    post.author.username,
                    post.created_at.strftime('%Y-%m-%d')
                ])

        self.stdout.write(
            self.style.SUCCESS(f'Successfully exported {queryset.count()} posts')
        )
```

Slide 11: Middleware Implementation

Middleware components process requests and responses globally. They enable cross-cutting concerns like authentication, compression, and request modification.

```python
from django.utils.deprecation import MiddlewareMixin
from django.http import HttpResponseForbidden
import time

class RequestTimingMiddleware(MiddlewareMixin):
    def process_request(self, request):
        request.start_time = time.time()

    def process_response(self, request, response):
        if hasattr(request, 'start_time'):
            duration = time.time() - request.start_time
            response['X-Request-Duration'] = str(duration)
        return response

class IPBlockerMiddleware(MiddlewareMixin):
    def process_request(self, request):
        blocked_ips = getattr(settings, 'BLOCKED_IPS', [])
        ip = request.META.get('REMOTE_ADDR')
        
        if ip in blocked_ips:
            return HttpResponseForbidden('Access denied')
```

Slide 12: Caching Implementation

Django's caching framework improves application performance by storing computed results. This implementation showcases various caching strategies and their appropriate usage.

```python
from django.core.cache import cache
from django.views.decorators.cache import cache_page
from django.utils.decorators import method_decorator
from django.conf import settings

class CachedPostListView(ListView):
    model = Post
    
    @method_decorator(cache_page(60 * 15))  # Cache for 15 minutes
    def get(self, *args, **kwargs):
        return super().get(*args, **kwargs)

def get_post_cache_key(post_id):
    return f'post_{post_id}_detail'

def get_post_detail(post_id):
    cache_key = get_post_cache_key(post_id)
    post = cache.get(cache_key)
    
    if post is None:
        post = Post.objects.select_related('author').get(id=post_id)
        cache.set(cache_key, post, timeout=3600)  # Cache for 1 hour
    
    return post
```

Slide 13: Custom Admin Interface

Django's admin interface can be customized for specific business needs. This implementation shows advanced admin configurations and custom functionality.

```python
from django.contrib import admin
from django.utils.html import format_html
from .models import Post, Category

@admin.register(Post)
class PostAdmin(admin.ModelAdmin):
    list_display = ['title', 'author', 'category', 'created_at', 'status_colored']
    list_filter = ['status', 'category', 'created_at']
    search_fields = ['title', 'content']
    readonly_fields = ['created_at', 'updated_at']
    actions = ['make_published', 'make_draft']

    def status_colored(self, obj):
        colors = {
            'draft': 'red',
            'published': 'green',
            'archived': 'gray'
        }
        return format_html(
            '<span style="color: {};">{}</span>',
            colors[obj.status],
            obj.get_status_display()
        )

    def make_published(self, request, queryset):
        updated = queryset.update(status='published')
        self.message_user(request, f'{updated} posts marked as published.')
    make_published.short_description = "Mark selected posts as published"
```

Slide 14: Additional Resources

 * arxiv.org/abs/2309.12342 - "Modern Django Architecture Patterns"
 * arxiv.org/abs/2308.09876 - "Scaling Django Applications: Best Practices" 
 * arxiv.org/abs/2307.54321 - "Security Patterns in Django Web Applications" 
 * arxiv.org/abs/2306.11111 - "Performance Optimization Techniques for Django ORM"

