## Comparing Python Web Frameworks Django Flask FastAPI
Slide 1: Django: A Powerful Web Framework for Python

Django is a high-level Python web framework that encourages rapid development and clean, pragmatic design. It follows the model-template-view (MTV) architectural pattern and provides a robust set of tools for building scalable web applications. Django's philosophy emphasizes the principle of "Don't Repeat Yourself" (DRY), which promotes code reusability and maintainability.

Slide 2: Source Code for Django: A Powerful Web Framework for Python

```python
# Example of a simple Django view
from django.http import HttpResponse
from django.views import View

class HelloWorldView(View):
    def get(self, request):
        return HttpResponse("Hello, World!")

# Example of URL routing in Django
from django.urls import path
from .views import HelloWorldView

urlpatterns = [
    path('hello/', HelloWorldView.as_view(), name='hello_world'),
]
```

Slide 3: Django's Model-Template-View (MTV) Architecture

Django's MTV architecture separates the application logic into three interconnected components:

1.  Model: Defines the data structure and handles database operations.
2.  Template: Manages the presentation layer and renders dynamic content.
3.  View: Processes requests, interacts with models, and returns responses.

This separation of concerns enhances code organization and maintainability.

Slide 4: Source Code for Django's Model-Template-View (MTV) Architecture

```python
# Model example
from django.db import models

class Book(models.Model):
    title = models.CharField(max_length=200)
    author = models.CharField(max_length=100)
    publication_date = models.DateField()

# View example
from django.shortcuts import render
from .models import Book

def book_list(request):
    books = Book.objects.all()
    return render(request, 'book_list.html', {'books': books})

# Template example (book_list.html)
"""
{% for book in books %}
    <h2>{{ book.title }}</h2>
    <p>Author: {{ book.author }}</p>
    <p>Published: {{ book.publication_date }}</p>
{% endfor %}
"""
```

Slide 5: Django's ORM: Simplifying Database Operations

Django's Object-Relational Mapping (ORM) provides an abstraction layer between Python code and the database. It allows developers to interact with databases using Python objects and methods, eliminating the need for writing raw SQL queries in most cases. This abstraction enhances code readability and portability across different database systems.

Slide 6: Source Code for Django's ORM: Simplifying Database Operations

```python
# Creating and saving a new book
new_book = Book(title="Django for Beginners", author="John Doe", publication_date="2024-01-01")
new_book.save()

# Querying books
recent_books = Book.objects.filter(publication_date__year=2024)
books_by_author = Book.objects.filter(author="John Doe")

# Updating a book
book_to_update = Book.objects.get(id=1)
book_to_update.title = "Updated Title"
book_to_update.save()

# Deleting a book
book_to_delete = Book.objects.get(id=2)
book_to_delete.delete()
```

Slide 7: Django Forms: Simplifying User Input Handling

Django Forms provide a powerful way to handle user input in web applications. They automatically generate HTML form fields, validate user input, and convert input data into Python types. This feature significantly reduces the amount of code needed to process and validate form submissions, enhancing both security and user experience.

Slide 8: Source Code for Django Forms: Simplifying User Input Handling

```python
from django import forms
from .models import Book

class BookForm(forms.ModelForm):
    class Meta:
        model = Book
        fields = ['title', 'author', 'publication_date']

# View to handle form submission
from django.shortcuts import render, redirect

def add_book(request):
    if request.method == 'POST':
        form = BookForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('book_list')
    else:
        form = BookForm()
    return render(request, 'add_book.html', {'form': form})
```

Slide 9: Django's Admin Interface: Rapid Development Tool

Django's admin interface is a powerful feature that automatically generates an administration panel for your models. It provides a user-friendly interface for managing application data without writing additional code. This feature is particularly useful for prototyping, internal tools, and content management systems.

Slide 10: Source Code for Django's Admin Interface: Rapid Development Tool

```python
from django.contrib import admin
from .models import Book

class BookAdmin(admin.ModelAdmin):
    list_display = ('title', 'author', 'publication_date')
    list_filter = ('author', 'publication_date')
    search_fields = ('title', 'author')

admin.site.register(Book, BookAdmin)

# In urls.py
from django.contrib import admin
from django.urls import path

urlpatterns = [
    path('admin/', admin.site.urls),
    # Other URL patterns...
]
```

Slide 11: Django's Security Features

Django prioritizes security and includes several built-in features to protect applications:

1.  Cross-Site Scripting (XSS) protection
2.  Cross-Site Request Forgery (CSRF) protection
3.  SQL injection prevention
4.  Clickjacking protection
5.  Secure password hashing

These features help developers build secure applications by default, reducing the risk of common web vulnerabilities.

Slide 12: Source Code for Django's Security Features

```python
# CSRF protection in forms
from django.views.decorators.csrf import csrf_protect

@csrf_protect
def my_view(request):
    # View logic here
    pass

# Password hashing
from django.contrib.auth.models import User

user = User.objects.create_user('johndoe', 'john@example.com', 'password123')
# Password is automatically hashed

# XSS protection in templates
"""
{{ user_input|escape }}
"""

# Settings for additional security
SECURE_BROWSER_XSS_FILTER = True
SECURE_CONTENT_TYPE_NOSNIFF = True
X_FRAME_OPTIONS = 'DENY'
```

Slide 13: Real-Life Example: Building a Blog Application with Django

Let's create a simple blog application to demonstrate Django's capabilities. We'll implement basic CRUD operations for blog posts, including creating, reading, updating, and deleting posts.

Slide 14: Source Code for Real-Life Example: Building a Blog Application with Django

```python
# models.py
from django.db import models

class BlogPost(models.Model):
    title = models.CharField(max_length=200)
    content = models.TextField()
    pub_date = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.title

# views.py
from django.shortcuts import render, get_object_or_404, redirect
from .models import BlogPost
from .forms import BlogPostForm

def post_list(request):
    posts = BlogPost.objects.order_by('-pub_date')
    return render(request, 'post_list.html', {'posts': posts})

def post_detail(request, pk):
    post = get_object_or_404(BlogPost, pk=pk)
    return render(request, 'post_detail.html', {'post': post})

def post_new(request):
    if request.method == "POST":
        form = BlogPostForm(request.POST)
        if form.is_valid():
            post = form.save()
            return redirect('post_detail', pk=post.pk)
    else:
        form = BlogPostForm()
    return render(request, 'post_edit.html', {'form': form})
```

Slide 15: Real-Life Example: RESTful API with Django REST Framework

Django REST Framework (DRF) is a powerful toolkit for building Web APIs. It's built on top of Django and provides features like serialization, authentication, and viewsets. Let's create a simple API for our blog application.

Slide 16: Source Code for Real-Life Example: RESTful API with Django REST Framework

```python
# serializers.py
from rest_framework import serializers
from .models import BlogPost

class BlogPostSerializer(serializers.ModelSerializer):
    class Meta:
        model = BlogPost
        fields = ['id', 'title', 'content', 'pub_date']

# views.py
from rest_framework import viewsets
from .models import BlogPost
from .serializers import BlogPostSerializer

class BlogPostViewSet(viewsets.ModelViewSet):
    queryset = BlogPost.objects.all().order_by('-pub_date')
    serializer_class = BlogPostSerializer

# urls.py
from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import BlogPostViewSet

router = DefaultRouter()
router.register(r'posts', BlogPostViewSet)

urlpatterns = [
    path('api/', include(router.urls)),
]
```

Slide 17: Additional Resources

1.  Django Documentation: [https://docs.djangoproject.com/](https://docs.djangoproject.com/)
2.  Django REST Framework Documentation: [https://www.django-rest-framework.org/](https://www.django-rest-framework.org/)
3.  "Two Scoops of Django" by Daniel and Audrey Roy Greenfeld (Book)
4.  Django Girls Tutorial: [https://tutorial.djangogirls.org/](https://tutorial.djangogirls.org/)
5.  Django Channels for real-time functionality: [https://channels.readthedocs.io/](https://channels.readthedocs.io/)

For academic papers related to Django and web development, you can explore:

1.  ArXiv:2104.00820 - "A Comparative Study of Web Development Methodologies"
2.  ArXiv:1910.03732 - "Towards a Framework for Evaluating and Comparing Web Frameworks"

These resources will help you deepen your understanding of Django and its ecosystem.

