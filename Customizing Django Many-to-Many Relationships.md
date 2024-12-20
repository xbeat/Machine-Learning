## Customizing Django Many-to-Many Relationships
Slide 1: Basic Many-to-Many Model Structure

A many-to-many relationship in Django requires careful consideration of the model structure. This slide demonstrates the foundational setup of two models connected through a many-to-many relationship, showing how to establish the basic relationship before adding custom fields.

```python
from django.db import models

class Student(models.Model):
    name = models.CharField(max_length=100)
    email = models.EmailField(unique=True)
    
    def __str__(self):
        return self.name

class Course(models.Model):
    name = models.CharField(max_length=200)
    code = models.CharField(max_length=10, unique=True)
    students = models.ManyToManyField(Student)
    
    def __str__(self):
        return f"{self.code} - {self.name}"
```

Slide 2: Creating a Custom Through Model

The through model serves as an intermediate table that stores additional information about the relationship between two models. This implementation shows how to create a through model that tracks enrollment dates and grades for students in courses.

```python
from django.db import models
from django.utils import timezone

class Student(models.Model):
    name = models.CharField(max_length=100)
    email = models.EmailField(unique=True)

class Course(models.Model):
    name = models.CharField(max_length=200)
    code = models.CharField(max_length=10, unique=True)
    students = models.ManyToManyField(
        Student,
        through='Enrollment',
        through_fields=('course', 'student'),
    )

class Enrollment(models.Model):
    student = models.ForeignKey(Student, on_delete=models.CASCADE)
    course = models.ForeignKey(Course, on_delete=models.CASCADE)
    date_enrolled = models.DateTimeField(default=timezone.now)
    grade = models.DecimalField(
        max_digits=5, 
        decimal_places=2,
        null=True,
        blank=True
    )
```

Slide 3: Adding Custom Methods to Through Model

The through model can be enhanced with custom methods to handle complex operations and validations. This implementation demonstrates how to add business logic to manage enrollment status and grade calculations.

```python
class Enrollment(models.Model):
    STATUS_CHOICES = [
        ('active', 'Active'),
        ('dropped', 'Dropped'),
        ('completed', 'Completed'),
    ]
    
    student = models.ForeignKey(Student, on_delete=models.CASCADE)
    course = models.ForeignKey(Course, on_delete=models.CASCADE)
    date_enrolled = models.DateTimeField(default=timezone.now)
    status = models.CharField(
        max_length=20,
        choices=STATUS_CHOICES,
        default='active'
    )
    grade = models.DecimalField(
        max_digits=5,
        decimal_places=2,
        null=True,
        blank=True
    )
    
    def calculate_final_grade(self, assignments):
        """Calculate final grade based on assignments"""
        if not assignments:
            return None
        total_weight = sum(a.weight for a in assignments)
        weighted_sum = sum(a.grade * a.weight for a in assignments)
        return weighted_sum / total_weight
    
    def can_enroll(self):
        """Check if student can enroll in course"""
        active_enrollments = Enrollment.objects.filter(
            student=self.student,
            status='active'
        ).count()
        return active_enrollments < 5
```

Slide 4: Implementing Custom Managers

Custom managers provide a clean interface for common operations on the through model. This implementation shows how to create specialized queries and validation methods for enrollment management.

```python
class EnrollmentManager(models.Manager):
    def active_enrollments(self):
        return self.filter(status='active')
    
    def completed_with_grade(self, minimum_grade=2.0):
        return self.filter(
            status='completed',
            grade__gte=minimum_grade
        )
    
    def enroll_student(self, student, course):
        if self.filter(student=student, course=course).exists():
            raise ValueError("Student already enrolled in this course")
        
        return self.create(
            student=student,
            course=course,
            status='active'
        )

class Enrollment(models.Model):
    # ... previous fields ...
    
    objects = EnrollmentManager()
    
    class Meta:
        unique_together = ['student', 'course']
```

Slide 5: Adding Data Validation

Proper data validation ensures data integrity in many-to-many relationships. This implementation demonstrates how to add custom validation methods and clean data before saving.

```python
from django.core.exceptions import ValidationError
from django.db import models
import datetime

class Enrollment(models.Model):
    # ... previous fields ...
    
    def clean(self):
        if self.grade is not None and (self.grade < 0 or self.grade > 4.0):
            raise ValidationError({
                'grade': 'Grade must be between 0.0 and 4.0'
            })
        
        if self.status == 'completed' and self.grade is None:
            raise ValidationError({
                'grade': 'Completed courses must have a grade'
            })
    
    def save(self, *args, **kwargs):
        self.full_clean()
        super().save(*args, **kwargs)
```

Slide 6: Implementing Course Prerequisites

The through model can be extended to handle complex course prerequisites. This implementation demonstrates how to create a separate through model for managing course dependencies and validating enrollment eligibility.

```python
class CoursePrerequisite(models.Model):
    course = models.ForeignKey(
        Course,
        on_delete=models.CASCADE,
        related_name='prerequisites'
    )
    prerequisite = models.ForeignKey(
        Course,
        on_delete=models.CASCADE,
        related_name='is_prerequisite_for'
    )
    minimum_grade = models.DecimalField(
        max_digits=3,
        decimal_places=2,
        default=2.0
    )

    def validate_prerequisite(self, student):
        completed_prereq = Enrollment.objects.filter(
            student=student,
            course=self.prerequisite,
            status='completed',
            grade__gte=self.minimum_grade
        ).exists()
        return completed_prereq

    class Meta:
        unique_together = ['course', 'prerequisite']
```

Slide 7: Advanced Queries with Through Models

Understanding how to perform complex queries with through models is essential for efficient data retrieval. This implementation shows advanced query patterns for analyzing enrollment data.

```python
from django.db.models import Avg, Count, Q
from datetime import datetime, timedelta

class Course(models.Model):
    # ... previous fields ...
    
    def get_enrollment_statistics(self):
        return self.enrollment_set.aggregate(
            total_students=Count('student'),
            average_grade=Avg('grade'),
            active_count=Count('id', filter=Q(status='active')),
            dropped_count=Count('id', filter=Q(status='dropped'))
        )
    
    def get_recent_enrollments(self, days=30):
        cutoff_date = datetime.now() - timedelta(days=days)
        return self.enrollment_set.filter(
            date_enrolled__gte=cutoff_date
        ).select_related('student')
```

Slide 8: Implementing Enrollment History

Tracking historical changes in many-to-many relationships provides valuable insights. This implementation demonstrates how to maintain an audit trail of enrollment changes.

```python
class EnrollmentHistory(models.Model):
    enrollment = models.ForeignKey(
        Enrollment,
        on_delete=models.CASCADE,
        related_name='history'
    )
    status_from = models.CharField(max_length=20)
    status_to = models.CharField(max_length=20)
    grade_from = models.DecimalField(
        max_digits=5,
        decimal_places=2,
        null=True
    )
    grade_to = models.DecimalField(
        max_digits=5,
        decimal_places=2,
        null=True
    )
    changed_at = models.DateTimeField(auto_now_add=True)
    changed_by = models.ForeignKey(
        'auth.User',
        on_delete=models.SET_NULL,
        null=True
    )

    @classmethod
    def log_change(cls, enrollment, changed_by, **changes):
        history = cls(
            enrollment=enrollment,
            changed_by=changed_by
        )
        for field, (old_value, new_value) in changes.items():
            setattr(history, f"{field}_from", old_value)
            setattr(history, f"{field}_to", new_value)
        history.save()
```

Slide 9: Custom Signals for Through Models

Signals help maintain data consistency and trigger actions when relationships change. This implementation shows how to use signals with through models effectively.

```python
from django.db.models.signals import pre_save, post_save
from django.dispatch import receiver

@receiver(pre_save, sender=Enrollment)
def enrollment_pre_save(sender, instance, **kwargs):
    if instance.pk:  # Existing enrollment
        old_instance = Enrollment.objects.get(pk=instance.pk)
        if old_instance.status != instance.status:
            # Validate status transition
            valid_transitions = {
                'active': ['dropped', 'completed'],
                'dropped': ['active'],
                'completed': []
            }
            if instance.status not in valid_transitions[old_instance.status]:
                raise ValueError(f"Invalid status transition: {old_instance.status} -> {instance.status}")

@receiver(post_save, sender=Enrollment)
def enrollment_post_save(sender, instance, created, **kwargs):
    if not created:
        old_instance = Enrollment.objects.get(pk=instance.pk)
        changes = {}
        
        if old_instance.status != instance.status:
            changes['status'] = (old_instance.status, instance.status)
        if old_instance.grade != instance.grade:
            changes['grade'] = (old_instance.grade, instance.grade)
            
        if changes:
            EnrollmentHistory.log_change(instance, None, **changes)
```

Slide 10: API Integration for Through Models

Exposing many-to-many relationships through APIs requires careful serialization. This implementation demonstrates how to create API endpoints for enrollment management.

```python
from rest_framework import serializers, viewsets
from rest_framework.decorators import action
from rest_framework.response import Response

class EnrollmentSerializer(serializers.ModelSerializer):
    class Meta:
        model = Enrollment
        fields = ['id', 'student', 'course', 'date_enrolled', 
                 'status', 'grade']
        read_only_fields = ['date_enrolled']

class EnrollmentViewSet(viewsets.ModelViewSet):
    queryset = Enrollment.objects.all()
    serializer_class = EnrollmentSerializer

    @action(detail=True, methods=['post'])
    def change_status(self, request, pk=None):
        enrollment = self.get_object()
        new_status = request.data.get('status')
        
        try:
            old_status = enrollment.status
            enrollment.status = new_status
            enrollment.save()
            
            return Response({
                'success': True,
                'message': f'Status changed from {old_status} to {new_status}'
            })
        except ValueError as e:
            return Response({
                'success': False,
                'error': str(e)
            }, status=400)
```

Slide 11: Bulk Operations with Through Models

Efficient handling of multiple enrollments requires optimized bulk operations. This implementation shows how to manage batch enrollments and updates while maintaining data integrity.

```python
from django.db import transaction
from typing import List, Dict, Any

class BulkEnrollmentManager:
    def __init__(self, course):
        self.course = course
        self.successful_enrollments = []
        self.failed_enrollments = []
    
    @transaction.atomic
    def bulk_enroll(self, student_data: List[Dict[str, Any]]):
        for data in student_data:
            try:
                enrollment = Enrollment.objects.create(
                    course=self.course,
                    student_id=data['student_id'],
                    status='active'
                )
                self.successful_enrollments.append({
                    'student_id': data['student_id'],
                    'enrollment_id': enrollment.id
                })
            except Exception as e:
                self.failed_enrollments.append({
                    'student_id': data['student_id'],
                    'error': str(e)
                })
        
        return {
            'successful': self.successful_enrollments,
            'failed': self.failed_enrollments
        }
```

Slide 12: Performance Optimization for Through Models

Optimizing queries involving through models is crucial for application performance. This implementation demonstrates techniques for reducing database queries and improving response times.

```python
from django.db.models import Prefetch

class CourseManager(models.Manager):
    def get_courses_with_enrollment_stats(self):
        return self.annotate(
            total_enrollments=Count('enrollment'),
            active_enrollments=Count(
                'enrollment',
                filter=Q(enrollment__status='active')
            ),
            avg_grade=Avg('enrollment__grade')
        )
    
    def get_detailed_course_info(self, course_id):
        return self.filter(id=course_id).prefetch_related(
            Prefetch(
                'enrollment_set',
                queryset=Enrollment.objects.select_related('student')
                    .filter(status='active')
            ),
            'prerequisites',
            'is_prerequisite_for'
        ).first()
```

Slide 13: Data Migration and Through Models

Managing schema changes and data migrations for through models requires careful planning. This implementation shows how to handle model changes while preserving relationship data.

```python
from django.db import migrations, models
import django.db.models.deletion

class Migration(migrations.Migration):
    dependencies = [
        ('courses', '0001_initial'),
    ]

    def transfer_enrollment_data(apps, schema_editor):
        OldEnrollment = apps.get_model('courses', 'OldEnrollment')
        NewEnrollment = apps.get_model('courses', 'Enrollment')
        
        for old_enrollment in OldEnrollment.objects.all():
            NewEnrollment.objects.create(
                student_id=old_enrollment.student_id,
                course_id=old_enrollment.course_id,
                date_enrolled=old_enrollment.created_at,
                status='active' if old_enrollment.is_active else 'completed',
                grade=old_enrollment.final_grade
            )

    operations = [
        migrations.RunPython(
            transfer_enrollment_data,
            reverse_code=migrations.RunPython.noop
        ),
    ]
```

Slide 14: Testing Through Models

Comprehensive testing ensures the reliability of many-to-many relationships. This implementation demonstrates how to write tests for through model functionality.

```python
from django.test import TestCase
from django.core.exceptions import ValidationError

class EnrollmentTests(TestCase):
    def setUp(self):
        self.student = Student.objects.create(
            name="Test Student",
            email="test@example.com"
        )
        self.course = Course.objects.create(
            name="Advanced Python",
            code="PY301"
        )

    def test_enrollment_validation(self):
        # Test invalid grade
        with self.assertRaises(ValidationError):
            enrollment = Enrollment.objects.create(
                student=self.student,
                course=self.course,
                grade=5.0  # Invalid grade
            )

        # Test duplicate enrollment
        Enrollment.objects.create(
            student=self.student,
            course=self.course
        )
        with self.assertRaises(ValidationError):
            Enrollment.objects.create(
                student=self.student,
                course=self.course
            )
```

Slide 15: Additional Resources

*   Improving Django's ManyToMany Field Performance: [https://www.dabapps.com/blog/performance-tips-tricks-django-manytomany-field/](https://www.dabapps.com/blog/performance-tips-tricks-django-manytomany-field/)
*   Advanced Database Optimization in Django: [https://docs.djangoproject.com/en/stable/topics/db/optimization/](https://docs.djangoproject.com/en/stable/topics/db/optimization/)
*   Django Through Models Best Practices: [https://django-best-practices.readthedocs.io/en/latest/many\_to\_many.html](https://django-best-practices.readthedocs.io/en/latest/many_to_many.html)
*   Django ORM Cookbook: [https://books.agiliq.com/projects/django-orm-cookbook/](https://books.agiliq.com/projects/django-orm-cookbook/)
*   Django Documentation on Many-to-Many Relationships: [https://docs.djangoproject.com/en/stable/topics/db/examples/many\_to\_many/](https://docs.djangoproject.com/en/stable/topics/db/examples/many_to_many/)

