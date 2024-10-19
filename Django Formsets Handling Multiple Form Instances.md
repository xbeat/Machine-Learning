## Django Formsets Handling Multiple Form Instances
Slide 1: Introduction to Django Formsets

Formsets in Django provide a powerful way to handle multiple form instances in a single request. They're particularly useful when dealing with related data or when you need to create, update, or delete multiple objects at once.

```python
from django import forms
from django.forms import formset_factory

class ItemForm(forms.Form):
    name = forms.CharField(max_length=100)
    quantity = forms.IntegerField(min_value=0)

ItemFormSet = formset_factory(ItemForm, extra=3)
```

Slide 2: Creating a Formset

To create a formset, use the `formset_factory` function. It takes a form class as input and returns a formset class. The `extra` parameter specifies the number of empty forms to display initially.

```python
from django.shortcuts import render
from .forms import ItemFormSet

def manage_items(request):
    if request.method == 'POST':
        formset = ItemFormSet(request.POST)
        if formset.is_valid():
            for form in formset:
                name = form.cleaned_data.get('name')
                quantity = form.cleaned_data.get('quantity')
                # Process the data (e.g., save to database)
    else:
        formset = ItemFormSet()
    return render(request, 'manage_items.html', {'formset': formset})
```

Slide 3: Rendering Formsets in Templates

When rendering formsets in templates, it's crucial to include the management form and iterate over each form in the formset.

```html
<form method="post">
    {% csrf_token %}
    {{ formset.management_form }}
    {% for form in formset %}
        {{ form.as_p }}
    {% endfor %}
    <button type="submit">Submit</button>
</form>
```

Slide 4: Model Formsets

Django also provides model formsets, which are formsets based on a model. They're useful when working directly with model instances.

```python
from django.forms import modelformset_factory
from .models import Item

ItemModelFormSet = modelformset_factory(Item, fields=['name', 'quantity'], extra=2)

def manage_items(request):
    formset = ItemModelFormSet(queryset=Item.objects.all())
    # Process the formset
    return render(request, 'manage_items.html', {'formset': formset})
```

Slide 5: Inline Formsets

Inline formsets are used to handle related objects in forms. They're particularly useful for parent-child relationships.

```python
from django.forms import inlineformset_factory
from .models import Author, Book

BookFormSet = inlineformset_factory(Author, Book, fields=['title', 'publication_date'], extra=1)

def manage_books(request, author_id):
    author = Author.objects.get(id=author_id)
    formset = BookFormSet(instance=author)
    # Process the formset
    return render(request, 'manage_books.html', {'formset': formset})
```

Slide 6: Customizing Formset Behavior

Formsets can be customized to suit specific needs, such as setting minimum or maximum number of forms, or providing custom validation.

```python
from django.forms import BaseFormSet
from django.core.exceptions import ValidationError

class BaseItemFormSet(BaseFormSet):
    def clean(self):
        if any(self.errors):
            return
        names = []
        for form in self.forms:
            name = form.cleaned_data.get('name')
            if name in names:
                raise ValidationError("Items in a set must have distinct names.")
            names.append(name)

ItemFormSet = formset_factory(ItemForm, formset=BaseItemFormSet, min_num=1, validate_min=True, max_num=5, validate_max=True)
```

Slide 7: Handling Form Prefixes

When working with multiple formsets on a single page, it's important to use prefixes to distinguish between them.

```python
def manage_inventory(request):
    item_formset = ItemFormSet(request.POST or None, prefix='items')
    supplier_formset = SupplierFormSet(request.POST or None, prefix='suppliers')
    
    if request.method == 'POST' and item_formset.is_valid() and supplier_formset.is_valid():
        # Process both formsets
        pass
    
    return render(request, 'manage_inventory.html', {
        'item_formset': item_formset,
        'supplier_formset': supplier_formset
    })
```

Slide 8: Dynamic Formsets with JavaScript

To enhance user experience, you can use JavaScript to dynamically add or remove forms in a formset.

```html
<form method="post" id="item-formset">
    {% csrf_token %}
    {{ formset.management_form }}
    <div id="form-container">
        {% for form in formset %}
            <div class="item-form">
                {{ form.as_p }}
                <button type="button" class="remove-form">Remove</button>
            </div>
        {% endfor %}
    </div>
    <button type="button" id="add-form">Add Item</button>
    <button type="submit">Submit</button>
</form>

<script>
    $(document).ready(function() {
        var formCount = {{ formset.total_form_count }};
        $('#add-form').click(function() {
            var form = $('.item-form:first').clone(true);
            form.find('input').val('');
            form.find('input[name]').each(function() {
                var name = $(this).attr('name').replace('-0-', '-' + formCount + '-');
                var id = 'id_' + name;
                $(this).attr({'name': name, 'id': id});
            });
            $('#form-container').append(form);
            formCount++;
            $('#id_form-TOTAL_FORMS').val(formCount);
        });
        $('.remove-form').click(function() {
            $(this).parent().remove();
            formCount--;
            $('#id_form-TOTAL_FORMS').val(formCount);
        });
    });
</script>
```

Slide 9: Real-life Example: Task Management

Let's create a simple task management system where users can add multiple tasks with priorities.

```python
from django import forms
from django.forms import formset_factory

class TaskForm(forms.Form):
    description = forms.CharField(max_length=200, widget=forms.TextInput(attrs={'placeholder': 'Enter task description'}))
    priority = forms.ChoiceField(choices=[('1', 'Low'), ('2', 'Medium'), ('3', 'High')])

TaskFormSet = formset_factory(TaskForm, extra=3)

def manage_tasks(request):
    if request.method == 'POST':
        formset = TaskFormSet(request.POST)
        if formset.is_valid():
            for form in formset:
                if form.cleaned_data:
                    description = form.cleaned_data['description']
                    priority = form.cleaned_data['priority']
                    Task.objects.create(description=description, priority=priority)
            return redirect('task_list')
    else:
        formset = TaskFormSet()
    return render(request, 'manage_tasks.html', {'formset': formset})
```

Slide 10: Real-life Example: Product Variants

Here's an example of using inline formsets to manage product variants in an e-commerce system.

```python
from django.forms import inlineformset_factory
from .models import Product, ProductVariant

class ProductForm(forms.ModelForm):
    class Meta:
        model = Product
        fields = ['name', 'description']

VariantFormSet = inlineformset_factory(
    Product, ProductVariant,
    fields=['size', 'color', 'price'],
    extra=1, can_delete=True
)

def manage_product(request, product_id=None):
    if product_id:
        product = get_object_or_404(Product, id=product_id)
    else:
        product = None

    if request.method == 'POST':
        form = ProductForm(request.POST, instance=product)
        formset = VariantFormSet(request.POST, instance=product)
        if form.is_valid() and formset.is_valid():
            product = form.save()
            formset.save()
            return redirect('product_detail', product_id=product.id)
    else:
        form = ProductForm(instance=product)
        formset = VariantFormSet(instance=product)

    return render(request, 'manage_product.html', {
        'form': form,
        'formset': formset,
    })
```

Slide 11: Formset Validation

Implementing custom validation for formsets ensures data integrity and improves user experience.

```python
from django.core.exceptions import ValidationError
from django.forms import BaseFormSet

class BaseTaskFormSet(BaseFormSet):
    def clean(self):
        if any(self.errors):
            return

        descriptions = []
        high_priority_count = 0

        for form in self.forms:
            if form.cleaned_data:
                description = form.cleaned_data['description']
                priority = form.cleaned_data['priority']

                if description in descriptions:
                    raise ValidationError("Tasks must have unique descriptions.")
                descriptions.append(description)

                if priority == '3':  # High priority
                    high_priority_count += 1

        if high_priority_count > 3:
            raise ValidationError("You can't have more than 3 high-priority tasks.")

TaskFormSet = formset_factory(TaskForm, formset=BaseTaskFormSet, extra=3, max_num=10)
```

Slide 12: Formset Factory Options

The `formset_factory` function provides various options to customize formset behavior.

```python
from django.forms import formset_factory

ItemFormSet = formset_factory(
    ItemForm,
    extra=3,              # Number of extra empty forms
    max_num=10,           # Maximum number of forms
    validate_max=True,    # Enforce max_num
    min_num=1,            # Minimum number of forms
    validate_min=True,    # Enforce min_num
    can_order=True,       # Allow reordering of forms
    can_delete=True       # Allow deletion of forms
)

def manage_items(request):
    if request.method == 'POST':
        formset = ItemFormSet(request.POST)
        if formset.is_valid():
            for form in formset.ordered_forms:
                if form.cleaned_data and not form.cleaned_data.get('DELETE', False):
                    # Process non-deleted forms
                    pass
    else:
        formset = ItemFormSet()
    return render(request, 'manage_items.html', {'formset': formset})
```

Slide 13: Debugging Formsets

When working with formsets, debugging can be challenging. Here are some tips to help identify and resolve issues.

```python
def debug_formset(request):
    if request.method == 'POST':
        formset = ItemFormSet(request.POST)
        if not formset.is_valid():
            for i, form in enumerate(formset):
                if form.errors:
                    print(f"Form {i} errors: {form.errors}")
            print(f"Formset non-form errors: {formset.non_form_errors()}")
        
        # Inspect cleaned data
        for form in formset:
            if form.is_valid():
                print(f"Cleaned data: {form.cleaned_data}")
    
    # Check initial data
    formset = ItemFormSet(initial=[
        {'name': 'Item 1', 'quantity': 5},
        {'name': 'Item 2', 'quantity': 3},
    ])
    
    return render(request, 'debug_formset.html', {'formset': formset})
```

Slide 14: Additional Resources

For more advanced topics and in-depth understanding of Django formsets, consider exploring these resources:

1.  Django Official Documentation on Formsets: [https://docs.djangoproject.com/en/stable/topics/forms/formsets/](https://docs.djangoproject.com/en/stable/topics/forms/formsets/)
2.  Django Forms and Formsets: Advanced techniques [https://arxiv.org/abs/2104.13497](https://arxiv.org/abs/2104.13497)
3.  Django Formset Tutorial on Real Python: [https://realpython.com/django-formsets/](https://realpython.com/django-formsets/)

Remember to always refer to the official Django documentation for the most up-to-date information on formsets and their usage.

