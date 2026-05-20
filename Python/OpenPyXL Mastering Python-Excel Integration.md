## OpenPyXL Mastering Python-Excel Integration
Slide 1: Introduction to OpenPyXL

OpenPyXL is a powerful Python library that enables seamless integration between Python and Microsoft Excel. It allows developers to read, write, and manipulate Excel spreadsheets programmatically, opening up a world of possibilities for data analysis, reporting, and automation.

```python
import openpyxl

# Create a new workbook
workbook = openpyxl.Workbook()
sheet = workbook.active

# Write data to cells
sheet['A1'] = 'Hello'
sheet['B1'] = 'OpenPyXL'

# Save the workbook
workbook.save('example.xlsx')
```

Slide 2: Installing OpenPyXL

Before we dive into using OpenPyXL, let's install it. OpenPyXL can be easily installed using pip, Python's package installer. Open your terminal or command prompt and run the following command to install OpenPyXL:

```python
# In your terminal or command prompt
pip install openpyxl

# Verify installation
import openpyxl
print(openpyxl.__version__)
```

Slide 3: Creating a New Workbook

Let's start by creating a new Excel workbook using OpenPyXL. We'll create a workbook, add some data, and save it as an Excel file.

```python
from openpyxl import Workbook

# Create a new workbook and select the active sheet
wb = Workbook()
sheet = wb.active

# Add data to cells
sheet['A1'] = 'Name'
sheet['B1'] = 'Age'
sheet['A2'] = 'Alice'
sheet['B2'] = 30

# Save the workbook
wb.save('new_workbook.xlsx')

print("Workbook created and saved as 'new_workbook.xlsx'")
```

Slide 4: Reading an Existing Excel File

OpenPyXL allows us to read existing Excel files. We'll open an Excel file, access its sheets, and read cell values.

```python
from openpyxl import load_workbook

# Load the workbook
wb = load_workbook('new_workbook.xlsx')

# Get the active sheet
sheet = wb.active

# Read cell values
name = sheet['A2'].value
age = sheet['B2'].value

print(f"Name: {name}, Age: {age}")

# Output: Name: Alice, Age: 30
```

Slide 5: Writing Data to Cells

Now let's explore how to write data to specific cells in an Excel sheet using OpenPyXL.

```python
from openpyxl import Workbook

wb = Workbook()
sheet = wb.active

# Writing to specific cells
sheet['A1'] = 'Product'
sheet['B1'] = 'Quantity'
sheet['C1'] = 'Price'

products = [('Apple', 50, 0.5), ('Banana', 30, 0.3), ('Orange', 40, 0.4)]

for row, (product, quantity, price) in enumerate(products, start=2):
    sheet.cell(row=row, column=1, value=product)
    sheet.cell(row=row, column=2, value=quantity)
    sheet.cell(row=row, column=3, value=price)

wb.save('inventory.xlsx')
print("Inventory data saved to 'inventory.xlsx'")
```

Slide 6: Formatting Cells

OpenPyXL provides various options for formatting cells. Let's apply some basic formatting to our inventory sheet.

```python
from openpyxl import load_workbook
from openpyxl.styles import Font, Alignment, PatternFill

wb = load_workbook('inventory.xlsx')
sheet = wb.active

# Apply bold font to header row
for cell in sheet[1]:
    cell.font = Font(bold=True)

# Center align all cells
for row in sheet.iter_rows():
    for cell in row:
        cell.alignment = Alignment(horizontal='center')

# Add background color to header
for cell in sheet[1]:
    cell.fill = PatternFill(start_color="FFFF00", end_color="FFFF00", fill_type="solid")

wb.save('inventory_formatted.xlsx')
print("Formatted inventory saved as 'inventory_formatted.xlsx'")
```

Slide 7: Working with Multiple Sheets

Excel workbooks can contain multiple sheets. Let's see how to create, access, and manipulate multiple sheets using OpenPyXL.

```python
from openpyxl import Workbook

wb = Workbook()
default_sheet = wb.active
default_sheet.title = "First Sheet"

# Create additional sheets
wb.create_sheet("Second Sheet")
wb.create_sheet("Third Sheet")

# Access and write to different sheets
sheet1 = wb["First Sheet"]
sheet2 = wb["Second Sheet"]
sheet3 = wb["Third Sheet"]

sheet1['A1'] = "Data in First Sheet"
sheet2['A1'] = "Data in Second Sheet"
sheet3['A1'] = "Data in Third Sheet"

wb.save('multi_sheet_workbook.xlsx')
print("Workbook with multiple sheets saved as 'multi_sheet_workbook.xlsx'")
```

Slide 8: Merging and Unmerging Cells

OpenPyXL allows us to merge and unmerge cells, which is useful for creating headers or organizing data in a specific layout.

```python
from openpyxl import Workbook
from openpyxl.styles import Alignment

wb = Workbook()
sheet = wb.active

# Merge cells
sheet.merge_cells('A1:D1')
sheet['A1'] = 'Merged Cells'
sheet['A1'].alignment = Alignment(horizontal='center')

# Unmerge cells
sheet.unmerge_cells('A1:D1')

# Merge cells in a different way
sheet.merge_cells(start_row=3, start_column=1, end_row=3, end_column=4)
sheet['A3'] = 'Another Merged Row'
sheet['A3'].alignment = Alignment(horizontal='center')

wb.save('merged_cells.xlsx')
print("Workbook with merged cells saved as 'merged_cells.xlsx'")
```

Slide 9: Adding Charts

OpenPyXL supports creating various types of charts. Let's add a simple bar chart to our inventory data.

```python
from openpyxl import load_workbook
from openpyxl.chart import BarChart, Reference

wb = load_workbook('inventory.xlsx')
sheet = wb.active

# Create a bar chart
chart = BarChart()
chart.title = "Product Inventory"
chart.x_axis.title = "Products"
chart.y_axis.title = "Quantity"

# Define data for the chart
data = Reference(sheet, min_col=2, min_row=1, max_row=4, max_col=2)
categories = Reference(sheet, min_col=1, min_row=2, max_row=4)

# Add data to the chart
chart.add_data(data, titles_from_data=True)
chart.set_categories(categories)

# Add the chart to the sheet
sheet.add_chart(chart, "E1")

wb.save('inventory_with_chart.xlsx')
print("Inventory with chart saved as 'inventory_with_chart.xlsx'")
```

Slide 10: Handling Formulas

OpenPyXL can work with Excel formulas. Let's add some formulas to our inventory sheet to calculate total values.

```python
from openpyxl import load_workbook

wb = load_workbook('inventory.xlsx')
sheet = wb.active

# Add a total column
sheet['D1'] = 'Total Value'
for row in range(2, sheet.max_row + 1):
    sheet[f'D{row}'] = f'=B{row}*C{row}'

# Add a sum formula for the total value
total_row = sheet.max_row + 1
sheet[f'C{total_row}'] = 'Total:'
sheet[f'D{total_row}'] = f'=SUM(D2:D{sheet.max_row})'

wb.save('inventory_with_formulas.xlsx')
print("Inventory with formulas saved as 'inventory_with_formulas.xlsx'")
```

Slide 11: Real-Life Example: Automated Reporting

Let's create a simple automated reporting system that generates a summary of daily tasks completed by team members.

```python
from openpyxl import Workbook
from datetime import date, timedelta

def generate_daily_report(team_data):
    wb = Workbook()
    sheet = wb.active
    sheet.title = "Daily Task Report"
    
    # Add headers
    headers = ["Date", "Team Member", "Tasks Completed"]
    for col, header in enumerate(headers, start=1):
        sheet.cell(row=1, column=col, value=header)
    
    # Add data
    row = 2
    for member, tasks in team_data.items():
        for task_date, task_count in tasks.items():
            sheet.cell(row=row, column=1, value=task_date)
            sheet.cell(row=row, column=2, value=member)
            sheet.cell(row=row, column=3, value=task_count)
            row += 1
    
    # Save the report
    report_date = date.today().strftime("%Y-%m-%d")
    filename = f"daily_report_{report_date}.xlsx"
    wb.save(filename)
    print(f"Daily report generated: {filename}")

# Sample data
team_data = {
    "Alice": {date.today(): 5, date.today() - timedelta(days=1): 4},
    "Bob": {date.today(): 6, date.today() - timedelta(days=1): 7},
    "Charlie": {date.today(): 4, date.today() - timedelta(days=1): 5}
}

generate_daily_report(team_data)
```

Slide 12: Real-Life Example: Data Analysis

Let's use OpenPyXL to perform a simple data analysis task on a dataset of student grades.

```python
from openpyxl import Workbook
import random

# Generate sample data
def generate_student_data(num_students):
    subjects = ['Math', 'Science', 'English', 'History']
    data = [['Student ID'] + subjects]
    for i in range(1, num_students + 1):
        student_data = [f'S{i:03}'] + [random.randint(60, 100) for _ in subjects]
        data.append(student_data)
    return data

# Analyze and report
def analyze_grades(data):
    wb = Workbook()
    sheet = wb.active
    sheet.title = "Grade Analysis"
    
    # Write data
    for row in data:
        sheet.append(row)
    
    # Calculate averages
    num_subjects = len(data[0]) - 1
    sheet.cell(row=len(data)+1, column=1, value="Average")
    for col in range(2, num_subjects + 2):
        column_letter = sheet.cell(row=1, column=col).column_letter
        sheet[f"{column_letter}{len(data)+1}"] = f"=AVERAGE({column_letter}2:{column_letter}{len(data)})"
    
    # Find highest score for each subject
    sheet.cell(row=len(data)+2, column=1, value="Highest Score")
    for col in range(2, num_subjects + 2):
        column_letter = sheet.cell(row=1, column=col).column_letter
        sheet[f"{column_letter}{len(data)+2}"] = f"=MAX({column_letter}2:{column_letter}{len(data)})"
    
    wb.save("student_grade_analysis.xlsx")
    print("Grade analysis completed and saved as 'student_grade_analysis.xlsx'")

# Run the analysis
student_data = generate_student_data(50)
analyze_grades(student_data)
```

Slide 13: Best Practices and Tips

When working with OpenPyXL, consider these best practices:

1. Always close workbooks after use to free up system resources.
2. Use cell styles and named styles for consistent formatting across your workbook.
3. When dealing with large datasets, consider using openpyxl's read\_only and write\_only modes for better performance.
4. Regularly save your work to prevent data loss in case of unexpected errors.
5. Use error handling to manage potential issues when reading from or writing to Excel files.

```python
from openpyxl import load_workbook, Workbook
from openpyxl.styles import NamedStyle, Font, Border, Side

# Example of using named styles
def create_workbook_with_styles():
    wb = Workbook()
    sheet = wb.active
    
    # Create a named style
    highlight = NamedStyle(name="highlight")
    highlight.font = Font(bold=True, size=12)
    highlight.border = Border(left=Side(style='thin'), 
                              right=Side(style='thin'), 
                              top=Side(style='thin'), 
                              bottom=Side(style='thin'))
    
    # Add the named style to the workbook
    wb.add_named_style(highlight)
    
    # Use the named style
    sheet['A1'] = "Styled Cell"
    sheet['A1'].style = highlight
    
    wb.save('styled_workbook.xlsx')
    print("Workbook with named styles saved as 'styled_workbook.xlsx'")

create_workbook_with_styles()
```

Slide 14: Additional Resources

For further exploration of OpenPyXL and its capabilities, consider these resources:

1. Official OpenPyXL Documentation: [https://openpyxl.readthedocs.io/](https://openpyxl.readthedocs.io/)
2. Python Excel Tutorial on Real Python: [https://realpython.com/openpyxl-excel-spreadsheets-python/](https://realpython.com/openpyxl-excel-spreadsheets-python/)
3. OpenPyXL GitHub Repository: [https://github.com/theorchard/openpyxl](https://github.com/theorchard/openpyxl)
4. "Automate the Boring Stuff with Python" by Al Sweigart (Chapter on Excel manipulation)
5. Stack Overflow's OpenPyXL tag for community-driven Q&A: [https://stackoverflow.com/questions/tagged/openpyxl](https://stackoverflow.com/questions/tagged/openpyxl)

These resources provide in-depth information, tutorials, and community support to help you master OpenPyXL and Excel integration with Python.

