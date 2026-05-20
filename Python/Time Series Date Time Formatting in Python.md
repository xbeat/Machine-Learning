## Time Series Date Time Formatting in Python
Slide 1: Introduction to Time Series Date Time Formatting

Time series data often involves working with dates and times. Proper formatting of these elements is crucial for accurate analysis and visualization. In Python, the datetime module provides powerful tools for handling date and time data. This slideshow will explore various aspects of time series date time formatting, offering practical examples and techniques.

```python
from datetime import datetime, timedelta

# Current date and time
now = datetime.now()
print(f"Current date and time: {now}")

# Formatted date and time
formatted = now.strftime("%Y-%m-%d %H:%M:%S")
print(f"Formatted date and time: {formatted}")
```

Slide 2: Basic Date Time Formatting

The strftime() method is a powerful tool for formatting date and time objects. It uses format codes to specify how the date and time should be represented as a string. Common format codes include %Y for year, %m for month, %d for day, %H for hour, %M for minute, and %S for second.

```python
from datetime import datetime

date_time = datetime(2024, 3, 15, 14, 30, 0)

# Different format examples
formats = [
    "%Y-%m-%d",
    "%d/%m/%Y",
    "%B %d, %Y",
    "%H:%M:%S",
    "%I:%M %p",
    "%Y-%m-%d %H:%M:%S"
]

for fmt in formats:
    print(f"Format {fmt}: {date_time.strftime(fmt)}")
```

Slide 3: Results for: Basic Date Time Formatting

```
Format %Y-%m-%d: 2024-03-15
Format %d/%m/%Y: 15/03/2024
Format %B %d, %Y: March 15, 2024
Format %H:%M:%S: 14:30:00
Format %I:%M %p: 02:30 PM
Format %Y-%m-%d %H:%M:%S: 2024-03-15 14:30:00
```

Slide 4: Parsing Date Time Strings

When working with time series data, you often need to convert date time strings into datetime objects. The strptime() function is used for this purpose. It takes a string and a format specification as input and returns a datetime object.

```python
from datetime import datetime

date_strings = [
    "2024-03-15 14:30:00",
    "15/03/2024",
    "March 15, 2024 2:30 PM"
]

formats = [
    "%Y-%m-%d %H:%M:%S",
    "%d/%m/%Y",
    "%B %d, %Y %I:%M %p"
]

for date_str, fmt in zip(date_strings, formats):
    parsed_date = datetime.strptime(date_str, fmt)
    print(f"Original: {date_str}")
    print(f"Parsed: {parsed_date}")
    print()
```

Slide 5: Working with Time Zones

Time zones are crucial when dealing with global time series data. The pytz library is commonly used for handling time zones in Python. It allows you to create timezone-aware datetime objects and perform conversions between different time zones.

```python
from datetime import datetime
import pytz

# Create a timezone-aware datetime
utc_time = datetime.now(pytz.UTC)
print(f"UTC time: {utc_time}")

# Convert to a different time zone
ny_tz = pytz.timezone("America/New_York")
ny_time = utc_time.astimezone(ny_tz)
print(f"New York time: {ny_time}")

# List available time zones
print("\nSome available time zones:")
for tz in pytz.common_timezones[:5]:
    print(tz)
```

Slide 6: Time Series Operations

When working with time series data, you often need to perform operations like adding or subtracting time intervals. Python's timedelta class is useful for such operations.

```python
from datetime import datetime, timedelta

start_date = datetime(2024, 1, 1)
print(f"Start date: {start_date}")

# Add 30 days
end_date = start_date + timedelta(days=30)
print(f"End date (30 days later): {end_date}")

# Subtract 5 hours
five_hours_earlier = start_date - timedelta(hours=5)
print(f"5 hours earlier: {five_hours_earlier}")

# Calculate duration
duration = end_date - start_date
print(f"Duration: {duration}")
```

Slide 7: Date Ranges and Periods

When analyzing time series data, you often need to create date ranges or work with regular time periods. Python's pandas library provides powerful tools for this purpose.

```python
import pandas as pd

# Create a date range
date_range = pd.date_range(start="2024-01-01", end="2024-01-10", freq="D")
print("Date range:")
print(date_range)

# Create a period range
period_range = pd.period_range(start="2024-01", end="2024-12", freq="M")
print("\nPeriod range:")
print(period_range)

# Resample time series data
ts = pd.Series(range(30), index=pd.date_range(start="2024-01-01", periods=30, freq="D"))
resampled = ts.resample("W").sum()
print("\nResampled time series (weekly sum):")
print(resampled)
```

Slide 8: Real-Life Example: Weather Data Analysis

Let's analyze a simple weather dataset with temperature readings. We'll parse the date-time strings, create a time series, and perform basic analysis.

```python
import pandas as pd
import matplotlib.pyplot as plt

# Sample weather data
data = {
    "date": ["2024-03-01 12:00", "2024-03-02 12:00", "2024-03-03 12:00", "2024-03-04 12:00", "2024-03-05 12:00"],
    "temperature": [20.5, 22.1, 21.8, 19.7, 23.2]
}

# Create DataFrame and parse dates
df = pd.DataFrame(data)
df["date"] = pd.to_datetime(df["date"])
df.set_index("date", inplace=True)

# Calculate daily average temperature
daily_avg = df.resample("D").mean()

# Plot the data
plt.figure(figsize=(10, 6))
plt.plot(df.index, df["temperature"], marker="o", label="Hourly")
plt.plot(daily_avg.index, daily_avg["temperature"], marker="s", label="Daily Avg")
plt.title("Temperature Analysis")
plt.xlabel("Date")
plt.ylabel("Temperature (°C)")
plt.legend()
plt.grid(True)
plt.show()
```

Slide 9: Real-Life Example: Website Traffic Analysis

In this example, we'll analyze website traffic data, parsing timestamps and aggregating data by hour and day of the week.

```python
import pandas as pd
import matplotlib.pyplot as plt

# Sample website traffic data
data = {
    "timestamp": [
        "2024-03-01 08:30:00", "2024-03-01 12:15:00", "2024-03-01 18:45:00",
        "2024-03-02 09:00:00", "2024-03-02 14:30:00", "2024-03-02 20:00:00",
        "2024-03-03 10:45:00", "2024-03-03 16:20:00", "2024-03-03 22:10:00"
    ],
    "visitors": [50, 120, 80, 65, 150, 95, 75, 130, 70]
}

# Create DataFrame and parse timestamps
df = pd.DataFrame(data)
df["timestamp"] = pd.to_datetime(df["timestamp"])
df.set_index("timestamp", inplace=True)

# Aggregate by hour
hourly_visitors = df.groupby(df.index.hour)["visitors"].mean()

# Aggregate by day of week
daily_visitors = df.groupby(df.index.dayofweek)["visitors"].mean()

# Plot hourly visitors
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
hourly_visitors.plot(kind="bar")
plt.title("Average Hourly Visitors")
plt.xlabel("Hour of Day")
plt.ylabel("Average Visitors")

# Plot daily visitors
plt.subplot(1, 2, 2)
daily_visitors.plot(kind="bar")
plt.title("Average Daily Visitors")
plt.xlabel("Day of Week (0=Monday, 6=Sunday)")
plt.ylabel("Average Visitors")

plt.tight_layout()
plt.show()
```

Slide 10: Working with Fiscal Years

In some industries, fiscal years don't align with calendar years. Let's explore how to handle fiscal year calculations in Python.

```python
from datetime import datetime

def get_fiscal_year(date, fiscal_start_month=7):
    if date.month < fiscal_start_month:
        return date.year
    else:
        return date.year + 1

# Example dates
dates = [
    datetime(2024, 1, 1),
    datetime(2024, 6, 30),
    datetime(2024, 7, 1),
    datetime(2024, 12, 31),
    datetime(2025, 1, 1)
]

# Calculate fiscal years (assuming fiscal year starts in July)
for date in dates:
    fiscal_year = get_fiscal_year(date)
    print(f"Date: {date.strftime('%Y-%m-%d')}, Fiscal Year: {fiscal_year}")
```

Slide 11: Custom Date Formatting

Sometimes, you may need to create custom date formats that aren't directly supported by strftime(). Let's create a function to format dates using ordinal suffixes for days.

```python
from datetime import datetime

def format_date_with_ordinal(date):
    day = date.day
    if 4 <= day <= 20 or 24 <= day <= 30:
        suffix = "th"
    else:
        suffix = ["st", "nd", "rd"][day % 10 - 1]
    
    return date.strftime(f"%B {day}{suffix}, %Y")

# Example dates
dates = [
    datetime(2024, 3, 1),
    datetime(2024, 3, 2),
    datetime(2024, 3, 3),
    datetime(2024, 3, 4),
    datetime(2024, 3, 21),
    datetime(2024, 3, 22),
    datetime(2024, 3, 23)
]

for date in dates:
    formatted = format_date_with_ordinal(date)
    print(f"Formatted date: {formatted}")
```

Slide 12: Working with ISO Week Dates

ISO week dates are widely used in some industries. Let's explore how to work with ISO week dates in Python.

```python
from datetime import datetime, timedelta

def iso_year_start(iso_year):
    """The gregorian calendar date of the first day of the given ISO year"""
    fourth_jan = datetime(iso_year, 1, 4)
    delta = timedelta(fourth_jan.isoweekday()-1)
    return fourth_jan - delta 

def iso_to_gregorian(iso_year, iso_week, iso_day):
    """Gregorian calendar date for the given ISO year, week and day"""
    year_start = iso_year_start(iso_year)
    return year_start + timedelta(days=iso_day-1, weeks=iso_week-1)

# Example: Convert ISO date to Gregorian
iso_date = (2024, 10, 1)  # Year 2024, Week 10, Monday
greg_date = iso_to_gregorian(*iso_date)
print(f"ISO date {iso_date} is equivalent to Gregorian date: {greg_date.strftime('%Y-%m-%d')}")

# Example: Get ISO week number for a Gregorian date
date = datetime(2024, 3, 15)
iso_calendar = date.isocalendar()
print(f"Gregorian date {date.strftime('%Y-%m-%d')} is in ISO week: {iso_calendar.week}")
```

Slide 13: Handling Daylight Saving Time (DST)

Daylight Saving Time (DST) can complicate time series analysis. Let's explore how to handle DST transitions in Python.

```python
from datetime import datetime
import pytz

def print_dst_info(timezone_name, year):
    tz = pytz.timezone(timezone_name)
    
    # Find DST start
    dst_start = None
    for day in range(1, 366):
        date = datetime(year, 1, 1) + timedelta(days=day-1)
        if tz.localize(date).dst() != timedelta(0):
            dst_start = date
            break
    
    # Find DST end
    dst_end = None
    for day in range(366, 1, -1):
        date = datetime(year, 1, 1) + timedelta(days=day-1)
        if tz.localize(date).dst() != timedelta(0):
            dst_end = date + timedelta(days=1)
            break
    
    print(f"DST information for {timezone_name} in {year}:")
    print(f"DST starts: {dst_start.strftime('%Y-%m-%d')}")
    print(f"DST ends: {dst_end.strftime('%Y-%m-%d')}")

# Example: Print DST information for New York in 2024
print_dst_info("America/New_York", 2024)
```

Slide 14: Additional Resources

For further exploration of time series date time formatting in Python, consider the following resources:

1.  Python datetime documentation: [https://docs.python.org/3/library/datetime.html](https://docs.python.org/3/library/datetime.html)
2.  Pandas Time Series documentation: [https://pandas.pydata.org/pandas-docs/stable/user\_guide/timeseries.html](https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html)
3.  pytz documentation: [https://pythonhosted.org/pytz/](https://pythonhosted.org/pytz/)
4.  "Practical Time Series Analysis" by Aileen Nielsen (O'Reilly Media)
5.  ArXiv paper: "Time Series Analysis: Methods and Applications for Flight Data" ([https://arxiv.org/abs/1903.00122](https://arxiv.org/abs/1903.00122))

These resources provide in-depth information on handling time series data, advanced date-time operations, and real-world applications in various domains.

