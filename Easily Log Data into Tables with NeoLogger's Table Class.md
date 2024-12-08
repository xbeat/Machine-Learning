## Easily Log Data into Tables with NeoLogger's Table Class
Slide 1: Introduction to NeoLogger's Table Class

The Table class in NeoLogger provides a powerful and flexible way to organize and display tabular data with customizable formatting options. It offers an intuitive API for creating structured tables with support for dynamic column alignment, borders, and styling.

```python
from neologger import Table

# Create a basic table
table = Table()
table.add_column("ID", justify="center")
table.add_column("Name", justify="left") 
table.add_column("Score", justify="right")

# Add rows of data
table.add_row("1", "Alice", "95.5")
table.add_row("2", "Bob", "87.3")

print(table.render())
```

Slide 2: Customizing Table Appearance

NeoLogger's Table class supports extensive customization through style parameters, allowing developers to control borders, colors, alignment and other visual aspects to create professional-looking console output.

```python
from neologger import Table, Style

# Create styled table
table = Table(
    show_header=True,
    header_style="bold magenta",
    border_style="blue",
    row_styles=["dim", ""]  # Alternating styles
)

table.add_column("Metric", style="cyan", justify="right")
table.add_column("Value", style="green")

table.add_row("Accuracy", "0.945")
table.add_row("Precision", "0.892")
table.add_row("Recall", "0.913")

print(table.render())
```

Slide 3: Dynamic Data Loading

The Table class efficiently handles dynamic data loading from various sources, making it ideal for real-time logging and monitoring applications. This implementation demonstrates loading and formatting data from a CSV file.

```python
import csv
from neologger import Table
from datetime import datetime

# Create table for logging
log_table = Table()
log_table.add_column("Timestamp")
log_table.add_column("Event")
log_table.add_column("Status")

# Load and display log data
with open('system_logs.csv', 'r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        timestamp = datetime.fromisoformat(row['timestamp'])
        log_table.add_row(
            timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            row['event'],
            row['status']
        )

print(log_table.render())
```

Slide 4: Real-time Performance Monitoring

Modern applications require real-time performance monitoring. This implementation shows how to create a live-updating table that displays system metrics with automatic refresh capabilities.

```python
import psutil
import time
from neologger import Table, Live

def get_system_metrics():
    table = Table()
    table.add_column("Metric")
    table.add_column("Value")
    
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    
    table.add_row("CPU Usage", f"{cpu_percent}%")
    table.add_row("Memory Used", f"{memory.percent}%")
    table.add_row("Disk Usage", 
                  f"{psutil.disk_usage('/').percent}%")
    
    return table

with Live(get_system_metrics(), refresh_per_second=2) as live:
    while True:
        live.update(get_system_metrics())
        time.sleep(0.5)
```

Slide 5: Nested Tables Implementation

NeoLogger supports nested table structures, enabling complex data representation for hierarchical information. This advanced feature allows developers to create sophisticated layouts for displaying related data sets within a single table.

```python
from neologger import Table, Box

def create_nested_table():
    # Create outer table
    main_table = Table(box=Box.DOUBLE)
    main_table.add_column("Department")
    main_table.add_column("Details")

    # Create nested tables for each department
    for dept in ["Engineering", "Marketing"]:
        nested_table = Table(box=Box.SIMPLE)
        nested_table.add_column("Employee")
        nested_table.add_column("Projects")
        
        # Add data to nested table
        if dept == "Engineering":
            nested_table.add_row("John", "Backend API")
            nested_table.add_row("Alice", "Frontend UI")
        else:
            nested_table.add_row("Bob", "Social Media")
            nested_table.add_row("Carol", "Content")
            
        main_table.add_row(dept, nested_table)
    
    return main_table

print(create_nested_table().render())
```

Slide 6: Custom Formatting and Styling

The Table class provides extensive formatting capabilities allowing precise control over data presentation. This implementation demonstrates advanced styling techniques including conditional formatting and custom renderers.

```python
from neologger import Table, Style, Colors

class MetricsTable:
    def __init__(self):
        self.table = Table(
            title="Performance Metrics",
            title_style="bold cyan",
            caption="Updated hourly",
            caption_style="italic"
        )
        
    def format_value(self, value, threshold):
        if float(value) < threshold:
            return f"[red]{value}[/red]"
        return f"[green]{value}[/green]"
    
    def create_report(self, metrics_data):
        self.table.add_column("Metric", style="bold")
        self.table.add_column("Value", justify="right")
        self.table.add_column("Threshold", justify="right")
        
        for metric in metrics_data:
            self.table.add_row(
                metric['name'],
                self.format_value(metric['value'], 
                                metric['threshold']),
                str(metric['threshold'])
            )
        
        return self.table

# Usage example
metrics = [
    {'name': 'Response Time', 'value': '1.2', 'threshold': 2.0},
    {'name': 'Error Rate', 'value': '2.5', 'threshold': 1.0}
]
report = MetricsTable().create_report(metrics)
print(report.render())
```

Slide 7: Pagination and Data Streaming

When dealing with large datasets, proper pagination and streaming capabilities become crucial. This implementation shows how to handle large data streams while maintaining memory efficiency.

```python
from neologger import Table, Console
import itertools

class PaginatedTable:
    def __init__(self, page_size=10):
        self.page_size = page_size
        self.table = Table(show_lines=True)
        self.console = Console()
        
    def stream_data(self, data_generator):
        # Configure table columns
        self.table.add_column("Index")
        self.table.add_column("Data")
        
        # Process data in chunks
        for chunk_idx, chunk in enumerate(
            itertools.islice(data_generator, 0, None, 
                           self.page_size)):
            self.table.rows.clear()  # Reset for new page
            
            for idx, item in enumerate(chunk):
                absolute_idx = chunk_idx * self.page_size + idx
                self.table.add_row(str(absolute_idx), str(item))
            
            # Render current page
            self.console.print(self.table)
            self.console.print(f"Page {chunk_idx + 1}")
            
            # Wait for user input to continue
            if input("Next page? (y/n): ").lower() != 'y':
                break

# Example usage with a data generator
def generate_data():
    for i in range(100):
        yield f"Data point {i}"

paginated_table = PaginatedTable(page_size=5)
paginated_table.stream_data(generate_data())
```

Slide 8: Event Logging System

Implementing a comprehensive event logging system using NeoLogger's Table class for structured output and real-time monitoring of system events and errors.

```python
from neologger import Table, Console
from datetime import datetime
import threading
import queue

class EventLogger:
    def __init__(self):
        self.event_queue = queue.Queue()
        self.table = Table(
            title="System Event Log",
            expand=True
        )
        self.setup_table()
        
    def setup_table(self):
        self.table.add_column("Timestamp", 
                             style="cyan")
        self.table.add_column("Level", 
                             style="magenta")
        self.table.add_column("Message", 
                             style="white")
        self.table.add_column("Source", 
                             style="green")
        
    def log_event(self, level, message, source):
        timestamp = datetime.now().strftime(
            '%Y-%m-%d %H:%M:%S.%f')[:-3]
        self.event_queue.put({
            'timestamp': timestamp,
            'level': level,
            'message': message,
            'source': source
        })
        
    def process_events(self):
        while True:
            event = self.event_queue.get()
            self.table.add_row(
                event['timestamp'],
                event['level'],
                event['message'],
                event['source']
            )
            Console().print(self.table)

# Example usage
logger = EventLogger()
process_thread = threading.Thread(
    target=logger.process_events, 
    daemon=True
)
process_thread.start()

# Simulate events
logger.log_event("INFO", "Application started", "Main")
logger.log_event("WARNING", "High memory usage", "Monitor")
logger.log_event("ERROR", "Database connection failed", "DB")
```

Slide 9: Performance Monitoring Dashboard

The Table class can be integrated into a comprehensive monitoring dashboard that tracks system metrics, application performance, and resource utilization in real-time with customizable thresholds and alerts.

```python
from neologger import Table, Live, Console
import psutil
import time
from datetime import datetime

class MonitoringDashboard:
    def __init__(self):
        self.console = Console()
        self.metrics_table = self.create_metrics_table()
        self.alerts_table = self.create_alerts_table()
        self.thresholds = {
            'cpu': 80,
            'memory': 85,
            'disk': 90
        }
        
    def create_metrics_table(self):
        table = Table(title="System Metrics")
        table.add_column("Metric")
        table.add_column("Current")
        table.add_column("Peak")
        table.add_column("Status")
        return table
        
    def create_alerts_table(self):
        table = Table(title="Recent Alerts")
        table.add_column("Time")
        table.add_column("Alert")
        table.add_column("Value")
        return table
        
    def get_status(self, current, threshold):
        return "🔴" if current > threshold else "🟢"
        
    def update_dashboard(self):
        self.metrics_table.rows = []
        
        # Collect metrics
        cpu = psutil.cpu_percent()
        memory = psutil.virtual_memory().percent
        disk = psutil.disk_usage('/').percent
        
        # Update metrics table
        metrics = [
            ("CPU Usage", cpu, self.thresholds['cpu']),
            ("Memory", memory, self.thresholds['memory']),
            ("Disk", disk, self.thresholds['disk'])
        ]
        
        for name, value, threshold in metrics:
            self.metrics_table.add_row(
                name,
                f"{value}%",
                f"{max(value, threshold)}%",
                self.get_status(value, threshold)
            )
            
            if value > threshold:
                self.alerts_table.add_row(
                    datetime.now().strftime('%H:%M:%S'),
                    f"High {name}",
                    f"{value}%"
                )
        
        return self.metrics_table

# Usage
dashboard = MonitoringDashboard()
with Live(dashboard.update_dashboard(), 
         refresh_per_second=2) as live:
    while True:
        live.update(dashboard.update_dashboard())
        time.sleep(0.5)
```

Slide 10: Advanced Data Analysis Integration

This implementation demonstrates how to integrate the Table class with data analysis tools to create detailed statistical reports with formatted output and dynamic calculations.

```python
import numpy as np
import pandas as pd
from neologger import Table
from scipy import stats

class AnalyticsReport:
    def __init__(self, data):
        self.data = pd.DataFrame(data)
        self.stats_table = Table(
            title="Statistical Analysis Report",
            show_header=True,
            header_style="bold magenta"
        )
        
    def calculate_statistics(self):
        stats_dict = {}
        for column in self.data.columns:
            column_data = self.data[column].dropna()
            stats_dict[column] = {
                'mean': np.mean(column_data),
                'median': np.median(column_data),
                'std': np.std(column_data),
                'skew': stats.skew(column_data),
                'kurtosis': stats.kurtosis(column_data)
            }
        return stats_dict
        
    def generate_report(self):
        self.stats_table.add_column("Metric")
        for column in self.data.columns:
            self.stats_table.add_column(column)
            
        statistics = self.calculate_statistics()
        metrics = ['mean', 'median', 'std', 'skew', 'kurtosis']
        
        for metric in metrics:
            row_data = [metric.capitalize()]
            for column in self.data.columns:
                value = statistics[column][metric]
                formatted_value = f"{value:.3f}"
                row_data.append(formatted_value)
            self.stats_table.add_row(*row_data)
            
        return self.stats_table

# Example usage
data = {
    'Value1': np.random.normal(100, 15, 1000),
    'Value2': np.random.exponential(50, 1000),
    'Value3': np.random.uniform(0, 100, 1000)
}

report = AnalyticsReport(data)
print(report.generate_report().render())
```

Slide 11: Custom Table Exporters

The Table class can be extended with custom exporters to support various output formats. This implementation demonstrates how to create exporters for CSV, JSON, and HTML formats while maintaining the table's styling.

```python
from neologger import Table
import json
import csv
import html
from typing import Dict, List

class TableExporter:
    def __init__(self, table: Table):
        self.table = table
        self.data = self._extract_data()
    
    def _extract_data(self) -> List[Dict]:
        data = []
        headers = [col.header for col in self.table.columns]
        
        for row in self.table.rows:
            row_data = {}
            for header, cell in zip(headers, row.cells):
                row_data[header] = cell.value
            data.append(row_data)
        return data
    
    def to_csv(self, filename: str) -> None:
        with open(filename, 'w', newline='') as f:
            writer = csv.DictWriter(f, 
                fieldnames=self.data[0].keys())
            writer.writeheader()
            writer.writerows(self.data)
    
    def to_json(self, filename: str) -> None:
        with open(filename, 'w') as f:
            json.dump(self.data, f, indent=2)
    
    def to_html(self, filename: str) -> None:
        html_content = ['<table border="1">']
        
        # Add headers
        headers = self.data[0].keys()
        header_row = ''.join(
            f'<th>{html.escape(str(h))}</th>' 
            for h in headers
        )
        html_content.append(f'<tr>{header_row}</tr>')
        
        # Add data rows
        for row in self.data:
            cells = ''.join(
                f'<td>{html.escape(str(v))}</td>' 
                for v in row.values()
            )
            html_content.append(f'<tr>{cells}</tr>')
        
        html_content.append('</table>')
        
        with open(filename, 'w') as f:
            f.write('\n'.join(html_content))

# Example usage
table = Table()
table.add_column("Name")
table.add_column("Age")
table.add_column("City")

table.add_row("Alice", "25", "New York")
table.add_row("Bob", "30", "London")
table.add_row("Carol", "28", "Tokyo")

exporter = TableExporter(table)
exporter.to_csv("output.csv")
exporter.to_json("output.json")
exporter.to_html("output.html")
```

Slide 12: Integration with Machine Learning Pipelines

NeoLogger's Table class can be effectively integrated into machine learning workflows for displaying model metrics, hyperparameter tuning results, and training progress in a structured format.

```python
from neologger import Table, Console
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import numpy as np

class MLExperimentLogger:
    def __init__(self):
        self.console = Console()
        self.results_table = self._create_results_table()
        self.metrics_table = self._create_metrics_table()
        
    def _create_results_table(self):
        table = Table(title="Grid Search Results")
        table.add_column("Parameters")
        table.add_column("Mean CV Score")
        table.add_column("Std CV Score")
        return table
        
    def _create_metrics_table(self):
        table = Table(title="Model Metrics")
        table.add_column("Metric")
        table.add_column("Value")
        return table
    
    def log_grid_search(self, grid_search: GridSearchCV):
        cv_results = grid_search.cv_results_
        for params, mean_score, std_score in zip(
            cv_results['params'],
            cv_results['mean_test_score'],
            cv_results['std_test_score']
        ):
            params_str = ', '.join(
                f'{k}={v}' for k, v in params.items()
            )
            self.results_table.add_row(
                params_str,
                f"{mean_score:.4f}",
                f"{std_score:.4f}"
            )
    
    def log_metrics(self, y_true, y_pred):
        report = classification_report(
            y_true, y_pred, output_dict=True
        )
        
        for metric, values in report.items():
            if isinstance(values, dict):
                for k, v in values.items():
                    self.metrics_table.add_row(
                        f"{metric}_{k}",
                        f"{v:.4f}"
                    )
    
    def display_results(self):
        self.console.print(self.results_table)
        self.console.print(self.metrics_table)

# Example usage
X = np.random.rand(100, 4)
y = np.random.randint(0, 2, 100)

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20]
}

grid_search = GridSearchCV(
    RandomForestClassifier(),
    param_grid,
    cv=5
)
grid_search.fit(X, y)

logger = MLExperimentLogger()
logger.log_grid_search(grid_search)
logger.log_metrics(y, grid_search.predict(X))
logger.display_results()
```

Slide 13: Real-time Log Aggregation System

NeoLogger's Table class can be used to build a sophisticated log aggregation system that collects, filters, and displays logs from multiple sources in real-time with support for severity levels and pattern matching.

```python
from neologger import Table, Live, Console
from datetime import datetime
import re
import queue
import threading
from typing import Dict, List, Pattern

class LogAggregator:
    def __init__(self):
        self.log_queue = queue.Queue()
        self.filters: Dict[str, Pattern] = {}
        self.table = self._create_log_table()
        self.severity_colors = {
            'DEBUG': 'dim blue',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'bold red'
        }
    
    def _create_log_table(self) -> Table:
        table = Table(title="Aggregated Logs")
        table.add_column("Timestamp")
        table.add_column("Source")
        table.add_column("Severity")
        table.add_column("Message")
        table.add_column("Tags")
        return table
    
    def add_filter(self, name: str, pattern: str):
        self.filters[name] = re.compile(pattern)
    
    def process_log(self, log_entry: Dict) -> bool:
        for pattern in self.filters.values():
            if pattern.search(str(log_entry)):
                return True
        return len(self.filters) == 0
    
    def add_log(self, source: str, severity: str, 
                message: str, tags: List[str] = None):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        log_entry = {
            'timestamp': timestamp,
            'source': source,
            'severity': severity,
            'message': message,
            'tags': tags or []
        }
        
        if self.process_log(log_entry):
            self.log_queue.put(log_entry)
    
    def update_display(self) -> Table:
        while not self.log_queue.empty():
            log = self.log_queue.get()
            self.table.add_row(
                log['timestamp'],
                log['source'],
                f"[{self.severity_colors[log['severity']]}]"
                f"{log['severity']}[/]",
                log['message'],
                ', '.join(log['tags'])
            )
        return self.table

# Example usage
aggregator = LogAggregator()

# Add filters for specific patterns
aggregator.add_filter("errors", r"error|exception|fail",)
aggregator.add_filter("database", r"database|sql|query")

# Simulate log entries
def generate_logs():
    while True:
        aggregator.add_log(
            "API Server",
            "ERROR",
            "Database connection failed",
            ["database", "connectivity"]
        )
        aggregator.add_log(
            "Web Server",
            "INFO",
            "Request processed successfully",
            ["http", "performance"]
        )
        time.sleep(2)

# Start log generation in background
log_thread = threading.Thread(
    target=generate_logs, 
    daemon=True
)
log_thread.start()

# Display live updates
with Live(aggregator.update_display(), 
         refresh_per_second=2) as live:
    while True:
        live.update(aggregator.update_display())
        time.sleep(0.5)
```

Slide 14: Additional Resources

*   General guides and research papers for logging systems:
    *   [https://arxiv.org/abs/2207.09399](https://arxiv.org/abs/2207.09399) - "Modern Logging Techniques in Distributed Systems"
    *   [https://arxiv.org/abs/2103.07133](https://arxiv.org/abs/2103.07133) - "Automated Log Analysis: State of the Art and Future Trends"
    *   [https://arxiv.org/abs/1908.11239](https://arxiv.org/abs/1908.11239) - "Log-based Anomaly Detection: Approaches and Frameworks"
*   Recommended search terms for further exploration:
    *   "Real-time log analysis systems"
    *   "Distributed logging architectures"
    *   "Log aggregation best practices"
    *   "Event correlation in logging systems"
*   Online resources:
    *   [https://logging.apache.org/log4j/2.x/](https://logging.apache.org/log4j/2.x/)
    *   [https://docs.python.org/3/library/logging.html](https://docs.python.org/3/library/logging.html)
    *   [https://www.elastic.co/guide/en/elasticsearch/reference/current/logging.html](https://www.elastic.co/guide/en/elasticsearch/reference/current/logging.html)

