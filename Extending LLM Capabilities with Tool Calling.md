## Extending LLM Capabilities with Tool Calling
Slide 1: Tool Calling Fundamentals

Tool calling enables Large Language Models to interact with external functions and APIs, extending their capabilities beyond text generation. This architectural pattern allows LLMs to recognize when they need external assistance and orchestrate the execution of specialized tools to accomplish complex tasks.

```python
from typing import List, Dict, Callable
import inspect

class ToolManager:
    def __init__(self):
        self.tools: Dict[str, Callable] = {}
    
    def register_tool(self, func: Callable):
        """Register a new tool with its signature"""
        self.tools[func.__name__] = func
        return func
    
    def get_tool_signatures(self) -> Dict[str, str]:
        """Return all registered tool signatures"""
        return {
            name: str(inspect.signature(func))
            for name, func in self.tools.items()
        }

# Example tool registration
tool_manager = ToolManager()

@tool_manager.register_tool
def fetch_stock_price(symbol: str) -> float:
    """Fetch current stock price for given symbol"""
    return 150.25  # Simplified example
```

Slide 2: Tool Definition Interface

A robust tool calling system requires clear interface definitions that specify input parameters, expected outputs, and error handling. This framework provides type hints and documentation to ensure reliable tool integration with the LLM.

```python
from dataclasses import dataclass
from typing import Optional, Any

@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: Dict[str, type]
    returns: type
    error_handling: Optional[Dict[str, Any]] = None

def create_tool_spec(func: Callable) -> ToolSpec:
    sig = inspect.signature(func)
    return ToolSpec(
        name=func.__name__,
        description=func.__doc__ or "",
        parameters={
            name: param.annotation 
            for name, param in sig.parameters.items()
        },
        returns=sig.return_annotation
    )

# Example usage
stock_tool_spec = create_tool_spec(fetch_stock_price)
```

Slide 3: Tool Execution Pipeline

The execution pipeline manages the flow from LLM request to tool invocation and result processing. This system handles parameter validation, execution monitoring, and proper error propagation to maintain robust tool interactions.

```python
class ToolExecutor:
    def __init__(self, tool_manager: ToolManager):
        self.tool_manager = tool_manager
    
    async def execute_tool(self, 
                          tool_name: str, 
                          parameters: Dict[str, Any]) -> Any:
        if tool_name not in self.tool_manager.tools:
            raise ValueError(f"Unknown tool: {tool_name}")
        
        tool = self.tool_manager.tools[tool_name]
        try:
            result = await tool(**parameters)
            return {
                "status": "success",
                "result": result,
                "tool": tool_name
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "tool": tool_name
            }
```

Slide 4: LLM Integration Layer

The integration layer provides the interface between the LLM and registered tools. It handles parsing LLM requests, mapping them to appropriate tools, and formatting responses for LLM consumption.

```python
class LLMToolIntegration:
    def __init__(self, executor: ToolExecutor):
        self.executor = executor
        
    async def process_llm_request(self, 
                                request: str, 
                                context: Dict[str, Any]) -> str:
        # Parse LLM request to identify tool and parameters
        tool_call = self._parse_tool_request(request)
        
        if tool_call:
            result = await self.executor.execute_tool(
                tool_call["name"],
                tool_call["parameters"]
            )
            return self._format_tool_response(result)
        
        return "No tool call identified in request"
    
    def _parse_tool_request(self, request: str) -> Dict[str, Any]:
        # Simplified parsing logic
        return {
            "name": "fetch_stock_price",
            "parameters": {"symbol": "AAPL"}
        }
```

Slide 5: Real-world Example: Stock Analysis System

A practical implementation of tool calling for financial analysis, combining market data retrieval with LLM-powered insights. This system demonstrates how to chain multiple tools for comprehensive analysis.

```python
import yfinance as yf
from datetime import datetime, timedelta

class StockAnalyzer:
    def __init__(self, tool_manager: ToolManager):
        self.tool_manager = tool_manager
        self._register_tools()
    
    def _register_tools(self):
        @self.tool_manager.register_tool
        async def get_historical_data(symbol: str, 
                                    days: int = 30) -> Dict[str, float]:
            stock = yf.Ticker(symbol)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            data = stock.history(start=start_date, end=end_date)
            return {
                "current_price": data['Close'][-1],
                "avg_volume": data['Volume'].mean(),
                "price_change": (data['Close'][-1] - data['Close'][0]) / 
                               data['Close'][0] * 100
            }

# Example usage
analyzer = StockAnalyzer(tool_manager)
```

Slide 6: Parameter Validation and Type Checking

Complex tool interactions require robust parameter validation to prevent runtime errors. This system implements comprehensive type checking and validation before tool execution, ensuring reliable operation in production environments.

```python
from pydantic import BaseModel, validator
from typing import Union, Any

class ToolParameter(BaseModel):
    name: str
    value: Any
    type_hint: type
    
    @validator('value')
    def validate_type(cls, v, values):
        expected_type = values['type_hint']
        if not isinstance(v, expected_type):
            try:
                return expected_type(v)
            except:
                raise ValueError(
                    f"Cannot convert {v} to {expected_type.__name__}"
                )
        return v

class ParameterValidator:
    def validate_tool_params(self, 
                           tool: Callable, 
                           params: Dict[str, Any]) -> Dict[str, Any]:
        sig = inspect.signature(tool)
        validated_params = {}
        
        for name, param in sig.parameters.items():
            if name not in params:
                if param.default is param.empty:
                    raise ValueError(f"Missing required parameter: {name}")
                continue
                
            validator = ToolParameter(
                name=name,
                value=params[name],
                type_hint=param.annotation
            )
            validated_params[name] = validator.value
            
        return validated_params
```

Slide 7: Asynchronous Tool Execution

Implementing asynchronous tool execution enables handling multiple concurrent requests efficiently. This pattern is crucial for scalable applications that need to manage numerous tool calls simultaneously.

```python
import asyncio
from typing import List, Dict, Any

class AsyncToolExecutor:
    def __init__(self, tool_manager: ToolManager):
        self.tool_manager = tool_manager
        self.active_tasks: Dict[str, asyncio.Task] = {}
        
    async def execute_concurrent_tools(self, 
                                    tool_requests: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        tasks = []
        for req in tool_requests:
            task = asyncio.create_task(
                self._execute_single_tool(req['tool'], req['params'])
            )
            tasks.append(task)
            
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return [
            self._process_tool_result(result) 
            for result in results
        ]
        
    async def _execute_single_tool(self, 
                                 tool_name: str, 
                                 params: Dict[str, Any]) -> Any:
        tool = self.tool_manager.tools[tool_name]
        task_id = f"{tool_name}_{id(params)}"
        
        try:
            result = await tool(**params)
            return {"status": "success", "result": result}
        except Exception as e:
            return {"status": "error", "error": str(e)}
```

Slide 8: Tool Result Processing and Formatting

Processing and formatting tool execution results ensures consistent output structure and proper error handling. This system implements a standardized way to handle successful results and errors.

```python
class ResultProcessor:
    def __init__(self):
        self.formatters = {}
        
    def register_formatter(self, tool_name: str, formatter: Callable):
        self.formatters[tool_name] = formatter
        
    def process_result(self, 
                      tool_name: str, 
                      result: Any) -> Dict[str, Any]:
        formatter = self.formatters.get(
            tool_name, 
            self._default_formatter
        )
        
        processed_result = formatter(result)
        return {
            "tool": tool_name,
            "timestamp": datetime.now().isoformat(),
            "data": processed_result,
            "format_version": "1.0"
        }
    
    @staticmethod
    def _default_formatter(result: Any) -> Dict[str, Any]:
        if isinstance(result, (int, float, str, bool)):
            return {"value": result}
        elif isinstance(result, dict):
            return result
        else:
            return {"value": str(result)}
```

Slide 9: Tool Chain Implementation

Tool chaining allows for complex workflows where multiple tools are executed in sequence. This implementation manages dependencies and data flow between consecutive tool executions.

```python
class ToolChain:
    def __init__(self, tool_manager: ToolManager):
        self.tool_manager = tool_manager
        self.chains: Dict[str, List[Dict[str, Any]]] = {}
        
    def create_chain(self, 
                    chain_id: str, 
                    steps: List[Dict[str, Any]]):
        """
        Create a tool execution chain
        steps: List of dicts with tool_name and param_mapping
        """
        self.chains[chain_id] = steps
        
    async def execute_chain(self, 
                          chain_id: str, 
                          initial_params: Dict[str, Any]) -> List[Any]:
        if chain_id not in self.chains:
            raise ValueError(f"Unknown chain: {chain_id}")
            
        results = []
        current_params = initial_params.copy()
        
        for step in self.chains[chain_id]:
            tool_name = step['tool_name']
            param_mapping = step.get('param_mapping', {})
            
            # Map parameters from previous results
            mapped_params = self._map_parameters(
                current_params, 
                param_mapping
            )
            
            # Execute tool
            result = await self.tool_manager.tools[tool_name](**mapped_params)
            results.append(result)
            
            # Update parameters for next step
            current_params.update(result)
            
        return results
```

Slide 10: Error Handling and Recovery

Robust error handling is crucial for production tool calling systems. This implementation provides comprehensive error catching, logging, and recovery mechanisms to ensure system reliability even when tools fail.

```python
from enum import Enum
from dataclasses import dataclass
from typing import Optional, Callable

class ToolErrorType(Enum):
    VALIDATION_ERROR = "validation_error"
    EXECUTION_ERROR = "execution_error"
    TIMEOUT_ERROR = "timeout_error"
    PERMISSION_ERROR = "permission_error"

@dataclass
class ToolError:
    type: ToolErrorType
    message: str
    traceback: Optional[str] = None
    recovery_action: Optional[Callable] = None

class ToolErrorHandler:
    def __init__(self):
        self.error_handlers: Dict[ToolErrorType, Callable] = {}
        self.max_retries = 3
        
    async def handle_error(self, 
                          error: ToolError, 
                          tool_name: str, 
                          params: Dict[str, Any]) -> Dict[str, Any]:
        handler = self.error_handlers.get(error.type)
        
        if handler:
            try:
                return await handler(error, tool_name, params)
            except Exception as e:
                return {
                    "status": "error",
                    "message": f"Error handler failed: {str(e)}",
                    "original_error": error
                }
        
        return {
            "status": "unhandled_error",
            "error": error,
            "tool": tool_name
        }
    
    async def retry_with_backoff(self, 
                               tool: Callable, 
                               params: Dict[str, Any]) -> Any:
        for attempt in range(self.max_retries):
            try:
                return await tool(**params)
            except Exception as e:
                wait_time = (2 ** attempt) * 1.5
                await asyncio.sleep(wait_time)
                
        raise Exception(f"Max retries ({self.max_retries}) exceeded")
```

Slide 11: Context Management for Tool Execution

Managing execution context ensures tools have access to necessary environmental variables and configuration while maintaining isolation between different tool calls.

```python
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional

class ToolContext:
    def __init__(self):
        self.global_context: Dict[str, Any] = {}
        self.context_stack: List[Dict[str, Any]] = []
        
    @asynccontextmanager
    async def execution_context(self, 
                              tool_name: str, 
                              local_context: Dict[str, Any]) -> AsyncGenerator[Dict[str, Any], None]:
        context = {
            **self.global_context,
            **local_context,
            "tool_name": tool_name,
            "execution_id": str(uuid.uuid4()),
            "timestamp": datetime.now().isoformat()
        }
        
        self.context_stack.append(context)
        try:
            yield context
        finally:
            self.context_stack.pop()
            
class ContextAwareToolExecutor:
    def __init__(self, 
                 tool_manager: ToolManager, 
                 context_manager: ToolContext):
        self.tool_manager = tool_manager
        self.context_manager = context_manager
        
    async def execute_with_context(self,
                                 tool_name: str,
                                 params: Dict[str, Any],
                                 context: Optional[Dict[str, Any]] = None) -> Any:
        async with self.context_manager.execution_context(tool_name, context or {}) as exec_context:
            tool = self.tool_manager.tools[tool_name]
            return await tool(context=exec_context, **params)
```

Slide 12: Real-world Example: Data Processing Pipeline

A complete example demonstrating tool calling in a data processing pipeline that combines multiple tools for data extraction, transformation, and analysis.

```python
class DataPipeline:
    def __init__(self, tool_manager: ToolManager):
        self.tool_manager = tool_manager
        self._register_pipeline_tools()
        
    def _register_pipeline_tools(self):
        @self.tool_manager.register_tool
        async def extract_data(source: str, 
                             query: Dict[str, Any]) -> pd.DataFrame:
            # Simulated data extraction
            data = pd.DataFrame({
                'value': np.random.randn(100),
                'timestamp': pd.date_range(
                    start='2024-01-01', 
                    periods=100, 
                    freq='D'
                )
            })
            return data
            
        @self.tool_manager.register_tool
        async def transform_data(data: pd.DataFrame, 
                               operations: List[Dict[str, Any]]) -> pd.DataFrame:
            for op in operations:
                if op['type'] == 'rolling_mean':
                    data['rolling_mean'] = data['value'].rolling(
                        window=op['window']
                    ).mean()
                elif op['type'] == 'normalize':
                    data['normalized'] = (
                        data['value'] - data['value'].mean()
                    ) / data['value'].std()
            return data
            
        @self.tool_manager.register_tool
        async def analyze_data(data: pd.DataFrame) -> Dict[str, float]:
            return {
                'mean': float(data['value'].mean()),
                'std': float(data['value'].std()),
                'skew': float(data['value'].skew())
            }

# Example pipeline execution
async def run_pipeline():
    pipeline = DataPipeline(tool_manager)
    data = await pipeline.tool_manager.tools['extract_data'](
        source='sample',
        query={'limit': 100}
    )
    transformed = await pipeline.tool_manager.tools['transform_data'](
        data=data,
        operations=[
            {'type': 'rolling_mean', 'window': 5},
            {'type': 'normalize'}
        ]
    )
    results = await pipeline.tool_manager.tools['analyze_data'](
        data=transformed
    )
    return results
```

Slide 13: Additional Resources

*   "Language Models are Few-Shot Learners" - [https://arxiv.org/abs/2005.14165](https://arxiv.org/abs/2005.14165)
*   "Tool Learning with Foundation Models" - [https://arxiv.org/abs/2304.08354](https://arxiv.org/abs/2304.08354)
*   "ReAct: Synergizing Reasoning and Acting in Language Models" - [https://arxiv.org/abs/2210.03629](https://arxiv.org/abs/2210.03629)
*   "Chain of Thought Prompting Elicits Reasoning in Large Language Models" - [https://arxiv.org/abs/2201.11903](https://arxiv.org/abs/2201.11903)
*   For more resources on tool calling implementations and best practices, search for "LLM Tool Use" on Google Scholar

