from typing import Type
from pydantic import BaseModel, Field
from crewai.tools import BaseTool

class CalculatorInput(BaseModel):
    """Input schema for CalculatorTool."""
    expression: str = Field(..., description="The mathematical expression to evaluate")

class CalculatorTool(BaseTool):
    name: str = "calculate"
    description: str = "Safely evaluate mathematical expressions like '2 + 3 * (4 - 1)'"
    args_schema: Type[BaseModel] = CalculatorInput

    def _run(self, expression: str) -> str:
        """Safely evaluate a math expression."""
        try:
            result = eval(expression, {"__builtins__": {}}, {"abs": abs, "round": round})
            return str(result)
        except Exception as e:
            return f"Error calculating expression: {str(e)}"
