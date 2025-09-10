import logging
from functools import wraps
from crewai import Agent, Task

logging.basicConfig(level=logging.INFO)

class DebugAgent(Agent):
    def __init__(self, *args, max_retries: int = 2, **kwargs):
        super().__init__(*args, **kwargs)
        self._tool_called = False
        self.max_retries = max_retries

        # Wrap each tool with a debug logger
        wrapped_tools = []
        for t in self.tools:
            wrapped_tools.append(self._wrap_tool(t))
        self.tools = wrapped_tools

    def _wrap_tool(self, tool):
        @wraps(tool)
        def wrapped_tool(*args, **kwargs):
            self._tool_called = True
            logging.info(f"[TOOL CALL] {tool.name} args={args} kwargs={kwargs}")
            return tool(*args, **kwargs)
        return wrapped_tool

    def run(self, task: Task, *args, **kwargs):
        attempt = 1
        result = None

        while attempt <= self.max_retries:
            self._tool_called = False
            logging.info(f"[AGENT START] {self.role}: attempt {attempt}, running task '{task.description}'")

            result = super().run(task, *args, **kwargs)

             # If tool was called
            if self._tool_called:
                # If tool returned empty data
                if self._last_tool_output in [None, [], {}, ""]:
                    logging.warning(f"[NO DATA] Tool returned empty response on attempt {attempt}. Forcing agent to return no_data.")
                    return {
                        "status": "no_data",
                        "message": f"No data available from tools for task: {task.description}"
                    }
                logging.info(f"[AGENT SUCCESS] {self.role} succeeded on attempt {attempt}")
                break

            else:
                logging.warning(f"[WARNING] Agent '{self.role}' did NOT use any tools on attempt {attempt}!")
                if attempt < self.max_retries:
                    task.description += "\nREMEMBER: You MUST use the provided tools. Do not answer without calling at least one tool."
                    attempt += 1
                    continue
                else:
                    logging.error(f"[FAILURE] Agent '{self.role}' exhausted {self.max_retries} attempts without tool use.")
            attempt += 1

        logging.info(f"[AGENT END] {self.role} result={result}")
        return result
