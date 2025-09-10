import os
os.environ["AZURE_API_KEY"] = "90858d0b603c4323a2df07d8064dbcf6"
os.environ["AZURE_API_BASE"] = "https://llm-mlops-openai.openai.azure.com/"
os.environ["AZURE_API_VERSION"] = "2025-01-01-preview"

from crewai import Agent, Crew,Task
from langchain_openai import AzureChatOpenAI
from crewai.tools import BaseTool

# Define the AzureChatOpenAI LLM
llm = AzureChatOpenAI(
    api_key="90858d0b603c4323a2df07d8064dbcf6",
    azure_endpoint="https://llm-mlops-openai.openai.azure.com/",
    api_version="2024-12-01-preview",
    azure_deployment="gpt-4o-mini",
    # api_base="https://llm-mlops-openai.openai.azure.com/",    
    model_name="azure/gpt-4o-mini",
    # client="AzureOpenAI"

)

class ClassifyCityTool(BaseTool):
    name: str = "classify_city"
    description: str = "Classify the city recommendation as Ideal or Not Ideal"

    def _run(self, recommendation: str) -> str:
        return llm.invoke(f"Classify if the recommended city is a Ideal or Not based on the cost and description: {recommendation}")


class JustifyCityTool(BaseTool):
    name: str = "justify_city"
    description: str = "Justify why the recommended City is classified as ideal or Not ideal"

    def _run(self, recommendation: str, classification: str) -> str:
        return llm.invoke(
            f"The city was classified as {classification}. Summary: {recommendation}. Explain why."
        )


class ClassifyTripTool(BaseTool):
    name: str = "classify_trip"
    description: str = "Classify the travel plan as Ideal or Not Ideal"

    def _run(self, summary: str) -> str:
        return llm.invoke(f"Classify the following travel plan as Ideal or Not Ideal: Travel plan::{summary}")


class JustifyTripTool(BaseTool):
    name: str = "justify_trip"
    description: str = "Justify why the travel plan is classified as Ideal or Not Ideal"

    def _run(self, classification: str, summary: str) -> str:
        return llm.invoke(
            f"The trip was classified as {classification}. Summary: {summary}. Explain why."
        )

