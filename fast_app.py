from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import json
from trip_planner import TripAgents  # adjust import path as needed
from crewai import Task
from langchain_openai import AzureChatOpenAI  # or ChatOpenAI, depending on your setup
import os
app = FastAPI()
os.environ["AZURE_API_KEY"] = ""
os.environ["AZURE_API_BASE"] = ""
os.environ["AZURE_API_VERSION"] = "2025-01-01-preview"
 
llm = AzureChatOpenAI(
    azure_deployment="gpt-4.1-nano",
    model_name="azure/gpt-4.1-nano",  # Your Azure deployment name
    azure_endpoint=os.getenv("AZURE_API_BASE"),
    api_key=os.getenv("AZURE_API_KEY"),
    api_version="2025-01-01-preview",  # Your API version
    temperature=0.7,
    streaming=True,
    model_kwargs={"stream_options": {"include_usage": True}}
)  

# Initialize your LLM here with your config
# llm = AzureChatOpenAI(
#     # Add your Azure or OpenAI config here,
#     # e.g. deployment_name="gpt-4-deployment",
#     # temperature=0
# )

trip_agents = TripAgents(llm)

# Pydantic model for input to /run-agent
class AgentInput(BaseModel):
    agent: str
    input: dict  # accept JSON object directly

# Mapping from agent name strings to TripAgents methods
agent_methods = {
    "expert_travel_agent": trip_agents.expert_travel_agent,
    "city_selection_expert": trip_agents.city_selection_expert,
    "local_tour_guide": trip_agents.local_tour_guide,
    "transportation_specialist": trip_agents.transportation_specialist,
    "accommodation_expert": trip_agents.accommodation_expert,
    "food_dining_guide": trip_agents.food_dining_guide,
    "travel_planning_expert": trip_agents.travel_planning_expert,
    "budget_planner": trip_agents.budget_planner,
    "city_classifier": trip_agents.city_classifier,
    "city_justifier": trip_agents.city_justifier,
    "trip_classifier": trip_agents.trip_classifier,
    "trip_justifier": trip_agents.trip_justifier,
}

import json

import json
from crewai import Crew


# Map agent names to their factory methods and expected output strings
agent_methods = {
    "expert_travel_agent": trip_agents.expert_travel_agent,
    "city_selection_expert": trip_agents.city_selection_expert,
    "local_tour_guide": trip_agents.local_tour_guide,
    "transportation_specialist": trip_agents.transportation_specialist,
    "accommodation_expert": trip_agents.accommodation_expert,
    "food_dining_guide": trip_agents.food_dining_guide,
    "travel_planning_expert": trip_agents.travel_planning_expert,
    "budget_planner": trip_agents.budget_planner,
    "city_classifier": trip_agents.city_classifier,
    "city_justifier": trip_agents.city_justifier,
    "trip_classifier": trip_agents.trip_classifier,
    "trip_justifier": trip_agents.trip_justifier,
}

expected_output_map = {
    "expert_travel_agent": "Detailed travel itinerary including budget, packing suggestions, and safety tips.",
    "city_selection_expert": "JSON with recommended city details matching user preferences.",
    "local_tour_guide": "Detailed day tour itinerary with food and cultural recommendations.",
    "transportation_specialist": "Optimized transportation plan with route details and costs.",
    "accommodation_expert": "Accommodation recommendations with booking tips and reviews.",
    "food_dining_guide": "Culinary recommendations and food tour itinerary.",
    "travel_planning_expert": "Travel plan with one activity, meal plan, and budget breakdown.",
    "budget_planner": "Detailed travel budget breakdown in JSON format.",
    "city_classifier": "Classification output: Ideal or Not Ideal.",
    "city_justifier": "Justification for city classification.",
    "trip_classifier": "Classification output: Ideal or Not Ideal trip.",
    "trip_justifier": "Justification for trip classification."
}

def run_agent_task(agent_name: str, input_data: dict):
    if agent_name not in agent_methods:
        raise ValueError(f"Agent '{agent_name}' not found")

    agent = agent_methods[agent_name]()
    input_str = json.dumps(input_data)

    task = {
        "description": input_str,
        "expected_output": expected_output_map.get(agent_name, "Expected output for the task"),
        "agent": agent
    }

    crew = Crew(tasks=[task])
    result = crew.kickoff()

    # Assuming the output key is 'task_1_output' (Crew assigns keys by task index)
    return result

class AgentRequest(BaseModel):
    agent_name: str
    input_data: dict

@app.post("/run-agent")
async def run_task(request: AgentRequest):
    try:
        output = run_agent_task(request.agent_name, request.input_data)
        return {"result": output}
    except Exception as e:
        return {"error": str(e)}

@app.get("/agents")
def list_agents():
    return {"available_agents": list(agent_methods.keys())}
