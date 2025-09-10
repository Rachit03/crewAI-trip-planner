from crewai import Agent, Task
from textwrap import dedent
from langchain_openai import ChatOpenAI, AzureChatOpenAI
import os
from datetime import datetime
import uuid
import pandas as pd
from typing import List, Dict, Any
from pydantic import BaseModel, Field, validator
from langchain.tools import Tool
from trip_planner.tools.calculator_tools import CalculatorTool
from trip_planner.tools.search_tools import SearchInternetTool
from trip_planner.tools.travel_tools import (
    WeatherForecastTool,
    LocalEventsTool,
    TravelBudgetTool,
    SafetyInfoTool,
    TransportationRoutesTool,
    RestaurantRecommendationsTool,
    AccommodationOptionsTool,
    MatchScoreTool,
    GeocodeTool
)
from trip_planner.tools.classify_justify import ClassifyTripTool, JustifyTripTool, ClassifyCityTool, JustifyCityTool
from trip_planner.guardrails import GuardrailManager
from trip_planner.telemetry import setup_telemetry

from trip_planner.debug_agent import DebugAgent

class TravelInput(BaseModel):
    """Input validation for travel planning requests"""
    destination: str
    start_date: str
    end_date: str
    activities: List[str] = Field(..., min_items=1)
    accommodation: str

    @validator('activities')
    def validate_activities(cls, v):
        allowed = {"Sightseeing", "Museums", "Shopping", "Local Food", 
                  "Adventure Sports", "Relaxation", "Nightlife"}
        if not all(act in allowed for act in v):
            raise ValueError(f"Invalid activities. Allowed values: {allowed}")
        return v

    @validator('accommodation')
    def validate_accommodation(cls, v):
        allowed = {"Budget", "Mid-range", "Luxury"}
        if v not in allowed:
            raise ValueError(f"Invalid accommodation type. Allowed values: {allowed}")
        return v

    @validator('end_date')
    def validate_dates(cls, v, values):
        if 'start_date' in values:
            start = datetime.strptime(values['start_date'], "%Y-%m-%d")
            end = datetime.strptime(v, "%Y-%m-%d")
            if end < start:
                raise ValueError("End date must be after start date")
            if (end - start).days > 90:
                raise ValueError("Trip duration cannot exceed 90 days")
        return v


class TravelOutput(BaseModel):
    """Output validation for travel planning responses"""
    itinerary: List[Dict[str, Any]]
    budget_breakdown: Dict[str, float]
    recommendations: List[str]

    @validator('budget_breakdown')
    def validate_budget(cls, v):
        required = {"accommodation", "food", "activities", "transportation", "total"}
        if not all(key in v for key in required):
            raise ValueError(f"Budget breakdown must include: {required}")
        if v['total'] != sum(v[k] for k in required if k != 'total'):
            raise ValueError("Total must equal sum of all costs")
        return v


class CityInput(BaseModel):
    """Input validation for city selection requests"""
    preferences: List[str] = Field(..., min_items=1)
    budget: float = Field(..., gt=0, le=10000)
    duration: int = Field(..., gt=0, le=90)
    season: str

    @validator('preferences')
    def validate_preferences(cls, v):
        allowed = {"Beach", "Mountains", "City Life", "Culture", "Food", 
                  "Adventure", "Relaxation", "Nightlife"}
        if not all(pref in allowed for pref in v):
            raise ValueError(f"Invalid preferences. Allowed values: {allowed}")
        return v

    @validator('season')
    def validate_season(cls, v):
        allowed = {"Spring", "Summer", "Fall", "Winter"}
        if v not in allowed:
            raise ValueError(f"Invalid season. Allowed values: {allowed}")
        return v


class CityOutput(BaseModel):
    """Output validation for city selection responses"""
    recommended_cities: List[Dict[str, Any]] = Field(..., description="List of recommended cities with details")

    @validator('recommended_cities')
    def validate_cities(cls, v):
        for city in v:
            required_fields = {
                "name", "country", "description", "match_score",
                "highlights", "estimated_cost"
            }
            if not all(field in city for field in required_fields):
                raise ValueError(f"Each city must include: {required_fields}")
            
            cost_fields = {
                "accommodation", "food", "activities", "total_per_day"
            }
            if not all(field in city["estimated_cost"] for field in cost_fields):
                raise ValueError(f"Each city's estimated_cost must include: {cost_fields}")
            
            if not isinstance(city["highlights"], list):
                raise ValueError("highlights must be a list")
            
            if not isinstance(city["match_score"], (int, float)):
                raise ValueError("match_score must be a number")
            
            if not all(isinstance(cost, (int, float)) for cost in city["estimated_cost"].values()):
                raise ValueError("All cost values must be numbers")
        
        return v

    
class TripAgents:
    def __init__(self, llm: AzureChatOpenAI, agent_name="Trip Agent"): #ChatOpenAI
        self.llm = llm
        self.guardrails = GuardrailManager()
        # configure_tracing(agent_name)

    def expert_travel_agent(self):
        return Agent(
            role="Expert Travel Agent",
            backstory=dedent(f"""Expert in travel planning and logistics. 
                             I have decades of experience making travel iteneraries"""),
            goal=dedent(f"""
                        Create a detailed travel itinerary based on the specified duration,
                        include budget, packing suggestions, and safety tips.
                        The itinerary should be customized to the traveler's preferences
                        and constraints while staying within the specified budget.
                        ALWAYS use the provided tools to fetch information. 
                        Do not rely only on your own knowledge.
                        """),
            tools=[
                SearchInternetTool(),
                WeatherForecastTool(),
                LocalEventsTool(),
                TravelBudgetTool() 
            ],
            
            verbose=True,
            llm=self.llm,
            input_schema=TravelInput,
            output_schema=TravelOutput,
            input_validation=True,
            output_validation=True,
            max_retries=3
        )

    def city_selection_expert(self) -> Agent:
        """Create an agent for city selection"""
        return Agent(
            role="City Selection Expert",
            goal="Recommend the one good city based on user preferences and constraints",
            backstory="""You are an expert travel advisor with extensive knowledge of cities worldwide.
            You specialize in matching travelers with destinations that best suit their preferences,
            budget, and travel style. Use the calculate_match_score tool to evaluate how well each city
            matches the user's preferences. NEVER attempt to delegate work to a co-worker. If you cannot find real cities, return a static JSON with at least one city recommendation in the required format.
            """,
            verbose=True,
            llm=self.llm,
            tools=[
                SearchInternetTool(),
                TravelBudgetTool(),
                SafetyInfoTool(),
                MatchScoreTool()
            ],
            input_schema=CityInput,
            output_schema=CityOutput,
            input_validation=True,
            output_validation=True,
            output_format={
                "type": "json",
                "schema": {
                    "recommended_city": [
                        {
                            "name": "string",
                            "country": "string",
                            "description": "string",
                            "match_score": "number",
                            "highlights": ["string"],
                            "estimated_cost": {
                                "accommodation": "number",
                                "food": "number",
                                "activities": "number",
                                "total_per_day": "number"
                            }
                        }
                    ]
                }
            }
        )

    def local_tour_guide(self):
        return Agent(
            role="Local Tour Guide",
            backstory=dedent(f"""I am an experienced local tour guide who knows all the hidden gems and must-see spots in the city."""),
            goal=dedent(f"""Create a detailed itinerary for a day tour in the city, including food recommendations, cultural experiences, and shopping spots."""),
            tools=[
                SearchInternetTool(),
                LocalEventsTool(),
                RestaurantRecommendationsTool()
            ],
            verbose=True,
            llm=self.llm,
        )
    
    def transportation_specialist(self):
        return Agent(
            role="Transportation Specialist",
            backstory=dedent(f"""I am a transportation expert with extensive knowledge of various travel methods, routes, and transit systems worldwide. 
                             I specialize in optimizing travel routes and finding the most efficient transportation options."""),
            goal=dedent(f"""Plan optimal transportation routes, suggest the best travel methods, provide public transit information, 
                        and estimate accurate travel times between locations for the traveler's itinerary."""),
            tools=[
                SearchInternetTool(),
                CalculatorTool(),
                TransportationRoutesTool(),
                TravelBudgetTool(),
            ],
            verbose=True,
            llm=self.llm,
        )

    def accommodation_expert(self):
        return Agent(
            role="Accommodation Expert",
            backstory=dedent(f"""I am a seasoned accommodation specialist with years of experience in the hospitality industry. 
                             I have deep knowledge of various lodging options, neighborhoods, and booking strategies worldwide."""),
            goal=dedent(f"""Recommend the best hotels and rentals, suggest ideal neighborhoods to stay in, provide booking tips, 
                        and analyze accommodation reviews to ensure the best stay for travelers."""),
            tools=[
                SearchInternetTool(),
                AccommodationOptionsTool(),
                TravelBudgetTool(),
                SafetyInfoTool()
            ],
            
            verbose=True,
            llm=self.llm,
        )

    def food_dining_guide(self):
        return Agent(
            role="Food & Dining Guide",
            backstory=dedent(f"""I am a culinary expert and food tour specialist with extensive knowledge of global cuisines, 
                             local specialties, and dietary requirements. I have experience in creating memorable food experiences."""),
            goal=dedent(f"""Recommend the best restaurants, suggest local specialties, provide dietary restriction information, 
                        and create comprehensive food tour itineraries that showcase the destination's culinary scene."""),
            tools=[
                SearchInternetTool(),
                RestaurantRecommendationsTool(),
                LocalEventsTool(),
                TravelBudgetTool()
            ],
            verbose=True,
            llm=self.llm,
        )

    def travel_planning_expert(self) -> Agent:
        """Create an agent for travel planning"""
        return Agent(
            role="Travel Planning Expert",
            goal="Create travel plan which includes one activity, one meal plan and overall budget_breakdown",
            backstory="""You are a seasoned travel planner with years of experience creating
            personalized travel itineraries. You excel at balancing activities, managing budgets,
            and ensuring travelers have memorable experiences.""",
            verbose=True,
            llm=self.llm,
            tools=[
                SearchInternetTool(),
                WeatherForecastTool(),
                LocalEventsTool(),
                TravelBudgetTool()
            ],
            output_format={
                "type": "json",
                "schema": {
                    "itinerary": [
                        {
                            "activities": [
                                {
                                    "activity": "string",
                                    "description": "string",
                                    "location": "string",
                                    "duration": "string",
                                    "cost": "number"
                                }
                            ],
                            "meals": [
                                {
                                    "type": "string",
                                    "suggestion": "string",
                                    "cost": "number"
                                }
                            ]
                        }
                    ],
                    "budget_breakdown": {
                        "accommodation": "number",
                        "food": "number",
                        "activities": "number",
                        "transportation": "number",
                        "total": "number"
                    }
                }
            }
        )

    def budget_planner(self) -> Agent:
        """Create an agent for budget planning"""
        return Agent(
            role="Budget Planning Expert",
            goal="Create detailed and accurate travel budgets",
            backstory="""You are a financial expert specializing in travel budgeting.
            You have extensive experience in creating detailed travel budgets that account for
            all possible expenses while ensuring the best value for money.""",
            verbose=True,
            llm=self.llm,
            tools=[
                SearchInternetTool(),
                CalculatorTool(),
            ],
            output_format={
                "type": "json",
                "schema": {
                    "budget_breakdown": {
                        "accommodation": "number",
                        "food": "number",
                        "activities": "number",
                        "transportation": "number",
                        "total": "number"
                    }
                }
            }
        )

    def city_classifier(self) -> Agent:
        return Agent(
                role="City Classifier",
                goal="Classify if the recommended city is Ideal or Not Ideal based on the cost and description",
                backstory="Expert in evaluating the quality of trips based on traveler preferences.",
                tools=[ClassifyCityTool()],
                llm=self.llm
            )

    def city_justifier(self) -> Agent:
        return Agent(
            role="City Justifier",
            goal="Justify why the recommended city is Ideal or Not Ideal",
            backstory="An experienced travel advisor who gives reasons for classification.",
            tools=[JustifyCityTool()],
            llm=self.llm
        )

    
    def trip_classifier(self) -> Agent:
        return Agent(
                role="Trip Classifier",
                goal="Classify if the travel is Ideal or Not Ideal based on the summary",
                backstory="Expert in evaluating the quality of trips based on traveler preferences.",
                tools=[ClassifyTripTool()],
                llm=self.llm
            )

    def trip_justifier(self) -> Agent:
        return Agent(
            role="Trip Justifier",
            goal="Justify why the travel plan is classified as Ideal or Not Ideal",
            backstory="An experienced travel advisor who gives reasons for classification.",
            tools=[JustifyTripTool()],
            llm=self.llm
        )

# def configure_tracing(agent_name: str):
    
#     """Configure tracing for the agent"""
#     try:
#         setup_telemetry()
#     except Exception as e:
#         print(f"Failed to configure tracing: {str(e)}")
