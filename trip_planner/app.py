import streamlit as st
import json
from datetime import datetime, timedelta
import pandas as pd
import plotly.express as px
import os
import litellm
from dotenv import load_dotenv
load_dotenv()
from trip_planner.telemetry import setup_telemetry
from langchain_openai import ChatOpenAI,AzureChatOpenAI
from .agents2 import TripAgents, TravelInput, CityInput
from .guardrails import GuardrailManager
from .tools.travel_tools import WeatherForecastTool, LocalEventsTool,SafetyInfoTool
from crewai import Task, Crew
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode
import openai
import re
from opentelemetry.context import get_current
from streamlit_javascript import st_javascript
import requests




# -------------------- NEW HELPER --------------------
def safe_json_loads(raw_text: str):
    """
    Cleans and safely parses JSON returned by an agent.
    - Removes code fences like ```json ... ```
    - Cuts off extra junk like 'undefined'
    - Ensures only valid JSON is passed to json.loads
    """
    try:
        # Remove markdown code fences
        cleaned = re.sub(r"^```json\s*|\s*```$", "", raw_text.strip())

        # Sometimes LLM appends junk after the last closing brace
        last_brace = cleaned.rfind("}")
        if last_brace != -1:
            cleaned = cleaned[:last_brace+1]

        return json.loads(cleaned)
    except Exception as e:
        st.error(f"JSON parsing failed: {e}")
        st.write("Raw agent output:", raw_text)
        return {"error": "invalid_json", "raw": raw_text}
# ----------------------------------------------------

st.success("Telemetry initialized successfully!")
tracer = trace.get_tracer(__name__)


st.info("Test trace created successfully!")
# from opentelemetry.instrumentation.crewai import CrewAIInstrumentor

# CrewAIInstrumentor().instrument(skip_dep_check=True)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
    }
    .css-1d391kg {
        padding: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)


# set environment as LiteLLM expects
os.environ["AZURE_API_KEY"] = st.secrets["AZURE_API_KEY"]
os.environ["AZURE_API_BASE"] = st.secrets["AZURE_API_BASE"]
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
    


# Initialize agents
agents = TripAgents(llm)

# Initialize guardrails
guardrails = GuardrailManager()

def get_user_location():
    ip = st_javascript("await fetch('https://api.ipify.org?format=json').then(res => res.json()).then(data => data.ip)")
    print("*"*20)
    print("ip: ", ip)
    if not ip:
        # st.warning("IP not yet retrieved. Please wait or refresh.")
        st.stop()
    print("ip: ", ip)
    if ip != 0:
        location_data = get_ip_info(ip)

        print("*"*20)
        print("location_data: ", location_data)

        return location_data
    else:
        return None

def get_ip_info(ip_address):
    try:
        url = f"https://ipapi.co/{ip_address}/json/"
        print("url: ", url)
        response = requests.get(url)
        print("response: ", response)
        print("response: ", response.reason)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": "Could not retrieve data"}
    except Exception as e:
        return {"error": str(e)}

location_data = get_user_location()

def run_crew_with_location_context(crew):
    """Wrapper to add custom attributes to crew execution"""
    with tracer.start_as_current_span("crew-kickoff-with-metadata") as span:
        # Add user context
        global location_data
        if location_data:
            for key, value in location_data.items():
                span.set_attribute(key, value)
            span.set_attribute("location_source", "auto_detected")        
        # Execute the crew
        result = crew.kickoff()
        
        # Add success metrics
        span.set_attribute("execution_status", "success")
        span.set_attribute("result_length", len(str(result)) if result else 0)
        
        return result


def initialize_session_state():
    """Initialize session state variables"""
    if 'travel_plan' not in st.session_state:
        st.session_state.travel_plan = None
    if 'selected_cities' not in st.session_state:
        st.session_state.selected_cities = None
    if 'current_step' not in st.session_state:
        st.session_state.current_step = 'city_selection'
    if 'show_proceed_button' not in st.session_state:
        st.session_state.show_proceed_button = False

def display_header():
    """Display the application header"""
    st.title("✈️ AI Travel Planner")
    st.markdown("""
    Plan your perfect trip with AI-powered travel agents. 
    Get personalized recommendations, detailed itineraries, and comprehensive travel plans.
    """)

def display_weather_forecast(destination: str, date: str):
    """Display weather forecast for a destination"""

    try:
        with st.spinner("Getting weather forecast..."):
            weather_tool = WeatherForecastTool()
            weather = json.loads(weather_tool._run(destination, date))
            col1, col2, col3, col4 = st.columns([1, 2, 1, 1])
            col1.metric("Temperature", f"{weather['temperature']}°C")
            col2.markdown(f"<span style='font-size:1.5em'>{weather['condition']}</span>", unsafe_allow_html=True)
            col3.metric("Humidity", f"{weather['humidity']}%")
            col4.metric("Wind Speed", f"{weather['wind_speed']} km/h")
        st.markdown("<br>", unsafe_allow_html=True)
        # span.set_status(Status(StatusCode.OK))
    except Exception as e:
        # span.set_status(Status(StatusCode.ERROR, str(e)))
        st.error(f"Error getting weather: {e}")

def display_safety_info(destination: str):
    """Display safety information for a destination"""

    try:
        with st.spinner("Getting safety information..."):
            safety_tool = SafetyInfoTool()
            safety_info = json.loads(safety_tool._run(destination))
            st.subheader("Safety Information")
            st.info(f"General Safety: {safety_info['general_safety']}")
            st.info(f"Health Concerns: {safety_info['health_concerns']}")
            st.info(f"Crime Rate: {safety_info['crime_rate']}")
            st.info(f"Natural Disasters: {safety_info['natural_disasters']}")
        st.markdown("<br>", unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error getting safety info: {e}")

def display_local_events(destination: str, date_range: dict = None):
    """Display local events for a destination"""
    try:
        with st.spinner("Getting local events..."):
            events_tool = LocalEventsTool()
            events = json.loads(events_tool._run(destination, date_range))
            st.subheader("Local Events")
            for event in events:
                with st.expander(f"🎉 {event.get('name', 'Event')} - {event.get('date', '')}"):
                    st.markdown(f"**Location:** {event.get('location', event.get('venue', ''))}")
                    st.markdown(f"**Description:** {event.get('description', '')}")
    except Exception as e:
        st.error(f"Error getting local events: {e}")

def display_budget_analysis(budget_breakdown: dict):
    """Display budget analysis with visualizations"""
    st.subheader("Budget Analysis")
    
    # Create pie chart for budget distribution
    fig = px.pie(
        values=list(budget_breakdown.values()),
        names=list(budget_breakdown.keys()),
        title="Budget Distribution"
    )
    st.plotly_chart(fig)
    
    # Display budget metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Accommodation", f"${budget_breakdown['accommodation']}")
    col2.metric("Food", f"${budget_breakdown['food']}")
    col3.metric("Activities", f"${budget_breakdown['activities']}")
    col4.metric("Transportation", f"${budget_breakdown['transportation']}")
    col5.metric("Total", f"${budget_breakdown['total']}")

def display_city_comparison(cities: list):
    """Display comparison of recommended cities only if 2 or more cities are present."""
    if len(cities) < 3:
        return  # Skip plots for 1-2 cities
    st.subheader("City Comparison")
    metrics = {
        "Match Score": [city["match_score"] for city in cities],
        "Daily Cost": [city["estimated_cost"]["total_per_day"] for city in cities],
        "City": [city["name"] for city in cities]
    }
    fig = px.bar(
        x="City",
        y="Match Score",
        data_frame=pd.DataFrame(metrics),
        title="City Match Scores"
    )
    st.plotly_chart(fig)
    fig = px.bar(
        x="City",
        y="Daily Cost",
        data_frame=pd.DataFrame(metrics),
        title="Daily Costs Comparison"
    )
    st.plotly_chart(fig)

def display_city_recommendations(cities):
    """Display city recommendations as summary cards for all cities."""
    st.subheader("Recommended Cities")
    display_city_comparison(cities)
    # Show all cities as cards in a grid
    cols = st.columns(min(3, len(cities)))
    for idx, city in enumerate(cities):
        with cols[idx % len(cols)]:
            st.markdown(f"### 🌆 {city['name']}, {city['country']} (Score: {city['match_score']:.2f})")
            st.markdown(f"**Description:** {city['description']}")
            st.markdown(f"**Estimated Daily Cost:** ${city['estimated_cost']['total_per_day']}")
            st.markdown(f"**Highlights:** {' | '.join(city['highlights'])}")
            # Weather summary
            weather_tool = WeatherForecastTool()
            weather = json.loads(weather_tool._run(city['name'], datetime.now().strftime("%Y-%m-%d")))
            st.markdown(f"**Weather:** {weather['temperature']}°C, {weather['condition']}")
            # Events summary
            events_tool = LocalEventsTool()
            events = json.loads(events_tool._run(city['name'], None))
            if events:
                st.markdown(f"**Events:** {events[0].get('name', 'Event')} ({events[0].get('date', '')[:10]})")
            # Safety summary
            st.markdown("[Travel Advisory](https://www.travel-advisory.info/) for safety info.")
            st.markdown("---")
    # Optionally, expand for details
    for city in cities:
        with st.expander(f"More about {city['name']}"):
            st.markdown(f"**Description:** {city['description']}")
            st.markdown("**Highlights:**")
            for highlight in city['highlights']:
                st.markdown(f"- :star: {highlight}")
            st.markdown("**Weather Forecast:**")
            display_weather_forecast(city['name'], datetime.now().strftime("%Y-%m-%d"))
            display_safety_info(city['name'])
            st.markdown("**Events:**")
            events_tool = LocalEventsTool()
            for event in json.loads(events_tool._run(city['name'], None)):
                st.markdown(f"- {event.get('name', 'Event')} ({event.get('date', '')[:10]})")
            st.markdown("**Estimated Daily Costs:**")
            costs = city['estimated_cost']
            st.metric("Accommodation", f"${costs['accommodation']}")
            st.metric("Food", f"${costs['food']}")
            st.metric("Activities", f"${costs['activities']}")
            st.metric("Total per day", f"${costs['total_per_day']}", 
                     delta=f"${costs['total_per_day'] - 200:.2f} vs budget")
            st.markdown("<br>", unsafe_allow_html=True)

def display_travel_plan(plan, travel_class,travel_justify):
    """Display travel plan in a visually appealing way"""
    st.subheader("Your Travel Plan")
    
    # Display budget breakdown with analysis
    display_budget_analysis(plan['budget_breakdown'])
    
    # Display itinerary
    print("PLAN****",plan)
    st.markdown("### 📅 Daily Itinerary")
    for day in plan['itinerary']:
           
            # Display activities
            st.markdown("#### 🎯 Activities")
            for activity in day['activities']:
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"**{activity['activity']}**")
                    st.markdown(f"📍 {activity['location']}")
                    st.markdown(f"⏱️ {activity['duration']}")
                with col2:
                    st.metric("Cost", f"${activity['cost']}")
                st.markdown(f"_{activity['description']}_")
                st.divider()
            
            # Display meals
            st.markdown("#### 🍽️ Meals")
            for meal in day['meals']:
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"**{meal['type']}**")
                    st.markdown(f"🍴 {meal['suggestion']}")
                with col2:
                    st.metric("Cost", f"${meal['cost']}")
    
    # Display recommendations
    
    st.markdown("### 💡 Recommendations")
    for i, rec in enumerate(plan['recommendations'], 1):
        st.markdown(f"{i}. {rec}")

    st.markdown("### 💡 Travel Categorization")
    st.text_area("Travel Category", value=travel_class, height=150, disabled=True)
    st.text_area("Category Justification", value=travel_justify, height=150, disabled=True)
    

def city_selection_form():
    """Display the city selection form and handle city recommendations"""
    st.subheader("Step 1: Select Your Destination")
    
    with st.form("city_selection"):
            # Get user preferences
            preferences = st.multiselect(
                "What are you looking for in your destination?",
                ["Beach", "Mountains", "City Life", "Culture", "Food", "Adventure", "Relaxation", "Nightlife"],
                default=["Beach", "Culture"]
            )
            
            budget = st.slider(
                "What's your budget per day (in USD)?",
                min_value=0,
                max_value=500,
                value=100,
                step=50
            )
            
            duration = st.slider(
                "How many days do you want to travel?",
                min_value=0,
                max_value=15,
                value=2,
                step=1
            )
            
            season = st.selectbox(
                "When do you plan to travel?",
                ["Spring", "Summer", "Fall", "Winter"]
            )
            
            submitted = st.form_submit_button("Get City Recommendations")
            
            if submitted:
                # Validate input using guardrails
                input_data = {
                    "preferences": preferences,
                    "budget": budget,
                    "duration": duration,
                    "season": season
                }
                
                is_valid, error_message = guardrails.validate_input(input_data)
                if not is_valid:
                    st.error(error_message)
                    return
                
                # Create input for city selection
                city_input = CityInput(
                    preferences=preferences,
                    budget=budget,
                    duration=duration,
                    season=season
                )
                
                # Get city recommendations
                with st.spinner("Getting city recommendations..."):
                    city_expert = agents.city_selection_expert()
                    city_classifier_expert = agents.city_classifier()
                    city_justifier_expert = agents.city_justifier()


                    crew = Crew(
                        tasks=[
                            {
                        "description":f"""Based on these preferences: {city_input.dict()}, recommend one city for travel.
                        Return output for only one city
                        Your response MUST be a valid JSON object with the following structure:
                        {{
                            "recommended_city": [
                                {{
                                    "name": "City Name",
                                    "country": "Country Name",
                                    "description": "Brief description of the city",
                                    "match_score": 0.95,
                                    "highlights": ["Highlight 1", "Highlight 2"],
                                    "estimated_cost": 
                                    {{
                                        "accommodation": 100,
                                        "food": 50,
                                        "activities": 75,
                                        "total_per_day": 225
                                    }}
                                }}
                            ]
                        }}""",
                        "expected_output":"JSON with a recommended city and its details.",
                        "agent":city_expert
                    
                    },
                    {
                    "description":"Classify if the city recommended below is Ideal or Not Ideal: \n\n'{{task_1_output}}'",
                        "expected_output":"Based on the recommended city classifiy if the city is either 'Ideal' or 'Not Ideal' place to visit",
                        "agent":city_classifier_expert
                    },
                    {
                        "description":"Explain in two lines why the city is '{{task_2_output}}' using the previous recommendation: '{{task_1_output}}'.",
                        "expected_output":"Detailed justification",
                        "agent":city_justifier_expert
                    }
                        ]
                    )

                        
                    result = None 
                    
                    try:
                        # result = crew.kickoff()
                        result = run_crew_with_location_context(
                            crew
                        )
                    except Exception as e:
                        st.error(f"Agent execution error: {e}")
                        import traceback
                        traceback.print_exc()
                        return
                   
                    try:
                        # The result from CrewAI is typically a CrewOutput object
                        # Extract the raw text from the result
                        if hasattr(result.tasks_output[0].raw, 'raw'):
                            raw_text = result.raw
                        elif isinstance(result.tasks_output[0].raw, str):
                            raw_text = result.tasks_output[0].raw
                        else:
                            raw_text = str(result.tasks_output[0].raw)
                        
                        st.write("DEBUG - Raw result:", raw_text)
                        
                        # Parse the JSON from the raw text
                        result_data = safe_json_loads(raw_text)
                        st.write("DEBUG - Parsed result data:", result_data)
                        
                        
                        # Validate the structure
                        if not isinstance(result_data, dict):
                            st.error("Invalid response format: expected a dictionary")
                            return
                        
                        if 'recommended_city' not in result_data:
                            st.error("Invalid response format: missing 'recommended_city' key")
                            return
                        
                        if not isinstance(result_data['recommended_city'], list):
                            st.error("Invalid response format: 'recommended_city' should be a list")
                            return
                        
                        # Validate each city's structure
                        for i, city in enumerate(result_data['recommended_city']):
                            required_fields = {
                                "name", "country", "description", "match_score",
                                "highlights", "estimated_cost"
                            }
                            missing_fields = required_fields - set(city.keys())
                            if missing_fields:
                                st.error(f"City {i+1} is missing required fields: {missing_fields}")
                                return
                            
                            if not isinstance(city['estimated_cost'], dict):
                                st.error(f"City {i+1}: 'estimated_cost' should be a dictionary")
                                return
                            
                            cost_fields = {
                                "accommodation", "food", "activities", "total_per_day"
                            }
                            missing_costs = cost_fields - set(city['estimated_cost'].keys())
                            if missing_costs:
                                st.error(f"City {i+1}: 'estimated_cost' is missing fields: {missing_costs}")
                                return
                        
                        # Store in session state
                        st.session_state.selected_cities = result_data
                        st.session_state.current_step = 'travel_planning'
                        st.session_state.selected_budget = budget
                        st.session_state.selected_preferences = preferences
                        st.session_state.duration = duration
                        st.session_state.city_class= result.tasks_output[1].raw
                        st.session_state.city_justify= result.tasks_output[2].raw
                        try:
                            print("******************************result.tasks_output[1]:",result.tasks_output[1],"$$",result.tasks_output[1])
                            print("******************************result.tasks_output[2]:",result.tasks_output[2])
                        except:
                            pass
                        display_city_recommendations(result_data['recommended_city'])
                        st.session_state.show_proceed_button = False
                        st.rerun()

                    
                
                    # except json.JSONDecodeError as e:
                    #     st.error("Invalid JSON response from the AI. Please try again.")
                    #     st.write("Raw result that failed to parse:", result)
                    #     # Provide a fallback response
                    #     fallback_response = {
                    #         "recommended_city": [
                    #             {
                    #                 "name": "Barcelona",
                    #                 "country": "Spain",
                    #                 "description": "A vibrant city known for its beaches and rich cultural heritage.",
                    #                 "match_score": 0.9,
                    #                 "highlights": ["Sagrada Familia", "Beach", "Local Cuisine"],
                    #                 "estimated_cost": {
                    #                     "accommodation": 80,
                    #                     "food": 40,
                    #                     "activities": 30,
                    #                     "total_per_day": 150
                    #                 }
                    #             }
                    #         ]
                    #     }
                    #     st.session_state.selected_cities = fallback_response
                        #display_city_recommendations(fallback_response['recommended_cities'])
                        # st.session_state.trigger_next_step = True
                    except Exception as e:
                        st.error(f"Unexpected error: {str(e)}")
                        st.write("Debug - Full error:", e)

    # OUTSIDE the form, show the proceed button if flag is set
    if st.session_state.get("show_proceed_button"):
        if st.button("Proceed to Travel Planning"):
            st.session_state.current_step = 'travel_planning'
            st.session_state.show_proceed_button = False
            st.rerun()

def travel_planning_form():
    """Display the travel planning form and handle travel plan generation"""
    st.subheader("Step 2: Plan Your Trip")

   
    with st.form("travel_planning"):
            # Get selected city
            selected_city = st.selectbox(
                "Select your destination",
                [city["name"] for city in st.session_state.selected_cities['recommended_city']]
            )
            
            # Get travel dates
            start_date = st.date_input(
                "Start Date",
                min_value=datetime.now().date(),
                value=datetime.now().date() + timedelta(days=7)
            )
            
            end_date = st.date_input(
                "End Date",
                min_value=start_date,
                value=start_date + timedelta(days=6)
            )
            
            # Get additional preferences
            activities = st.multiselect(
                "What activities interest you?",
                ["Sightseeing", "Museums", "Shopping", "Local Food", "Adventure Sports", "Relaxation", "Nightlife"],
                default=["Sightseeing", "Local Food"]
            )
            
            accommodation = st.selectbox(
                "Preferred accommodation type",
                ["Budget", "Mid-range", "Luxury"]
            )

            st.markdown("### 💡 City Categorization")
            st.text_area("City Category", value=st.session_state.city_class, height=70, disabled=True)
            st.text_area("Category Justification", value=st.session_state.city_justify, height=70, disabled=True)
            
            
            submitted = st.form_submit_button("Generate Travel Plan")
            # print("BUDGET---------------------------",st.session_state.budget)
            
            if submitted:
                # Validate input using guardrails
                input_data = {
                    "start_date": start_date.strftime("%Y-%m-%d"),
                    "end_date": end_date.strftime("%Y-%m-%d"),
                    "activities": activities
                }
                
                
                # Create input for travel planning
                travel_input = TravelInput(
                    destination=selected_city,
                    start_date=start_date.strftime("%Y-%m-%d"),
                    end_date=end_date.strftime("%Y-%m-%d"),
                    activities=activities,
                    accommodation=accommodation
                )
                
                # Generate travel plan
                with st.spinner("Generating your travel plan..."):
                    travel_expert = agents.travel_planning_expert()
                    trip_classifier_expert = agents.trip_classifier()
                    trip_justifier_expert = agents.trip_justifier()
                    
                    crew = Crew(
                        agents=[travel_expert,trip_classifier_expert,trip_justifier_expert],
                        tasks=[{
                            "description":f"""Create a detailed travel plan based on these preferences: {travel_input.dict()}
                        Your response MUST be a valid JSON object with the following structure:
                        {{
                            "itinerary": [
                                {{
                                    "activities": [
                                        {{
                                            "activity": "Activity name",
                                            "description": "Activity description",
                                            "location": "Location name",
                                            "duration": "2 hours",
                                            "cost": 50
                                        }}
                                    ],
                                    "meals": [
                                        {{
                                            "type": "Lunch",
                                            "suggestion": "Restaurant name",
                                            "cost": 30
                                        }}
                                    ]
                                }}
                            ],
                            "budget_breakdown": {{
                                "accommodation": 500,
                                "food": 300,
                                "activities": 400,
                                "transportation": 200,
                                "total": 1400
                            }},
                            "recommendations": [
                                "Recommendation 1",
                                "Recommendation 2"
                            ]
                        }}""",
                        "agent":travel_expert,
                        "expected_output":"A detailed travel plan in JSON format as specified, with itinerary, budget, and recommendations."

                        },
                        {
                    "description":"Using the travel plan: '{{task_1_output}}', classify the if the travel plan identified is 'Ideal' or 'Not Ideal' basd on the budget,itinerary and recommendations",
                        "expected_output":"Display either 'Ideal' or 'Not Ideal'",
                        "agent":trip_classifier_expert
                    },
                    {
                        "description":"Explain in two sentences why the travel plan is '{{task_2_output}}' using the previous recommendation: '{{task_1_output}}'",
                        "expected_output":"Detailed justification",
                        "agent":trip_justifier_expert
                    }
                        ],
                    sequential=True
                    )
                    # with tracer.start_as_current_span("travel_plan_generation_task") as span:  # ✅ Tracing block added
                    #     span.set_attribute("destination", travel_input.destination)
                    #     span.set_attribute("duration_days", (end_date - start_date).days)
                    #     span.set_attribute("activities", str(activities))
                    try:
                        # result = crew.kickoff()
                        result = run_crew_with_location_context(
                            crew
                        )

                       
                    except Exception as e:
                        # span.set_status(Status(StatusCode.ERROR, str(e)))
                        st.error(f"Agent execution error: {e}")
                        import traceback
                        traceback.print_exc()
                        return
                    
                    # Parse the result
                    # try:
                    # Extract raw text from the result
                    if hasattr(result.tasks_output[0].raw, 'raw'):
                        raw_text = result.tasks_output[0].raw
                        print("AA")
                    elif isinstance(result.tasks_output[0].raw, str):
                        raw_text = result.tasks_output[0].raw
                        print("BB")
                    else:
                        raw_text = str(result.tasks_output[0].raw)
                        print("CC")
                    print("RAW_TEXT::",raw_text,"**",type(raw_text))
                    result_data = safe_json_loads(raw_text)
                # Validate output using guardrails
                #try:
                    #result_data = json.loads(result)
                    is_valid, error_message = guardrails.validate_output(result_data, "travel_plan")
                    if not is_valid:
                        st.error(error_message)
                        return
                    
                    # Validate business rules
                    is_valid, error_message = guardrails.validate_business_rules(result_data)
                    if not is_valid:
                        st.error(error_message)
                        return
                    
                    st.session_state.travel_plan = result_data
                    st.session_state.travel_plan_class = result.tasks_output[1].raw
                    st.session_state.travel_plan_justify = result.tasks_output[2].raw
                    display_travel_plan(st.session_state.travel_plan, st.session_state.travel_plan_class,st.session_state.travel_plan_justify)
                    # except json.JSONDecodeError:
                    #     st.error("Invalid response format from the AI. Please try again.")

def main():

    """Main function to run the Streamlit app"""
    initialize_session_state()
    display_header()

    # Sidebar navigation
    st.sidebar.title("Navigation")
    st.sidebar.markdown(f"**Current Step →** `{st.session_state.current_step}`")   #new
    if st.sidebar.button("Start Over"):
        st.session_state.current_step = 'city_selection'
        st.session_state.travel_plan = None
        st.session_state.selected_cities = None
        st.rerun()
    
    # Main content based on current step
    if st.session_state.current_step == 'city_selection':
        city_selection_form()
    elif st.session_state.current_step == 'travel_planning':
        if st.session_state.selected_cities is None:
            st.error("No cities selected. Please go back to city selection.")
            if st.button("Back to City Selection"):
                st.session_state.current_step = 'city_selection'
                st.rerun()
        else:
            travel_planning_form()
    
    # Show success message when travel plan is ready
    if st.session_state.travel_plan:
        st.success("Your travel plan is ready! 🎉")
        st.balloons()

if __name__ == "__main__":
    main() 
