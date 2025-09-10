# Phoenix setup (must be first)
import os
from phoenix.otel import register
import streamlit as st

os.environ["PHOENIX_ENABLED"]="true"
os.environ["PHOENIX_CLIENT_HEADERS"]=st.secrets["PHOENIX_CLIENT_HEADERS"]
os.environ["PHOENIX_API_KEY"] = st.secrets["PHOENIX_API_KEY"]
# Configure Phoenix
os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = st.secrets["PHOENIX_COLLECTOR_ENDPOINT"]
tracer_provider = register(
    project_name="crewAI-trip-planner",
    endpoint=st.secrets["PHOENIX_COLLECTOR_ENDPOINT"],
    auto_instrument=True,
    batch=True
)

# from openinference.instrumentation.litellm import LiteLLMInstrumentor
# LiteLLMInstrumentor().instrument()


from openinference.instrumentation.litellm import LiteLLMInstrumentor
from openinference.instrumentation.crewai import CrewAIInstrumentor

LiteLLMInstrumentor().instrument(tracer_provider=tracer_provider)
CrewAIInstrumentor().instrument(tracer_provider=tracer_provider, skip_dep_check=True)

# from openinference.instrumentation.langchain import LangChainInstrumentor
# LangChainInstrumentor().instrument()

# Now import Streamlit and other dependencies

# Set page config (must be first Streamlit command)
st.set_page_config(
    page_title="AI Travel Planner",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ⬇️ Safely initialize required session state keys before anything else
def initialize_session_state():
    if "current_step" not in st.session_state:
        st.session_state.current_step = "city_selection"
    if "selected_cities" not in st.session_state:
        st.session_state.selected_cities = None
    if "travel_plan" not in st.session_state:
        st.session_state.travel_plan = None


# Import after page config
from trip_planner.app import main

if __name__ == "__main__":
    main() 
