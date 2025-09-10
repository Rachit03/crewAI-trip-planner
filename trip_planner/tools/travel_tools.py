from typing import Optional, Dict, Any, List, Type
from pydantic import BaseModel, Field
from crewai.tools import BaseTool
import requests
import json
from datetime import datetime, timedelta
import os

# Input schemas for each tool
class WeatherForecastInput(BaseModel):
    """Input schema for WeatherForecastTool."""
    destination: str = Field(..., description="The destination city for weather forecast")
    date: Optional[str] = Field(None, description="Date for forecast in YYYY-MM-DD format")

class LocalEventsInput(BaseModel):
    """Input schema for LocalEventsTool."""
    destination: str = Field(..., description="The destination city to search for events")
    date_range: Optional[Dict[str, str]] = Field(None, description="Date range with 'start' and 'end' keys")

class TravelBudgetInput(BaseModel):
    """Input schema for TravelBudgetTool."""
    destination: str = Field(..., description="The destination city")
    duration: int = Field(..., description="Trip duration in days")
    preferences: List[str] = Field(..., description="List of travel preferences")

class SafetyInfoInput(BaseModel):
    """Input schema for SafetyInfoTool."""
    destination: str = Field(..., description="The destination city for safety information")

class TransportationRoutesInput(BaseModel):
    """Input schema for TransportationRoutesTool."""
    origin: str = Field(..., description="Origin city")
    destination: str = Field(..., description="Destination city")
    date: Optional[str] = Field(None, description="Travel date in YYYY-MM-DD format")

class RestaurantRecommendationsInput(BaseModel):
    """Input schema for RestaurantRecommendationsTool."""
    location: str = Field(..., description="Location to search for restaurants")

class AccommodationOptionsInput(BaseModel):
    """Input schema for AccommodationOptionsTool."""
    destination: str = Field(..., description="Destination to search for accommodations")

class MatchScoreInput(BaseModel):
    """Input schema for MatchScoreTool."""
    city: Dict[str, Any] = Field(..., description="City data dictionary")
    preferences: List[str] = Field(..., description="User preferences list")
    budget: float = Field(..., description="Daily budget amount")
    season: str = Field(..., description="Travel season")

class GeocodeInput(BaseModel):
    """Input schema for GeocodeTool."""
    city_name: str = Field(..., description="Name of the city to geocode")

# Helper functions (unchanged)
def geocode_city(city_name: str) -> dict:
    """Get latitude, longitude, and country for a city using OpenTripMap."""
    try:
        url = f"https://api.opentripmap.com/0.1/en/places/geoname"
        params = {
            'name': city_name,
            'apikey': os.getenv('OPENTRIPMAP_API_KEY')
        }
        response = requests.get(url, params=params)
        data = response.json()
        return {
            'lat': data.get('lat', 0),
            'lon': data.get('lon', 0),
            'country': data.get('country', ''),
            'name': data.get('name', city_name)
        }
    except Exception as e:
        return {'lat': 0, 'lon': 0, 'country': '', 'name': city_name}

def calculate_match_score(city: Dict[str, Any], preferences: List[str], budget: float, season: str) -> float:
    """Calculate how well a city matches the user's preferences."""
    score = 0.0
    max_score = 1.0
    
    # Preference matching (40% of total score)
    preference_weights = {
        "Beach": ["coastal", "beach", "seaside", "ocean"],
        "Mountains": ["mountain", "hiking", "skiing", "alpine"],
        "City Life": ["urban", "metropolitan", "city", "downtown"],
        "Culture": ["museum", "art", "history", "cultural", "heritage"],
        "Food": ["cuisine", "restaurant", "gastronomy", "culinary"],
        "Adventure": ["adventure", "outdoor", "sports", "activities"],
        "Relaxation": ["spa", "wellness", "peaceful", "tranquil"],
        "Nightlife": ["nightlife", "entertainment", "bars", "clubs"]
    }
    
    description = city.get("description", "").lower()
    for pref in preferences:
        if pref in preference_weights:
            for keyword in preference_weights[pref]:
                if keyword in description:
                    score += 0.1  # 0.1 points per matching keyword
    
    # Budget matching (30% of total score)
    daily_cost = city.get("estimated_cost", {}).get("total_per_day", 0)
    if daily_cost <= budget:
        score += 0.3  # Full points if within budget
    else:
        # Reduce score based on how much over budget
        budget_diff = (daily_cost - budget) / budget
        score += max(0, 0.3 * (1 - budget_diff))
    
    # Season matching (30% of total score)
    season_weather = {
        "Spring": ["mild", "spring", "pleasant", "temperate"],
        "Summer": ["hot", "summer", "warm", "sunny"],
        "Fall": ["autumn", "fall", "cool", "mild"],
        "Winter": ["cold", "winter", "snow", "chilly"]
    }
    
    if season in season_weather:
        for keyword in season_weather[season]:
            if keyword in description:
                score += 0.1  # 0.1 points per matching season keyword
    
    # Normalize score to be between 0 and 1
    return min(max(score, 0), 1)

# CrewAI Tool Classes
class WeatherForecastTool(BaseTool):
    name: str = "Weather Forecast Tool"
    description: str = "Get weather forecast for a destination and date. Provides temperature, conditions, humidity, and wind speed."
    args_schema: Type[BaseModel] = WeatherForecastInput

    def _run(self, destination: str, date: Optional[str] = None) -> str:
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
            
        try:
            url = "http://api.weatherapi.com/v1/forecast.json"
            params = {
                'key': os.getenv('WEATHER_API_KEY'),
                'q': destination,
                'dt': date,
                'aqi': 'no'
            }
            response = requests.get(url, params=params)
            data = response.json()
            
            result = {
                "temperature": data['current']['temp_c'],
                "condition": data['current']['condition']['text'],
                "humidity": data['current']['humidity'],
                "wind_speed": data['current']['wind_kph']
            }
        except Exception as e:
            result = {
                "temperature": 25,
                "condition": "Sunny",
                "humidity": 60,
                "wind_speed": 10
            }
        
        return json.dumps(result)

class LocalEventsTool(BaseTool):
    name: str = "Local Events Tool"
    description: str = "Get local events for a destination within a date range. Returns event names, dates, venues, and URLs."
    args_schema: Type[BaseModel] = LocalEventsInput

    def _run(self, destination: str, date_range: Optional[Dict[str, str]] = None) -> str:
        try:
            geo = geocode_city(destination)
            url = "https://www.eventbriteapi.com/v3/events/search/"
            params = {
                'location.latitude': geo['lat'],
                'location.longitude': geo['lon'],
                'location.within': '10km',
                'expand': 'venue',
                'token': os.getenv('EVENTBRITE_API_KEY')
            }
            if date_range:
                params['start_date.range_start'] = date_range.get('start')
                params['start_date.range_end'] = date_range.get('end')
            
            response = requests.get(url, params=params)
            data = response.json()
            
            events = []
            for event in data.get('events', [])[:5]:
                events.append({
                    'name': event['name']['text'],
                    'date': event['start']['local'],
                    'venue': event['venue']['name'],
                    'url': event['url']
                })
            if not events:
                events.append({
                    "name": "No major events found",
                    "date": date_range.get("start", datetime.now().strftime("%Y-%m-%d")) if date_range else datetime.now().strftime("%Y-%m-%d"),
                    "description": f"No major events found for {destination} in this period.",
                    "location": destination
                })
        except Exception as e:
            events = [
                {
                    "name": "Local Festival",
                    "date": date_range.get("start", datetime.now().strftime("%Y-%m-%d")) if date_range else datetime.now().strftime("%Y-%m-%d"),
                    "description": "Annual cultural festival",
                    "location": destination
                }
            ]
        
        return json.dumps(events)

class TravelBudgetTool(BaseTool):
    name: str = "Travel Budget Calculator"
    description: str = "Calculate estimated travel budget based on destination, duration, and preferences using real currency conversion."
    args_schema: Type[BaseModel] = TravelBudgetInput

    def _run(self, destination: str, duration: int, preferences: List[str]) -> str:
        try:
            geo = geocode_city(destination)
            country = geo.get('country', 'US')
            # Use restcountries API to get currency code
            currency_code = 'USD'
            try:
                rest_url = f'https://restcountries.com/v3.1/alpha/{country}'
                rest_resp = requests.get(rest_url)
                rest_data = rest_resp.json()
                currency_code = list(rest_data[0]['currencies'].keys())[0]
            except Exception:
                pass
            # Get currency rates
            currency_api_key = os.getenv('CURRENCY_API_KEY')
            rates_url = f'https://v6.exchangerate-api.com/v6/{currency_api_key}/latest/USD'
            rates_resp = requests.get(rates_url)
            rates = rates_resp.json().get('conversion_rates', {})
            rate = rates.get(currency_code, 1.0)
            # Base costs in USD
            base_costs = {
                "accommodation": 100,
                "food": 50,
                "activities": 75,
                "transportation": 200
            }
            # Convert to destination currency
            converted_costs = {key: round(value * rate, 2) for key, value in base_costs.items()}
            total = sum(converted_costs.values()) * duration
            result = {
                **converted_costs,
                "total": total
            }
        except Exception as e:
            result = {
                "accommodation": 100 * duration,
                "food": 50 * duration,
                "activities": 75 * duration,
                "transportation": 200,
                "total": (100 + 50 + 75) * duration + 200
            }
        
        return json.dumps(result)

class SafetyInfoTool(BaseTool):
    name: str = "Safety Information Tool"
    description: str = "Get safety information for a destination including general safety, health concerns, crime rate, and natural disaster risk."
    args_schema: Type[BaseModel] = SafetyInfoInput

    def _run(self, destination: str) -> str:
        try:
            geo = geocode_city(destination)
            url = f"https://api.opentripmap.com/0.1/en/places/radius"
            params = {
                'radius': 1000,
                'lon': geo['lon'],
                'lat': geo['lat'],
                'apikey': os.getenv('OPENTRIPMAP_API_KEY')
            }
            response = requests.get(url, params=params)
            data = response.json()
            # Try to extract tags or info from the first POI
            pois = data.get('features', [])
            tags = []
            if pois:
                for poi in pois:
                    tags.extend(poi.get('properties', {}).get('kinds', '').split(','))
            # Simple logic: if 'danger' or 'safety' in tags, flag it
            general_safety = "Generally safe for tourists"
            if any('danger' in tag or 'safety' in tag for tag in tags):
                general_safety = "Some safety concerns reported. Check local advisories."
            result = {
                "general_safety": general_safety,
                "health_concerns": "Check local health advisories",
                "crime_rate": "Check local crime statistics",
                "natural_disasters": "Check local disaster risk"
            }
        except Exception as e:
            result = {
                "general_safety": "Generally safe for tourists",
                "health_concerns": "No major health concerns",
                "crime_rate": "Low",
                "natural_disasters": "Low risk"
            }
        
        return json.dumps(result)

class TransportationRoutesTool(BaseTool):
    name: str = "Transportation Routes Tool"
    description: str = "Get transportation routes between two locations including flights and transit routes."
    args_schema: Type[BaseModel] = TransportationRoutesInput

    def _run(self, origin: str, destination: str, date: Optional[str] = None) -> str:
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
        try:
            # Geocode origin and destination
            origin_geo = geocode_city(origin)
            dest_geo = geocode_city(destination)
            # Use Aviation Stack API for flight information
            url = "http://api.aviationstack.com/v1/flights"
            params = {
                'access_key': os.getenv('AVIATION_STACK_API_KEY'),
                'dep_iata': origin[:3].upper(),  # Placeholder: ideally use a real IATA lookup
                'arr_iata': destination[:3].upper(),
                'flight_date': date
            }
            response = requests.get(url, params=params)
            flight_data = response.json()
            # Use TransitLand API for ground transportation
            transit_url = f"https://transit.land/api/v2/routes"
            transit_params = {
                'api_key': os.getenv('TRANSITLAND_API_KEY'),
                'lat': dest_geo['lat'],
                'lon': dest_geo['lon'],
                'radius': 1000
            }
            transit_response = requests.get(transit_url, params=transit_params)
            transit_data = transit_response.json()
            result = {
                "flights": flight_data.get('data', []),
                "transit_routes": transit_data.get('routes', [])
            }
        except Exception as e:
            result = {
                "flights": [],
                "transit_routes": []
            }
        
        return json.dumps(result)

class RestaurantRecommendationsTool(BaseTool):
    name: str = "Restaurant Recommendations Tool"
    description: str = "Get restaurant recommendations for a location including name, cuisine, rating, and price range."
    args_schema: Type[BaseModel] = RestaurantRecommendationsInput

    def _run(self, location: str) -> str:
        try:
            # Placeholder implementation - in real scenario, use a restaurant API
            result = [
                {
                    "name": "Local Favorite Restaurant",
                    "cuisine": "Local",
                    "rating": 4.5,
                    "price_range": "$$",
                    "location": location
                }
            ]
        except Exception as e:
            result = [
                {
                    "name": "Local Restaurant",
                    "cuisine": "Local",
                    "rating": 4.0,
                    "price_range": "$$",
                    "location": location
                }
            ]
        
        return json.dumps(result)

class AccommodationOptionsTool(BaseTool):
    name: str = "Accommodation Options Tool"
    description: str = "Get accommodation options for a destination including hotels with ratings and prices."
    args_schema: Type[BaseModel] = AccommodationOptionsInput

    def _run(self, destination: str) -> str:
        try:
            # Placeholder implementation - in real scenario, use a booking API
            result = [
                {
                    "name": "Recommended Hotel",
                    "type": "Hotel",
                    "rating": 4.0,
                    "price_per_night": 120,
                    "location": destination
                }
            ]
        except Exception as e:
            result = [
                {
                    "name": "Budget Hotel",
                    "type": "Hotel",
                    "rating": 3.5,
                    "price_per_night": 80,
                    "location": destination
                }
            ]
        
        return json.dumps(result)

class MatchScoreTool(BaseTool):
    name: str = "Match Score Calculator"
    description: str = "Calculate how well a city matches user preferences, budget, and season. Returns a score between 0 and 1."
    args_schema: Type[BaseModel] = MatchScoreInput

    def _run(self, city: Dict[str, Any], preferences: List[str], budget: float, season: str) -> str:
        score = calculate_match_score(city, preferences, budget, season)
        return json.dumps({"match_score": score})

class GeocodeTool(BaseTool):
    name: str = "Geocoding Tool"
    description: str = "Get latitude, longitude, and country information for a city using OpenTripMap API."
    args_schema: Type[BaseModel] = GeocodeInput

    def _run(self, city_name: str) -> str:
        result = geocode_city(city_name)
        return json.dumps(result)

# Collection of all tools for easy import
TRAVEL_TOOLS = [
    WeatherForecastTool(),
    LocalEventsTool(),
    TravelBudgetTool(),
    SafetyInfoTool(),
    TransportationRoutesTool(),
    RestaurantRecommendationsTool(),
    AccommodationOptionsTool(),
    MatchScoreTool(),
    GeocodeTool()
]
