from textwrap import dedent
from typing import List, Dict, Any, Optional
from datetime import datetime

class TravelTasks:
    def plan_itinerary(self, agent, cities: List[str], date_range: Dict[str, str], 
                      interests: List[str], budget: Optional[float] = None) -> str:
        return dedent(f"""
                Create a detailed travel itinerary for the following trip:
                - Cities to visit: {', '.join(cities)}
                - Date range: {date_range['start']} to {date_range['end']}
                - Interests: {', '.join(interests)}
                - Budget: ${budget if budget else 'Not specified'}

                The itinerary should include:
                1. Day-by-day schedule
                2. Recommended activities based on interests
                3. Estimated costs for each activity
                4. Travel times between locations
                5. Local transportation options
                6. Restaurant recommendations
                7. Safety tips and local customs
                8. Weather-appropriate packing suggestions

                Make sure to:
                - Stay within budget if specified
                - Consider travel times between locations
                - Include a mix of activities based on interests
                - Account for local holidays or events
                - Consider weather conditions
                - Include emergency contact information
            """)

    def identify_city(self, agent, origin: str, cities: List[str], 
                     interests: List[str], date_range: Dict[str, str],
                     budget: Optional[float] = None) -> str:
        return dedent(f"""
                Analyze and recommend the best cities to visit based on:
                - Origin: {origin}
                - Potential cities: {', '.join(cities)}
                - Interests: {', '.join(interests)}
                - Date range: {date_range['start']} to {date_range['end']}
                - Budget: ${budget if budget else 'Not specified'}

                For each city, provide:
                1. Match score based on interests
                2. Estimated costs
                3. Weather during travel dates
                4. Local events during the stay
                5. Safety considerations
                6. Transportation options from origin
                7. Best time to visit
                8. Unique attractions

                Rank the cities based on:
                - Interest match
                - Budget compatibility
                - Weather conditions
                - Local events
                - Safety
                - Accessibility
            """)

    def gather_city_info(self, agent, cities: List[str], date_range: Dict[str, str],
                        interests: List[str]) -> str:
        return dedent(f"""
                Gather detailed information about each city:
                - Cities: {', '.join(cities)}
                - Date range: {date_range['start']} to {date_range['end']}
                - Interests: {', '.join(interests)}

                For each city, provide:
                1. Local customs and etiquette
                2. Popular neighborhoods
                3. Hidden gems and local favorites
                4. Food and dining recommendations
                5. Shopping areas
                6. Cultural events and festivals
                7. Public transportation system
                8. Safety tips for tourists
                9. Language considerations
                10. Local currency and payment methods
            """)

    def plan_transportation(self, agent, origin: str, cities: List[str],
                          date_range: Dict[str, str]) -> str:
        return dedent(f"""
                Plan transportation for the trip:
                - Origin: {origin}
                - Cities: {', '.join(cities)}
                - Date range: {date_range['start']} to {date_range['end']}

                Provide:
                1. Flight options between cities
                2. Ground transportation options
                3. Public transit information
                4. Estimated travel times
                5. Cost estimates
                6. Booking recommendations
                7. Transportation passes or cards
                8. Airport transfer options
                9. Local taxi/ride-sharing services
                10. Walking/biking routes
            """)

    def find_accommodation(self, agent, cities: List[str], date_range: Dict[str, str],
                         budget: Optional[float] = None) -> str:
        return dedent(f"""
                Find suitable accommodations for:
                - Cities: {', '.join(cities)}
                - Date range: {date_range['start']} to {date_range['end']}
                - Budget: ${budget if budget else 'Not specified'}

                For each city, provide:
                1. Hotel recommendations
                2. Alternative accommodation options
                3. Best neighborhoods to stay
                4. Price ranges
                5. Booking tips
                6. Amenities and facilities
                7. Location advantages
                8. Transportation access
                9. Safety considerations
                10. Special requirements options
            """)

    def create_budget(self, agent, cities: List[str], date_range: Dict[str, str],
                     interests: List[str], budget: Optional[float] = None) -> str:
        return dedent(f"""
                Create a detailed budget plan for:
                - Cities: {', '.join(cities)}
                - Date range: {date_range['start']} to {date_range['end']}
                - Interests: {', '.join(interests)}
                - Total budget: ${budget if budget else 'Not specified'}

                Provide:
                1. Daily budget breakdown
                2. Accommodation costs
                3. Transportation costs
                4. Food and dining budget
                5. Activity costs
                6. Shopping budget
                7. Emergency fund
                8. Currency exchange tips
                9. Payment methods
                10. Money-saving tips
            """) 
