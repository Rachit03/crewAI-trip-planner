from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field, validator
from datetime import datetime, timedelta
import re
import json
class InputGuardrails:
    @staticmethod
    def validate_budget(budget: float) -> bool:
        """Validate that the budget is within reasonable limits"""
        print("duration",budget, type(budget))
        return 50 <= budget <= 500

    @staticmethod
    def validate_duration(duration: int) -> bool:
        """Validate that the duration is within reasonable limits"""
        print("duration",duration)
        return 1 <= duration <= 15

    @staticmethod
    def validate_dates(start_date: str, end_date: str) -> bool:
        """Validate that the dates are valid and in the future"""
        try:
            start = datetime.strptime(start_date, "%Y-%m-%d")
            end = datetime.strptime(end_date, "%Y-%m-%d")
            today = datetime.now().date()
            
            return (
                start.date() >= today and
                end.date() >= start.date() and
                (end.date() - start.date()).days <= 90
            )
        except ValueError:
            return False

    @staticmethod
    def validate_preferences(preferences: List[str]) -> bool:
        """Validate that the preferences are from the allowed list"""
        allowed_preferences = {
            "Beach", "Mountains", "City Life", "Culture", "Food",
            "Adventure", "Relaxation", "Nightlife"
        }
        return all(pref in allowed_preferences for pref in preferences)

class OutputGuardrails:
    @staticmethod
    def validate_city_recommendation(city: Dict[str, Any]) -> bool:
        """Validate that a city recommendation has all required fields"""
        required_fields = {
            "name", "country", "description", "match_score",
            "highlights", "estimated_cost"
        }
        return all(field in city for field in required_fields)

    @staticmethod
    def validate_travel_plan(plan: Dict[str, Any]) -> bool:
        """Validate that a travel plan has all required fields"""
        required_fields = {"itinerary", "budget_breakdown", "recommendations"}
        if not all(field in plan for field in required_fields):
            return False
            
        # Validate itinerary
        for day in plan["itinerary"]:
            if not all(field in day for field in ["activities", "meals"]):
                return False
                
        # Validate budget breakdown
        required_costs = {"accommodation", "food", "activities", "transportation", "total"}
        if not all(cost in plan["budget_breakdown"] for cost in required_costs):
            return False
            
        return True

    @staticmethod
    def sanitize_description(text: str) -> str:
        """Remove potentially harmful content from descriptions"""
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        # Remove special characters
        text = re.sub(r'[^\w\s.,!?-]', '', text)
        return text.strip()

class SafetyGuardrails:
    @staticmethod
    def check_sensitive_content(text: str) -> bool:
        """Check for potentially sensitive or inappropriate content"""
        sensitive_keywords = {
            "illegal", "drugs", "weapons", "explicit",
            "offensive", "discriminatory"
        }
        text_lower = text.lower()
        return not any(keyword in text_lower for keyword in sensitive_keywords)

    @staticmethod
    def validate_location_safety(location: str) -> bool:
        """Validate that the location is safe for travel"""
        # This would typically integrate with a travel advisory API
        # For now, we'll use a simple placeholder
        return True

class BusinessGuardrails:
    @staticmethod
    def validate_budget_constraints(budget: float, costs: Dict[str, float]) -> bool:
        """Validate that the costs are within budget constraints"""
        total_cost = sum(costs.values())
        return total_cost <= budget * 1.1  # Allow 10% buffer

    @staticmethod
    def validate_time_constraints(activities: List[Dict[str, Any]]) -> bool:
        """Validate that activities fit within time constraints"""
        # This would typically check for scheduling conflicts
        # For now, we'll use a simple placeholder
        return True

class GuardrailManager:
    """Main class to manage all guardrails"""
    
    def __init__(self):
        self.input_guardrails = InputGuardrails()
        self.output_guardrails = OutputGuardrails()
        self.safety_guardrails = SafetyGuardrails()
        self.business_guardrails = BusinessGuardrails()

    def validate_input(self, input_data: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        """Validate all input data"""
        if not self.input_guardrails.validate_budget(input_data.get("budget", 0)):
            return False, "Budget must be between $50 and $10,000"
            
        if not self.input_guardrails.validate_duration(input_data.get("duration", 0)):
            return False, "Duration must be between 1 and 90 days"
            
        if not self.input_guardrails.validate_preferences(input_data.get("preferences", [])):
            return False, "Invalid preferences selected"
            
        if "start_date" in input_data and "end_date" in input_data:
            if not self.input_guardrails.validate_dates(
                input_data["start_date"],
                input_data["end_date"]
            ):
                return False, "Invalid date range"
                
        return True, None

    def validate_output(self, output_data: Dict[str, Any], output_type: str) -> tuple[bool, Optional[str]]:
        """Validate all output data"""
        if output_type == "city_recommendation":
            if not self.output_guardrails.validate_city_recommendation(output_data):
                return False, "Invalid city recommendation format"
                
        elif output_type == "travel_plan":
            if not self.output_guardrails.validate_travel_plan(output_data):
                return False, "Invalid travel plan format"
                
        # Sanitize descriptions
        if "description" in output_data:
            output_data["description"] = self.output_guardrails.sanitize_description(
                output_data["description"]
            )
            
        # Check for sensitive content
        if not self.safety_guardrails.check_sensitive_content(
            json.dumps(output_data)
        ):
            return False, "Output contains sensitive content"
            
        return True, None

    def validate_business_rules(self, data: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        """Validate all business rules"""
        if "budget" in data and "costs" in data:
            if not self.business_guardrails.validate_budget_constraints(
                data["budget"],
                data["costs"]
            ):
                return False, "Costs exceed budget constraints"
                
        if "activities" in data:
            if not self.business_guardrails.validate_time_constraints(
                data["activities"]
            ):
                return False, "Activities exceed time constraints"
                
        return True, None 
