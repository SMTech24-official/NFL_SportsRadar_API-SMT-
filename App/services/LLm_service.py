import os
import json
import httpx
from typing import Dict, List, Any
from App.core.config import settings

class LLMService:
    def __init__(self):
        self.api_key = settings.GROQ_API_KEY
        self.base_url = "https://api.groq.com/openai/v1/chat/completions"
        self.model = "llama3-70b-8192"

    async def generate_response(self, query: str, context_data: Dict[str, Any] = None) -> str:
        """
        Generate a response using Groq's LLM based on the user query and NFL data context
        
        Args:
            query (str): The user's query about NFL data
            context_data (dict): NFL data to provide as context to the LLM
            
        Returns:
            str: The LLM's response
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        # Preparing the system messages for reply
        system_message = (
            "You are an NFL analytics expert providing insights based on official NFL data. "
            "Focus on providing accurate, data-driven analysis in a conversational tone. "
            "When responding:\n"
            "1. Summarize key information from the data\n"
            "2. Provide relevant statistics\n"
            "3. Offer context about teams, players, or matchups\n"
            "4. Cite your sources as 'Based on official NFL data.'"
        )
        
        messages = [{"role": "system", "content": system_message}]
        
        # Process context data if available - with size limitation
        if context_data:
            # Summarize the data to avoid 413 errors
            summarized_data = self._summarize_context_data(context_data)
            
            # Format and add the summarized context data
            context_str = f"Here is the relevant NFL data:\n{json.dumps(summarized_data, indent=2)}"
            # Limit context string to avoid payload too large
            if len(context_str) > 20000:  # Maximum reasonable size
                context_str = context_str[:20000] + "...[additional data truncated for size]"
                
            messages.append({"role": "system", "content": context_str})
            print(f"Context data size after summary: {len(context_str)} characters")

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    self.base_url,
                    headers=headers,
                    json={
                        "model": self.model,
                        "messages": messages + [{"role": "user", "content": query}],
                        "temperature": 0.7,
                        "max_tokens": 512,
                    },
                )
                response.raise_for_status()
                
                result = response.json()
                return result['choices'][0]['message']['content']

        except Exception as e:
            print(f"Error generating response: {e}")
            return f"Sorry, I couldn't process your request at the moment. Error: {str(e)}"

    def _summarize_context_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Summarize the context data to a reasonable size for the LLM API
        """
        data_type = data.get("data_type", "unknown")
        
        try:
            if "teams" in data and isinstance(data["teams"], list):
                return self._summarize_teams_data(data)
            elif "conferences" in data and isinstance(data["conferences"], list):
                return self._summarize_league_structure(data)
            elif "schedule" in data and isinstance(data["schedule"], dict):
                return self._summarize_schedule_data(data)
            elif "week" in data and "injuries" in data:
                return self._summarize_injury_data(data)
            else:
                # Generic summarization - extract only crucial fields
                return self._create_generic_summary(data)
        except Exception as e:
            print(f"Error during data summarization: {e}")
            return {"summary": "Data available but could not be summarized due to an error",
                    "error": str(e)}

    def _summarize_teams_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize teams data to essential information"""
        summarized = {"teams": []}
        
        try:
            for team in data.get("teams", [])[:20]:  # Limit to first 20 teams
                summarized["teams"].append({
                    "id": team.get("id", ""),
                    "name": team.get("name", ""),
                    "market": team.get("market", ""),
                    "alias": team.get("alias", ""),
                    "conference": team.get("conference", ""),
                    "division": team.get("division", "")
                })
            return summarized
        except Exception as e:
            print(f"Error summarizing teams data: {e}")
            return {"summary": "Teams data available but could not be summarized"}

    def _summarize_league_structure(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract a simplified version of the league structure"""
        summarized = {"conferences": []}
        
        try:
            for conference in data.get("conferences", []):
                conf_summary = {
                    "name": conference.get("name", ""),
                    "alias": conference.get("alias", ""),
                    "divisions": []
                }
                
                for division in conference.get("divisions", []):
                    div_summary = {
                        "name": division.get("name", ""),
                        "alias": division.get("alias", ""),
                        "teams": []
                    }
                    
                    for team in division.get("teams", []):
                        div_summary["teams"].append({
                            "name": team.get("name", ""),
                            "market": team.get("market", ""),
                            "alias": team.get("alias", "")
                        })
                    
                    conf_summary["divisions"].append(div_summary)
                
                summarized["conferences"].append(conf_summary)
            
            return summarized
        except Exception as e:
            print(f"Error summarizing league structure: {e}")
            return {"summary": "League structure data available but could not be summarized"}

    def _summarize_schedule_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize schedule data to essential games info"""
        summarized = {
            "year": data.get("year", ""),
            "type": data.get("type", ""),
            "games": []
        }
        
        try:
            # Take only the first 10 games to limit size
            games = data.get("games", [])[:10]
            for game in games:
                game_summary = {
                    "id": game.get("id", ""),
                    "status": game.get("status", ""),
                    "scheduled": game.get("scheduled", ""),
                    "home_team": {
                        "name": game.get("home", {}).get("name", ""),
                        "alias": game.get("home", {}).get("alias", "")
                    },
                    "away_team": {
                        "name": game.get("away", {}).get("name", ""),
                        "alias": game.get("away", {}).get("alias", "")
                    }
                }
                summarized["games"].append(game_summary)
            
            return summarized
        except Exception as e:
            print(f"Error summarizing schedule data: {e}")
            return {"summary": "Schedule data available but could not be summarized"}

    def _summarize_injury_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize injury report data"""
        summarized = {
            "week": data.get("week", ""),
            "teams_with_injuries": []
        }
        
        try:
            teams = data.get("teams", [])[:10]  # Limit to 10 teams
            for team in teams:
                team_summary = {
                    "name": team.get("name", ""),
                    "alias": team.get("alias", ""),
                    "injuries": []
                }
                
                # Limit to 10 players per team
                players = team.get("players", [])[:10]
                for player in players:
                    player_summary = {
                        "name": player.get("name", ""),
                        "position": player.get("position", ""),
                        "status": player.get("status", ""),
                        "injury": player.get("injury", "")
                    }
                    team_summary["injuries"].append(player_summary)
                
                summarized["teams_with_injuries"].append(team_summary)
            
            return summarized
        except Exception as e:
            print(f"Error summarizing injury data: {e}")
            return {"summary": "Injury data available but could not be summarized"}

    def _create_generic_summary(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a generic summary for unrecognized data formats"""
        summary = {"data_summary": "NFL data available"}
        
        # Try to extract some useful information
        if isinstance(data, dict):
            # Extract top-level keys and some values
            keys = list(data.keys())[:10]  # First 10 keys
            summary["available_data"] = keys
            
            # If there are lists, report their sizes
            for key in keys:
                if isinstance(data[key], list):
                    summary[f"{key}_count"] = len(data[key])
                    # Sample a few items if they're dictionaries
                    if data[key] and isinstance(data[key][0], dict):
                        sample_keys = list(data[key][0].keys())[:5]
                        summary[f"{key}_contains"] = sample_keys
        
        return summary

llm_service = LLMService()