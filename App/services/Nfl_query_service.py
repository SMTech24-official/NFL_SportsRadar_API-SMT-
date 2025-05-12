from App.services.nfl_service import nfl_service
from App.services.LLm_service import llm_service
import re

class NFLQueryService:
    """
    Service to handle user queries related to NFL data
    """

    def __init__(self):
        self.nfl_service = nfl_service
        self.llm_service = llm_service

    async def process_query(self, query: str):
        """
        Process a natural language query about NFL data
        
        Args:
            query (str): The user's question about NFL data
            
        Returns:
            dict: Response containing the LLM's answer and relevant data
        """
        
        # Determine query type and fetch relevant data
        # Fix: Changed from __classify_query to _classify_query
        query_type, params = self._classify_query(query)
        context_data = await self._fetch_relevant_data(query_type, params)

        # Generate a LLM response with the context data
        llm_response = await self.llm_service.generate_response(query, context_data)

        return {
             "query": query,
             "answer": llm_response,
             "data_sources": self.get_data_sources(query_type)
        }
    
    # Fix: Changed from __classify_query to _classify_query
    def _classify_query(self, query: str):
        """
        Classify the user's query to determine the type of data needed
        
        Args:
            query (str): The user's question about NFL data
            
        Returns:
            tuple: A tuple containing the query type and parameters
        """
        query = query.lower()
        params = {}
        
        # Basic classification patterns
        if any(term in query for term in ["ranking", "rank", "best", "top", "projections"]):
            return "player_rankings", params
            
        if any(term in query for term in ["matchup", "vs", "versus", "against", "playing"]):
            # Extract team names
            return "matchups", params
            
        if any(term in query for term in ["injury", "injured", "hurt"]):
            # Try to extract year, season, week if mentioned
            year_match = re.search(r'20\d{2}', query)
            if year_match:
                params["year"] = year_match.group(0)
            return "injuries", params
            
        if any(term in query for term in ["schedule", "games", "playing"]):
            # Try to extract team
            return "schedule", params
            
        if any(term in query for term in ["depth chart", "roster", "lineup"]):
            # Try to extract team
            return "depth_chart", params
            
        # Default to general query
        return "general", params

    # Fix: Changed from __fetch_relevant_data to _fetch_relevant_data
    async def _fetch_relevant_data(self, query_type, params):
        """
        Fetch the relevant NFL data based on query type
        """
        try:
            if query_type == "player_rankings":
                # For player rankings, we could get team profiles with player stats
                return await self.nfl_service.get_teams()
                
            elif query_type == "matchups":
                # For matchups, we might want the current schedule
                return await self.nfl_service.get_schedule(2023, "REG")
                
            elif query_type == "injuries":
                year = params.get("year", "2023")
                return await self.nfl_service.get_weekly_injuries(year, "REG", "1")
                
            elif query_type == "schedule":
                return await self.nfl_service.get_schedule(2023, "REG")
                
            elif query_type == "depth_chart":
                # Get team hierarchy first, could refine this to get specific team
                return await self.nfl_service.get_teams()
                
            else:  # General query
                # For general queries, provide teams hierarchy as a starting point
                return await self.nfl_service.get_teams()
            
        except Exception as e:
            print(f"Error fetching relevant data: {e}")
            return {"error": str(e)}
        
    def get_data_sources(self, query_type):
        """
        Return information about data sources used
        """
        data_sources = {
            "player_rankings": ["NFL team and player data"],
            "matchups": ["NFL schedule data"],
            "injuries": ["NFL injury reports"],
            "schedule": ["NFL team schedules"],
            "depth_chart": ["NFL team rosters"],
            "general": ["NFL general data"]
        }
        return data_sources.get(query_type, ["NFL API data"])

nfl_query_service = NFLQueryService()