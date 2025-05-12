import os
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
       
       # Add context data if available
       if context_data:
           context_str = f"Here is the relevant NFL data:\n{str(context_data)}"
           messages.append({"role": "system", "content": context_str})

       try:
          async with httpx.AsyncClient(timeout=30.0) as client:
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
              
              resutlt = response.json()
              return resutlt['choices'][0]['message']['content']

       except Exception as e:
             print(f"Error generating response: {e}")
             return "Sorry, I couldn't process your request at the moment."  
       
llm_service = LLMService()