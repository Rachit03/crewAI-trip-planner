# from typing import Type
# from pydantic import BaseModel, Field
# from crewai.tools import BaseTool
# import requests
# from bs4 import BeautifulSoup
# import json

# class SearchInput(BaseModel):
#     """Input for search tool."""
#     query: str = Field(..., description="The search query")

# class SearchInternetTool(BaseTool):
#     name: str = "search_internet"
#     description: str = "Search the internet for the given query and return top results."
#     args_schema: Type[BaseModel] = SearchInput

#     def _run(self, query: str) -> str:
#         """Perform the search logic."""
#         try:
#             headers = {
#                 'User-Agent': (
#                     'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
#                     'AppleWebKit/537.36 (KHTML, like Gecko) '
#                     'Chrome/91.0.4472.124 Safari/537.36'
#                 )
#             }
#             response = requests.get(f"https://www.google.com/search?q={query}", headers=headers)
#             soup = BeautifulSoup(response.text, 'html.parser')

#             # Extract simplified top 5 results
#             results = []
#             for result in soup.find_all('div', class_='g')[:5]:
#                 title = result.find('h3')
#                 if title:
#                     results.append(title.text)

#             return json.dumps(results)

#         except Exception as e:
#             return f"Error during search: {str(e)}"

############ Duck Duck Go


import requests
from bs4 import BeautifulSoup
import json
from typing import Type
from pydantic import BaseModel, Field
from crewai.tools import BaseTool
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class SearchInput(BaseModel):
    """Input for search tool."""
    query: str = Field(..., description="The search query")

class SearchInternetTool(BaseTool):
    name: str = "search_internet"
    description: str = "Search the internet for the given query and return top results."
    args_schema: Type[BaseModel] = SearchInput

    def _run(self, query: str) -> str:
        """Perform the search using DuckDuckGo HTML results."""
        try:
            headers = {"User-Agent": "Mozilla/5.0"}
            url = f"https://duckduckgo.com/html/?q={query}"
            response = requests.get(
                url, headers=headers, timeout=10, verify=False   # ✅ disable SSL verification
            )
            soup = BeautifulSoup(response.text, "html.parser")

            results = []
            for a in soup.select(".result__title a")[:5]:
                title = a.get_text(strip=True)
                if title:
                    results.append(title)

            return json.dumps(results, ensure_ascii=False)

        except Exception as e:
            return json.dumps([f"Error during search: {str(e)}"])

# import requests
# from bs4 import BeautifulSoup
# import json
# from typing import Type
# from pydantic import BaseModel, Field
# from crewai.tools import BaseTool
# import urllib3

# urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# class SearchInput(BaseModel):
#     query: str = Field(..., description="The search query")

# class SearchInternetTool(BaseTool):
#     name: str = "search_internet"
#     description: str = "Scrape Google Search for the given query and return top results (no API key)."
#     args_schema: Type[BaseModel] = SearchInput

#     def _run(self, query: str) -> str:
#         try:
#             headers = {
#                 "User-Agent": (
#                     "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
#                     "AppleWebKit/537.36 (KHTML, like Gecko) "
#                     "Chrome/91.0.4472.124 Safari/537.36"
#                 )
#             }
#             url = f"https://www.google.com/search?q={query.replace(' ', '+')}&hl=en"
#             response = requests.get(url, headers=headers, timeout=10, verify=False)
#             soup = BeautifulSoup(response.text, "html.parser")

#             results = []
#             # Look for h3 tags inside result blocks
#             for g in soup.select("div.tF2Cxc h3")[:5]:
#                 title = g.get_text(strip=True)
#                 if title:
#                     results.append(title)

#             if not results:
#                 return json.dumps(["No results found or blocked by Google"], ensure_ascii=False)

#             return json.dumps(results, ensure_ascii=False)

#         except Exception as e:
#             return json.dumps([f"Error during search: {str(e)}"])
