from langchain_community.document_loaders import WebBaseLoader
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin 

main_link = "https://www.jpmorganchase.com/"

response = requests.get(main_link)
soup = BeautifulSoup(response.text, 'html.parser')

links = []
for a_tag in soup.find_all('a', href=True):
    absolute_url = urljoin(main_link, a_tag['href'])
    links.append(absolute_url)

unique_links = list(set(links))
exclude = [
               "signin", "register", "subscribe", "bookmark", "vote",
               "followers", "following", "jobs", "policy", "terms",
               "about", "help", "statuspage", "tag", "search", "sitemap",
               "privacy", "contact-us", "new-story" ,"post_page", "noredirect" , "locations"
]
filtered_links = [
    url for url in unique_links
    if url.startswith("https://www.jpmorganchase.com/") and  
       not any(                                   
           excl in url for excl in exclude
       )
]

for link in filtered_links:
    loader = WebBaseLoader(link)
    docs = loader.load()
    print(docs)
    print()  

