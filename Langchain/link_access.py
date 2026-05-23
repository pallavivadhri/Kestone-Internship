from langchain_community.document_loaders import WebBaseLoader
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin 

# Extract all links from given link
main_link = "https://www.morganstanley.com/"

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
               "clap", "contact-us", "new-story" ,"post_page","noredirect" , "locations"
]
filtered_links = [
    url for url in unique_links
    if url.startswith("https://www.morganstanley.com/") and  # Medium blog
       not any(                                    # Exclude these patterns
           excl in url for excl in exclude
       )
]


filtered2_links = [
    url for url in unique_links
    if not any(excl in url for excl in exclude)
]  

print("Extracted links:")
for i, link in enumerate(filtered_links):
    print(f"{i}: {link}")

# Ask user to choose a link
selected_index = int(input("\nEnter the index of the link you want to extract content from: "))
selected_url = filtered_links[selected_index]
print(f"\nFetching content from: {selected_url}")

# Load and print content from the selected link
loader = WebBaseLoader(selected_url)
docs = loader.load()
print(docs)  


