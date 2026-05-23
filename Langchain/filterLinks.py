
import re

def filter_links(links, patterns):
    # Compile the regex patterns for performance
    compiled_patterns = [re.compile(p) for p in patterns]
    
    # Filter links
    filtered_links = [
        link for link in links 
        if not any(pattern.search(link) for pattern in compiled_patterns)
    ]
    
    return filtered_links

# Example usage
links = [
    "https://example.com/mailbox",
    "https://mail.example.com/inbox",
    "https://example.com/contact",
    "https://example.com/about"
]

patterns = [
    r"mail.*",        # Matches "mail" followed by anything
    r"contact"        # Matches links with "contact"
    r"policy.*"
]

clean_links = filter_links(links, patterns)
print(clean_links)
