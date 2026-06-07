from langchain_community.document_loaders import WebBaseLoader
from bs4 import BeautifulSoup
import requests

def get_website_title(url):
    """Fetch the title of the website for better metadata."""
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        title= soup.title
        if title: 
            return title.text.strip()
        
        return "Untitled Page"
    
    except Exception:
        return "Unknown Title"
    
def load_website(url):
    """Load website content and attach the metadata."""
    try:
        loader = WebBaseLoader(url)
        docs = loader.load()
        page_title = get_website_title(url)
        
        for doc in docs:
            doc.metadata["title"] = page_title
            doc.metadata["source_url"] = url
        return docs
    except Exception as e:
        raise Exception(f"Failed to load website content: {str(e)}")