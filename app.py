from flask import Flask, render_template, request
import os
from dotenv import load_dotenv
import google.generativeai as genai
from newspaper import Article
import markdown
import logging
from concurrent.futures import ThreadPoolExecutor
from googleapiclient.discovery import build

# Configure logging to provide detailed feedback
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables from the .env file
load_dotenv()
try:
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    model = genai.GenerativeModel("gemini-2.5-flash")
    logging.info("Gemini API successfully configured.")
except Exception as e:
    logging.error(f"Failed to configure Gemini API. Please check your GEMINI_API_KEY in the .env file: {e}")
    model = None

app = Flask(__name__)

# ----------- Search & Scrape Functions -----------

def search_articles(topic, max_results=5):
    """
    Performs a Google search for a given topic using the Google Custom Search Engine (CSE) API.
    It returns a list of URLs for the most relevant articles found.
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    cse_id = os.getenv("GOOGLE_CSE_ID")
    
    if not api_key or not cse_id:
        logging.error("Google API key or CSE ID not found in .env file. Please configure them.")
        return []

    try:
        service = build("customsearch", "v1", developerKey=api_key)
        result = service.cse().list(
            q=topic,
            cx=cse_id,
            num=max_results,
        ).execute()
        
        links = [item['link'] for item in result.get('items', [])]
        logging.info(f"Found {len(links)} links from Google Search.")
        return links
    except Exception as e:
        logging.error(f"Google Search API call failed: {e}")
        return []

def fetch_article_content(url, max_chars=3000):
    """
    Downloads and extracts the main text content from a given article URL.
    It handles potential download and parsing errors gracefully.
    Returns the truncated content as a string, or None on failure.
    """
    try:
        article = Article(url)
        article.download()
        article.parse()
        content = article.text
        if not content:
            raise ValueError("Article text is empty.")
        logging.info(f"✅ Extracted content from: {url}")
        return content[:max_chars]  # Truncate content to a manageable size for the LLM
    except Exception as e:
        logging.error(f"❌ Failed to extract content from {url}: {e}")
        return None

def fetch_all_articles_parallel(links):
    """
    Uses a thread pool to fetch content from multiple links concurrently.
    This significantly speeds up the scraping process.
    Returns a list of extracted article contents.
    """
    articles = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        results = executor.map(fetch_article_content, links)
        for content in results:
            if content:
                articles.append(content)
    return articles

def generate_summary_with_model(topic, contents):
    """
    Generates a summary using the Gemini model. The prompt is carefully
    structured to ensure the summary is factual, concise, and based only
    on the provided articles.
    """
    if not model:
        return "The generative model is not available due to a configuration error."

    combined_text = "\n\n---\n\n".join(contents)
    
    if len(combined_text) > 20000:
        logging.warning("Combined text is too long, truncating further.")
        combined_text = combined_text[:20000]

    prompt = f"""
You are an expert research summarization bot. Your task is to synthesize information from multiple web articles related to the topic: **{topic}**.

Follow these strict rules:
1.  **Format:** Produce a summary in a clear, concise bullet-point list using Markdown.
2.  **Content:** The summary must contain only key findings, facts, trends, and statistics explicitly mentioned in the provided articles. Do not add any outside information or speculate.
3.  **Tone:** Maintain a neutral, factual, and professional tone.
4.  **Structure:** Each bullet point should start with a key insight, followed by one or two supporting sentences for context if necessary.

### Articles for Analysis:
{combined_text}
"""
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        logging.error(f"Gemini API call failed: {e}")
        return "An error occurred while generating the summary. Please try again."

# ----------- Flask Route -----------

@app.route("/", methods=["GET", "POST"])
def index():
    summary = None
    topic = ""
    links = []

    if request.method == "POST":
        topic = request.form.get("topic", "").strip()
        if not topic:
            summary = "Please enter a valid topic to summarize."
            return render_template("index.html", summary=summary)

        links = search_articles(topic)

        if not links:
            summary = "❌ Could not find any relevant articles for this topic. Please check your API keys or try a different topic."
            return render_template("index.html", summary=summary, topic=topic)

        contents = fetch_all_articles_parallel(links)

        if not contents:
            summary = "❌ Found articles, but failed to extract content from all of them."
            return render_template("index.html", summary=summary, topic=topic, links=links)

        summary_markdown = generate_summary_with_model(topic, contents)
        summary = markdown.markdown(summary_markdown, extensions=['fenced_code'])

    return render_template("index.html", summary=summary, topic=topic, links=links)

# ----------- Run App -----------

if __name__ == "__main__":
    print("🚀 Flask app running at http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=True)
