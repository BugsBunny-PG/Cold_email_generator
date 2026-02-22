import streamlit as st 
from langchain_community.document_loaders import WebBaseLoader

from chain import chain
from portfolio import Portifolio
from utils import clean_text


def create_streamlit_app(llm, portfolio, clean_text):
    """
    Main Streamlit UI function.
    Handles:
    - URL input
    - Web scraping
    - Text cleaning
    - Job extraction using LLM
    - Portfolio link retrieval using ChromaDB
    - Email generation
    """
    
    st.title("Cold Email Generator")  # App title

    # Input field to accept a job posting URL
    url_input = st.text_input(
        "Enter a URL:",
        value="https://careers.nike.com/software-engineer-i/job/R-78284"
    )

    # Button to trigger the email generation process
    submit_button = st.button("Submit")

    if submit_button:
        try:
            # Load the webpage content
            loader = WebBaseLoader([url_input])

            # Extract raw text and clean it
            data = clean_text(loader.load().pop().page_content)

            # Load portfolio data into vectorstore (if not already loaded)
            portfolio.load_portfolio()

            # Extract all jobs from the scraped page text
            jobs = llm.extract_jobs(data)

            # Generate emails for each job found
            for job in jobs:
                # Extract skills from job posting (may be missing)
                skills = job.get('skills', [])

                # Query vectorstore to find relevant portfolio links
                links = portfolio.query_links(skills)

                # Generate cold email using the LLM
                email = llm.write_mail(job, links)

                # Display email result on screen
                st.code(email, language='markdown')

        except Exception as e:
            # Catch any error and display it to the user
            st.error(f"An Error Occurred: {e}")


print(">>> Running UPDATED main.py")  # Debug output for backend logs


if __name__ == "__main__":
    # Create chain (LLM pipeline)
    chain = chain()

    # Create portfolio handler (ChromaDB + CSV loader)
    portfolio = Portifolio()

    # Configure Streamlit page layout
    st.set_page_config(layout="wide", page_title="Cold Email Generator")

    # Launch Streamlit App
    create_streamlit_app(chain, portfolio, clean_text)