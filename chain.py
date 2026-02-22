import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
from dotenv import load_dotenv  # Used to load API key from .env file


# Load environment variables from the .env file
load_dotenv(dotenv_path="/Users/pragyagupta/Documents/AI/projects/cold_emial_genrator/App/.env")

# Access the GROQ API key (used below)
os.getenv("GROQ_API_KEY")


class chain:
    """
    This class handles all interactions with the LLM (Groq Llama 3.3 model).
    It performs two main tasks:
        1. Extract job details from scraped text.
        2. Write cold email messages based on job details + portfolio links.
    """

    def __init__(self):
        """
        Initialize Groq LLM with:
        - Temperature 0 (deterministic output)
        - Model: Llama 3.3 70B
        - API key loaded from .env
        """
        self.llm = ChatGroq(
            temperature=0,
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model_name="llama-3.3-70b-versatile"
        )


    def extract_jobs(self, cleaned_text):
        """
        Takes the cleaned webpage text and extracts job details using LLM.

        Steps:
            1. Create a structured prompt instructing the LLM to extract jobs.
            2. Run LLM with prompt.
            3. Parse JSON output safely.
            4. Return list of job entries.

        Returns:
            A list of dictionaries → [{"role": "...", "experience": "..."}]
        """

        # Create prompt template for extracting job postings
        prompt_extract = PromptTemplate.from_template(
            """
            ### SCRAPED TEXT FROM WEBSITE:
            {page_data}

            ### Instructions:
            The scraped text is from the Careers page of a website.
            Your job is to extract job postings and return them as VALID JSON
            with the following keys: 'role', 'experience'.

            Only return valid JSON. No preamble or extra text.

            ### Valid JSON (NO PREAMBLE):
            """
        )

        # Combine prompt and LLM into a single chain
        chain_extract = prompt_extract | self.llm

        # Invoke chain with cleaned page data
        response = chain_extract.invoke(input={"page_data": cleaned_text})

        # Parse JSON output safely
        try:
            json_parser = JsonOutputParser()
            response = json_parser.parse(response.content)
        except OutputParserException:
            # If LLM output can't be parsed, raise a helpful error
            raise OutputParserException("Context too big. Unable to parse job JSON.")

        # Always return a list (even when LLM returns single object)
        return response if isinstance(response, list) else [response]


    def write_mail(self, job, links):
        """
        Generates a cold email using:
            - job description (dict converted to string)
            - relevant portfolio links from vector store

        Steps:
            1. Build cold-email writing prompt.
            2. Inject job + links.
            3. Invoke LLM to generate the email.
        
        Returns:
            A plain text cold email string.
        """

        # Prompt for generating cold email
        prompt_email = PromptTemplate.from_template(
            """
            ### JOB DESCRIPTION:
            {job_description}

            ### INSTRUCTION:
            You are Sally, a Business Development Executive at Zscaler.
            Zscaler is a leading Network Security & Software Consulting company.

            You have helped many clients with:
            - scalability
            - process optimization
            - cost reduction
            - improved operational efficiency

            Your task is to write a cold email to the client based on the job above,
            explaining how Zscaler can meet their requirements.

            Also, include the MOST relevant portfolio links:
            {link_list}

            Remember:
            - You are Sally (BDE at Zscaler)
            - NO PREAMBLE, only email body

            ### Email (NO PREAMBLE):
            """
        )

        # Combine prompt & LLM
        chain_email = prompt_email | self.llm

        # Generate email
        response = chain_email.invoke({
            "job_description": str(job),  # Insert job details
            "link_list": links            # Insert relevant portfolio links
        })

        # Return the generated email text
        return response.content


# Debug block to print API key when running this file directly
if __name__ == "__main__":
    print(os.getenv("GROQ_API_KEY"))
    print("Hello")