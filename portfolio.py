import pandas as pd
import chromadb
import uuid


class Portifolio:
    """
    Handles loading a CSV portfolio, storing tech stacks in ChromaDB,
    and querying relevant portfolio links based on extracted job skills.
    """

    def __init__(self, file_path="/Users/pragyagupta/Documents/AI/projects/cold_emial_genrator/App/resource/my_portfolio.csv"):
        """
        Initialize the Portfolio class.
        
        - Loads CSV data containing tech stacks + links
        - Connects to persistent ChromaDB vector store
        - Creates/gets the 'portfolio' collection
        """
        self.file_path = file_path

        # Load CSV into DataFrame
        self.data = pd.read_csv(file_path)

        # Connect to the persistent ChromaDB client (saved on disk)
        self.chroma_client = chromadb.PersistentClient(path="vectorstore")

        # Create collection if not exists
        self.collection = self.chroma_client.get_or_create_collection(name="portfolio")


    def load_portfolio(self):
        """
        Loads the portfolio data into ChromaDB.
        
        Only loads if the collection is empty.
        Each row:
            - Techstack (string) → stored as document
            - Links → stored as metadata
            - Random UUID → unique ID for vectorstore
        """
        if self.collection.count() == 0:  # Load only once
            for _, row in self.data.iterrows():

                # Convert Techstack to string (avoid issues with floats/nan)
                techstack = str(row["Techstack"]).strip()

                # Add to vectorstore — MUST pass list for documents & ids
                self.collection.add(
                    documents=[techstack],
                    metadatas={"Links": row["Links"]},
                    ids=[str(uuid.uuid4())]
                )


    def query_links(self, skills):
        """
        Query vectorstore to find the best-matching portfolio links
        based on a list of extracted job skills.

        Input:
            skills → list of strings, e.g. ["python", "aws", "docker"]

        Returns:
            A list of metadata dictionaries containing 'Links'
        """
        # If skills list is empty → nothing to match, return empty
        if not skills:
            return []

        # Convert skills list into a single query string
        query_text = " ".join(skills)

        # Query ChromaDB for the top 2 most relevant results
        result = self.collection.query(
            query_texts=[query_text],
            n_results=2
        )

        # Return metadata array safely
        return result.get("metadatas", [])