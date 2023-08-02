from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.vectorstores import Pinecone
import pinecone
import os

class LangChainProcessing:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings()
        self.chat = ChatOpenAI(model_name="gpt-3.5-turbo-16k", temperature=0.2, verbose=True) # Correct initialization
        pinecone.init(api_key=os.getenv("PINECONE_API_KEY"), environment=os.getenv("PINECONE_ENV"))
        

    def add_review_embeddings(self, texts, metadata, index_name='voice-test'):
        vstore = Pinecone.from_texts(texts, self.embeddings, index_name=index_name, metadatas=metadata)
        return vstore

    def similarity_search(self, vstore, query, filter_value):
        docs = vstore.similarity_search(query, 3, filter={"rating": filter_value})
        print(docs)
        return docs
    
    def get_existing_vector_store(self, index_name):
        return Pinecone.from_existing_index(embedding=self.embeddings, index_name=index_name)

    def write_summary_and_ad_copy(self, docs):
        # Write summary of reviews
        prompt_template_summary = """
        Write a summary of the reviews:
        {text}
        The summary should be about ten lines long
        """
        PROMPT = PromptTemplate(template=prompt_template_summary, input_variables=["text"])
        chain = load_summarize_chain(self.chat, chain_type="stuff", prompt=PROMPT, verbose=True)
        summary = chain.run(docs)

        # Write ad copy for Facebook ad
        prompt_template_fb = """
        Write the copy for a facebook ad based on the reviews:
        {text}
        As far as text goes, you can have up to 40 characters in your headline, 
        125 characters in your primary text, and 30 characters in your description
        """
        PROMPT = PromptTemplate(template=prompt_template_fb, input_variables=["text"])
        chain = load_summarize_chain(self.chat, chain_type="stuff", prompt=PROMPT)
        fb_copy = chain.run(docs)

        return summary, fb_copy
