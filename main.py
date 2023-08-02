from data_extraction import DataExtraction
from langchain_processing import LangChainProcessing
from email_processing import EmailProcessing
import pinecone
import os

def main():
    # Initialize Pinecone
    pinecone.init(api_key=os.getenv("PINECONE_API_KEY"), environment=os.getenv("PINECONE_ENV"))

    # Extract data
    data_extractor = DataExtraction()
    df, df_meta = data_extractor.extract_data('AMAZON_FASHION.json.gz', 'meta_AMAZON_FASHION.json.gz')

    # Process LangChain
    langchain_processor = LangChainProcessing()
    texts = df['reviewText'].tolist()
    metadata = [dict(rating=i) for i in df['overall'].tolist()]

# Check if you want to add review embeddings (set to False if you don't want to push to Pinecone)
    add_review_embeddings = False
    if add_review_embeddings:
        vstore = langchain_processor.add_review_embeddings(texts, metadata)
    else:
    # Retrieve the existing vector store from Pinecone (replace with the correct retrieval code)
        vstore = langchain_processor.get_existing_vector_store(index_name='voice-test')

    docs = langchain_processor.similarity_search(vstore, "The Powerstep Pinnacle Shoe Insoles are fantastic", 5.0)
    print(docs)

    summary, fb_copy = langchain_processor.write_summary_and_ad_copy(docs)

    # Process Emails
    email_processor = EmailProcessing()
    email_processor.send_emails(df, summary)

    # Deinitialize Pinecone if needed
    pinecone.deinit()

    print("Processing completed successfully!")

if __name__ == "__main__":
    main()
