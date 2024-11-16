import os
import pandas as pd
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import faiss
import argparse
import sys

# Configuration
DATA_DIR = '/mnt/fstore/ronit/Paragraph_with_Id/data'
EMBEDDINGS_DIR = '/mnt/fstore/ronit/Paragraph_with_Id/embeddings'
FAISS_DIR = '/mnt/fstore/ronit/Paragraph_with_Id/faiss'
METADATA_DIR = '/mnt/fstore/ronit/Paragraph_with_Id/metadata'

INPUT_CSV = os.path.join(DATA_DIR, '/mnt/fstore/ronit/Paragraph_with_Id/data/paragraphs_split_all.csv')  # Path to your input CSV
EMBEDDINGS_PKL = os.path.join(EMBEDDINGS_DIR, '/mnt/fstore/ronit/Paragraph_with_Id/embeddings/embeddings.pkl')  # Path to save/load embeddings
FAISS_INDEX_PATH = os.path.join(FAISS_DIR, '/mnt/fstore/ronit/Paragraph_with_Id/faiss/faiss_index.idx')  # Path to save/load FAISS index
METADATA_PKL = os.path.join(METADATA_DIR, '/mnt/fstore/ronit/Paragraph_with_Id/metadata/metadata.pkl')  # Path to save/load metadata

TEXT_COLUMN = 'paragraph_text'  # Column name containing the text to embed
MODEL_NAME = 'all-MiniLM-L6-v2'  # Pre-trained SentenceTransformer model
INDEX_TYPE = 'IndexFlatL2'  # Type of FAISS index: 'IndexFlatL2', 'IVFFlat', 'HNSW', etc.
TOP_K = 100  # Number of similar vectors to retrieve during search

def generate_embeddings(input_csv, text_column, model_name):
    """
    Generates embeddings for the specified text column in a CSV and saves them as a pickle file.

    Args:
        input_csv (str): Path to the input CSV file.
        text_column (str): Column name containing text data.
        model_name (str): Pre-trained SentenceTransformer model name.

    Returns:
        np.ndarray: Numpy array of embeddings.
    """
    # Load the data
    print("Loading CSV file...")
    df = pd.read_csv(input_csv)

    # Check if the TEXT_COLUMN exists
    if text_column not in df.columns:
        print(f"Error: Column '{text_column}' not found in the CSV file.")
        sys.exit(1)

    # Load the pre-trained model
    print(f"Loading the SentenceTransformer model '{model_name}'...")
    model = SentenceTransformer(model_name)

    # Generate embeddings
    print(f"Generating embeddings for column: '{text_column}'")
    embeddings = []
    for text in tqdm(df[text_column].astype(str), desc=f"Embedding '{text_column}'"):
        embedding = model.encode(text, show_progress_bar=False)
        embeddings.append(embedding)
    embeddings = np.array(embeddings)

    return embeddings

def save_embeddings(embeddings, embeddings_pkl):
    """
    Saves embeddings to a pickle file.

    Args:
        embeddings (np.ndarray): Numpy array of embeddings.
        embeddings_pkl (str): Path to save the pickle file.
    """
    print(f"Saving embeddings to {embeddings_pkl}...")
    with open(embeddings_pkl, 'wb') as f:
        pickle.dump(embeddings, f)
    print("Embeddings successfully saved!")

def load_embeddings(embeddings_pkl):
    """
    Loads embeddings from a pickle file.

    Args:
        embeddings_pkl (str): Path to the pickle file.

    Returns:
        np.ndarray: Numpy array of embeddings.
    """
    print(f"Loading embeddings from {embeddings_pkl}...")
    with open(embeddings_pkl, 'rb') as f:
        embeddings = pickle.load(f)
    print(f"Loaded embeddings with shape: {embeddings.shape}")
    return embeddings

def save_metadata(metadata, metadata_pkl):
    """
    Saves metadata to a pickle file.

    Args:
        metadata (list): List of metadata dictionaries.
        metadata_pkl (str): Path to save the pickle file.
    """
    print(f"Saving metadata to {metadata_pkl}...")
    with open(metadata_pkl, 'wb') as f:
        pickle.dump(metadata, f)
    print("Metadata successfully saved!")

def load_metadata(metadata_pkl):
    """
    Loads metadata from a pickle file.

    Args:
        metadata_pkl (str): Path to the pickle file.

    Returns:
        list: List of metadata dictionaries.
    """
    print(f"Loading metadata from {metadata_pkl}...")
    with open(metadata_pkl, 'rb') as f:
        metadata = pickle.load(f)
    print(f"Loaded metadata with {len(metadata)} records.")
    return metadata

def create_faiss_index(embeddings, index_type='IndexFlatL2'):
    """
    Creates a FAISS index from embeddings.

    Args:
        embeddings (np.ndarray): Numpy array of shape (num_vectors, dimension).
        index_type (str): Type of FAISS index to create.

    Returns:
        faiss.Index: The created FAISS index.
    """
    dimension = embeddings.shape[1]
    print(f"Creating FAISS Index of type '{index_type}' with dimension: {dimension}")

    if index_type == 'IndexFlatL2':
        index = faiss.IndexFlatL2(dimension)
    elif index_type == 'IVFFlat':
        nlist = 100  # Number of clusters
        quantizer = faiss.IndexFlatL2(dimension)
        index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_L2)
        print("Training IVFFlat index...")
        index.train(embeddings)
    elif index_type == 'HNSW':
        index = faiss.IndexHNSWFlat(dimension, 32)  # 32 is the number of neighbors
    else:
        print(f"Error: Unsupported index type '{index_type}'.")
        sys.exit(1)

    print("Adding embeddings to the FAISS index...")
    index.add(embeddings)
    print(f"Total vectors in the index: {index.ntotal}")

    return index

def save_faiss_index(index, index_path):
    """
    Saves a FAISS index to disk.

    Args:
        index (faiss.Index): The FAISS index to save.
        index_path (str): Path to save the index.
    """
    print(f"Saving FAISS index to {index_path}...")
    faiss.write_index(index, index_path)
    print("FAISS index successfully saved!")

def load_faiss_index(index_path):
    """
    Loads a FAISS index from disk.

    Args:
        index_path (str): Path to the FAISS index file.

    Returns:
        faiss.Index: The loaded FAISS index.
    """
    print(f"Loading FAISS index from {index_path}...")
    index = faiss.read_index(index_path)
    print(f"FAISS index loaded. Total vectors: {index.ntotal}")
    return index

def search_similar(index, query_embedding, top_k=100):
    """
    Searches for the top_k most similar vectors in the index to the query_embedding.

    Args:
        index (faiss.Index): The FAISS index to search.
        query_embedding (np.ndarray): Numpy array of shape (dimension,) representing the query.
        top_k (int): Number of top similar vectors to retrieve.

    Returns:
        distances (np.ndarray): Distances of the top_k similar vectors.
        indices (np.ndarray): Indices of the top_k similar vectors in the index.
    """
    if query_embedding.ndim == 1:
        query_embedding = query_embedding.reshape(1, -1)

    distances, indices = index.search(query_embedding, top_k)
    return distances, indices

def query_faiss(index, metadata, query_text, model_name, top_k=100):
    """
    Queries the FAISS index with a given text and retrieves the top_k similar records.

    Args:
        index (faiss.Index): The FAISS index to query.
        metadata (list): List of metadata mappings.
        query_text (str): The text to query.
        model_name (str): Pre-trained SentenceTransformer model name.
        top_k (int): Number of similar records to retrieve.

    Returns:
        list: List of top_k similar records with their metadata.
    """
    # Load the model
    model = SentenceTransformer(model_name)
    # Encode the query text
    query_embedding = model.encode(query_text, show_progress_bar=False)
    # Search in FAISS
    distances, indices_found = search_similar(index, query_embedding, top_k)
    # Retrieve metadata for each index
    results = []
    for dist, idx in zip(distances[0], indices_found[0]):
        record = metadata[idx]
        result = {
            'article_id': record.get('article_id', 'N/A'),
            'paragraph_id': record.get('paragraph_id', 'N/A'),
            'paragraph_text': record.get('paragraph_text', 'N/A'),
            'internal_links': record.get('internal_links', 'N/A'),
            'distance': dist
        }
        results.append(result)
    return results

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Query FAISS Index for Similar Documents")
    parser.add_argument('--query', type=str, help='Query text to search for similar documents.')
    parser.add_argument('--interactive', action='store_true', help='Run in interactive mode.')
    args = parser.parse_args()

    # Ensure output directories exist
    os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
    os.makedirs(FAISS_DIR, exist_ok=True)
    os.makedirs(METADATA_DIR, exist_ok=True)

    # Check if embeddings.pkl exists
    if os.path.exists(EMBEDDINGS_PKL) and os.path.exists(METADATA_PKL):
        print("Embeddings and metadata found. Loading...")
        embeddings = load_embeddings(EMBEDDINGS_PKL)
        metadata = load_metadata(METADATA_PKL)
    else:
        print("Embeddings or metadata not found. Generating embeddings...")
        embeddings = generate_embeddings(INPUT_CSV, TEXT_COLUMN, MODEL_NAME)
        # Load the entire CSV to create metadata
        print("Loading metadata from CSV...")
        df = pd.read_csv(INPUT_CSV)
        metadata = df.to_dict(orient='records')
        # Save embeddings and metadata
        save_embeddings(embeddings, EMBEDDINGS_PKL)
        save_metadata(metadata, METADATA_PKL)

    # Check if FAISS index exists
    if os.path.exists(FAISS_INDEX_PATH):
        print("FAISS index found. Loading...")
        index = load_faiss_index(FAISS_INDEX_PATH)
    else:
        print("FAISS index not found. Creating FAISS index...")
        index = create_faiss_index(embeddings, INDEX_TYPE)
        save_faiss_index(index, FAISS_INDEX_PATH)

    # If query is provided via command-line arguments
    if args.query:
        print(f"\nProcessing query: {args.query}")
        results = query_faiss(index, metadata, args.query, MODEL_NAME, top_k=TOP_K)
        print("\nSearch Results:")
        for rank, record in enumerate(results, start=1):
            print(f"Rank {rank}:")
            print(f"  Article ID     : {record['article_id']}")
            print(f"  Paragraph ID   : {record['paragraph_id']}")
            print(f"  Distance       : {record['distance']}")
            print(f"  Paragraph Text : {record['paragraph_text']}")
            print(f"  Internal Links : {record['internal_links']}\n")
    elif args.interactive:
        # Interactive query mode
        print("\n--- FAISS Query Interface ---")
        print("You can now perform similarity searches on your FAISS index.")
        print("Type 'exit' to quit the interface.\n")
        while True:
            query_text = input("Enter your query text (or type 'exit' to quit): ").strip()
            if query_text.lower() == 'exit':
                print("Exiting the query interface.")
                break
            if not query_text:
                print("Empty query. Please enter valid text.")
                continue
            results = query_faiss(index, metadata, query_text, MODEL_NAME, top_k=TOP_K)
            print("\nSearch Results:")
            for rank, record in enumerate(results, start=1):
                print(f"Rank {rank}:")
                print(f"  Article ID     : {record['article_id']}")
                print(f"  Paragraph ID   : {record['paragraph_id']}")
                print(f"  Distance       : {record['distance']}")
                print(f"  Paragraph Text : {record['paragraph_text']}")
                print(f"  Internal Links : {record['internal_links']}\n")
    else:
        # If no query is provided, enter interactive mode by default
        print("\nNo query provided. Entering interactive query interface.")
        print("--- FAISS Query Interface ---")
        print("You can now perform similarity searches on your FAISS index.")
        print("Type 'exit' to quit the interface.\n")
        while True:
            query_text = input("Enter your query text (or type 'exit' to quit): ").strip()
            if query_text.lower() == 'exit':
                print("Exiting the query interface.")
                break
            if not query_text:
                print("Empty query. Please enter valid text.")
                continue
            results = query_faiss(index, metadata, query_text, MODEL_NAME, top_k=TOP_K)
            print("\nSearch Results:")
            for rank, record in enumerate(results, start=1):
                print(f"Rank {rank}:")
                print(f"  Article ID     : {record['article_id']}")
                print(f"  Paragraph ID   : {record['paragraph_id']}")
                print(f"  Distance       : {record['distance']}")
                print(f"  Paragraph Text : {record['paragraph_text']}")
                print(f"  Internal Links : {record['internal_links']}\n")

if __name__ == "__main__":
    main()
