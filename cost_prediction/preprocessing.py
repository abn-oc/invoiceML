import pandas as pd
import sqlite3
from sklearn.model_selection import train_test_split

def load_and_clean_data(db_path):
    """
    Connects to the database, extracts data, and prepares it for training using ONLY Dollars.
    """
    conn = sqlite3.connect(db_path)
    
    # Selecting only Dollars and Freight
    query = "SELECT Dollars, Freight FROM vendor_invoice"
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    # Basic cleaning
    df = df.dropna()
    
    # Define features (X) and target (y)
    # Double brackets [[ ]] keep it as a 2D DataFrame (required for sklearn)
    X = df[['Dollars']]
    y = df['Freight']
    
    # Split the data
    return train_test_split(X, y, test_size=0.2, random_state=42)