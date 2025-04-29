import pandas as pd
import numpy as np
from surprise import Dataset, Reader, SVD
import pickle
import os
import time
from typing import List, Dict
import argparse

class ModelUpdater:
    def __init__(self, data_path: str, model_path: str):
        """Initialize the model updater."""
        self.data_path = data_path
        self.model_path = model_path
        self.model = None
        self.user_interactions = None
        self.load_data()
        self.load_model()

    def load_data(self):
        """Load the existing user interactions data."""
        if os.path.exists(self.data_path):
            self.user_interactions = pd.read_csv(self.data_path)
        else:
            raise FileNotFoundError(f"Data file not found at {self.data_path}")

    def load_model(self):
        """Load the existing model or create a new one."""
        if os.path.exists(self.model_path):
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
        else:
            self.model = SVD()

    def update_from_new_interactions(self):
        """Update model with new interactions and clean the new interactions file."""
        if not os.path.exists('new_interactions.csv'):
            print("No new interactions to process")
            return False

        # Load new interactions
        new_interactions = pd.read_csv('new_interactions.csv')
        if len(new_interactions) == 0:
            print("No new interactions to process")
            return False

        # Combine with existing data
        self.user_interactions = pd.concat([self.user_interactions, new_interactions], ignore_index=True)
        
        # Remove duplicates (keep the latest interaction)
        self.user_interactions = self.user_interactions.drop_duplicates(
            subset=['user_id', 'outfit_id'], 
            keep='last'
        )
        
        # Save updated data
        self.user_interactions.to_csv(self.data_path, index=False)
        print(f"Added {len(new_interactions)} new interactions. Total interactions: {len(self.user_interactions)}")

        # Retrain model
        self.retrain_model()

        # Clean new interactions file
        os.remove('new_interactions.csv')
        print("Cleaned new_interactions.csv")

        return True

    def retrain_model(self):
        """Retrain the model with updated data."""
        # Define the rating scale
        reader = Reader(rating_scale=(-4, 3))
        
        # Load the data into Surprise format
        data = Dataset.load_from_df(
            self.user_interactions[['user_id', 'outfit_id', 'interaction']], 
            reader
        )
        
        # Build and train the model
        trainset = data.build_full_trainset()
        self.model.fit(trainset)
        
        # Save the updated model
        with open(self.model_path, 'wb') as f:
            pickle.dump(self.model, f)
        
        print("Model retrained and saved successfully!")

def main():
    parser = argparse.ArgumentParser(description='Periodically update SVD recommender model')
    parser.add_argument('--data', type=str, default='small_user_outfit_interactions.csv', 
                      help='Path to the user interactions data file')
    parser.add_argument('--model', type=str, default='trained_svd_model.pkl',
                      help='Path to save the trained model')
    parser.add_argument('--interval', type=int, default=300,
                      help='Update interval in seconds (default: 300)')
    args = parser.parse_args()

    # Initialize the updater
    updater = ModelUpdater(args.data, args.model)
    
    print(f"Starting periodic updates every {args.interval} seconds...")
    while True:
        try:
            updater.update_from_new_interactions()
            time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\nStopping periodic updates...")
            break
        except Exception as e:
            print(f"Error during update: {e}")
            time.sleep(args.interval)

if __name__ == "__main__":
    main() 