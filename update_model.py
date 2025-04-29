import pandas as pd
from surprise import Dataset, Reader, SVD
import pickle
import os
import time
import argparse
from typing import List, Dict, Tuple # Added typing

# --- Firebase Admin Setup ---
import firebase_admin
from firebase_admin import credentials, firestore

# Initialize Firebase Admin SDK
if not firebase_admin._apps:
    firebase_admin.initialize_app()
db = firestore.client()
# --- End Firebase Admin Setup ---

class ModelUpdater:
    def __init__(self, model_path: str):
        """Initialize the model updater."""
        self.model_path = model_path
        self.model: Optional[SVD] = None
        self.user_interactions_df: Optional[pd.DataFrame] = None
        self.load_or_create_model() # Changed method name

    def load_or_create_model(self):
        """Load the existing model or create a new SVD instance."""
        if os.path.exists(self.model_path):
            try:
                with open(self.model_path, 'rb') as f:
                    self.model = pickle.load(f)
                print(f"Loaded existing model from {self.model_path}")
            except Exception as e:
                print(f"Error loading model file {self.model_path}: {e}. Creating new SVD.")
                self.model = SVD() # Hyperparameters can be set here if needed
        else:
            print(f"Model file {self.model_path} not found. Creating new SVD.")
            self.model = SVD()

    def fetch_all_interactions(self) -> bool:
        """Fetch all interactions from Likes, Dislikes, Wishlist and load into DataFrame."""
        print("Fetching interactions from Firestore (Likes, Dislikes, Wishlist)...")
        all_interactions: List[Dict] = []
        processed_count = 0

        try:
            # Fetch Likes (interaction = 1)
            print("Fetching Likes...")
            likes_docs = db.collection('Likes').stream()
            for doc in likes_docs:
                data = doc.to_dict()
                user_ref = data.get('user_ref_like')
                outfit_ref = data.get('outfit_ref_like')
                if user_ref and outfit_ref and hasattr(user_ref, 'id') and hasattr(outfit_ref, 'id'):
                    all_interactions.append({
                        'user_id': user_ref.id,
                        'outfit_id': outfit_ref.id,
                        'interaction': 1
                    })
                    processed_count += 1
            print(f"Fetched {processed_count} Likes.")
            
            # Fetch Dislikes (interaction = -1)
            print("Fetching Dislikes...")
            dislikes_docs = db.collection('Dislikes').stream()
            dislike_count = 0
            for doc in dislikes_docs:
                data = doc.to_dict()
                user_ref = data.get('user_ref_dislike')
                outfit_ref = data.get('outfit_ref_dislike')
                if user_ref and outfit_ref and hasattr(user_ref, 'id') and hasattr(outfit_ref, 'id'):
                    all_interactions.append({
                        'user_id': user_ref.id,
                        'outfit_id': outfit_ref.id,
                        'interaction': -1
                    })
                    dislike_count += 1
            processed_count += dislike_count
            print(f"Fetched {dislike_count} Dislikes.")

            # Fetch Wishlist (interaction = 3)
            print("Fetching Wishlist...")
            wishlist_docs = db.collection('Wishlist').stream()
            wish_count = 0
            for doc in wishlist_docs:
                data = doc.to_dict()
                user_ref = data.get('user_ref_wishlist')
                outfit_ref = data.get('outfit_ref_wishlist')
                if user_ref and outfit_ref and hasattr(user_ref, 'id') and hasattr(outfit_ref, 'id'):
                    all_interactions.append({
                        'user_id': user_ref.id,
                        'outfit_id': outfit_ref.id,
                        'interaction': 3
                    })
                    wish_count += 1
            processed_count += wish_count
            print(f"Fetched {wish_count} Wishlist items.")

            print(f"Finished fetching. Total valid interactions processed: {processed_count}")

            if not all_interactions:
                print("Warning: No interactions found in Firestore. Cannot train model.")
                self.user_interactions_df = pd.DataFrame(columns=['user_id', 'outfit_id', 'interaction'])
                return False

            self.user_interactions_df = pd.DataFrame(all_interactions)
            
            # Important: Handle potential duplicates if user could like AND dislike same item (shouldn't happen with good UI)
            # If duplicates exist, decide how to handle (e.g., keep latest based on timestamp - would require fetching timestamp too)
            # For now, assume UI prevents conflicting states or we just use whatever is fetched.
            
            print(f"Loaded DataFrame with {len(self.user_interactions_df)} interactions.")
            return True

        except Exception as e:
            print(f"ERROR fetching interactions from Firestore: {e}")
            import traceback
            traceback.print_exc()
            self.user_interactions_df = pd.DataFrame(columns=['user_id', 'outfit_id', 'interaction'])
            return False

    def retrain_model(self) -> bool:
        """Retrain the SVD model using the fetched Firestore data."""
        if self.user_interactions_df is None or self.user_interactions_df.empty:
            print("No interaction data loaded, skipping model retraining.")
            return False
        if self.model is None:
             print("Model instance not available, skipping retraining.")
             return False

        try:
            print("Preparing data for Surprise...")
            min_rating = self.user_interactions_df['interaction'].min()
            max_rating = self.user_interactions_df['interaction'].max()
            print(f"Detected rating scale: min={min_rating}, max={max_rating}")
            if pd.isna(min_rating) or pd.isna(max_rating):
                 # Handle case where df might be empty after filtering, etc.
                 if self.user_interactions_df.empty:
                      print("Interaction DataFrame is empty. Cannot determine rating scale.")
                      return False
                 else:
                      raise ValueError("Interaction data contains NaN values after processing.")

            reader = Reader(rating_scale=(min_rating, max_rating))
            data = Dataset.load_from_df(
                self.user_interactions_df[['user_id', 'outfit_id', 'interaction']],
                reader
            )

            print("Building trainset...")
            trainset = data.build_full_trainset()
            print(f"Trainset built with {trainset.n_users} users and {trainset.n_items} items.")

            print("Retraining SVD model...")
            self.model.fit(trainset) # Retrain the loaded/created model instance

            print("Saving updated model...")
            # Ensure directory exists? (Not usually needed if path is just filename)
            with open(self.model_path, 'wb') as f:
                pickle.dump(self.model, f)

            print(f"Model retrained and saved successfully to {self.model_path}!")
            return True

        except Exception as e:
            print(f"ERROR during model retraining: {e}")
            import traceback
            traceback.print_exc()
            return False

# --- Main Execution Loop ---
def main():
    parser = argparse.ArgumentParser(description='Periodically update SVD recommender model from Firestore')
    parser.add_argument('--model', type=str, default='trained_svd_model.pkl',
                      help='Path to load/save the trained model')
    parser.add_argument('--interval', type=int, default=3600, # Default to 1 hour
                      help='Update interval in seconds (default: 3600)')
    args = parser.parse_args()

    updater = ModelUpdater(args.model)

    print(f"Starting periodic model update check every {args.interval} seconds...")
    while True:
        print(f"\n--- Running update cycle at {time.ctime()} ---")
        cycle_success = False
        try:
            # 1. Fetch latest data from Firestore
            data_loaded = updater.fetch_all_interactions()

            # 2. Retrain if data was loaded successfully
            if data_loaded:
                cycle_success = updater.retrain_model()
            else:
                print("Skipping retraining due to issues loading data.")

        except KeyboardInterrupt:
            print("\nStopping periodic updates...")
            break
        except Exception as e:
            print(f"!! Unexpected error during update cycle: {e}")
            import traceback
            traceback.print_exc()

        status = "completed successfully" if cycle_success else "finished with errors/no data"
        print(f"--- Update cycle {status}. Sleeping for {args.interval} seconds... ---")
        time.sleep(args.interval)

if __name__ == "__main__":
    main()
