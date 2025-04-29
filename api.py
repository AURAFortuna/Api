from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import pickle
import os
import argparse
import pandas as pd
from surprise import Dataset, Reader, SVD

app = FastAPI(title="SVD Recommender API")

# Global variable to store the trained model
recommender = None

class UserRequest(BaseModel):
    user_id: str
    n_recommendations: Optional[int] = 5

class Interaction(BaseModel):
    user_id: str
    outfit_id: str
    interaction: int  # -1 for dislike, 1 for like, 3 for wishlist

class InteractionRequest(BaseModel):
    interactions: List[Interaction]

class Recommendation(BaseModel):
    outfit_id: str
    score: float
    is_new: bool

class RecommendationResponse(BaseModel):
    recommendations: List[Recommendation]

@app.on_event("startup")
async def startup_event():
    """Load the trained model when the API starts."""
    global recommender
    try:
        # Initialize the recommender
        recommender = SVD()
        
        # Load the trained model
        model_path = os.getenv('MODEL_PATH', 'trained_svd_model.pkl')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
            
        with open(model_path, 'rb') as f:
            recommender = pickle.load(f)
            
        # Load the user interactions data
        data_path = os.getenv('DATA_PATH', 'small_user_outfit_interactions.csv')
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found at {data_path}")
            
        print("Model loaded successfully!")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "healthy", "message": "SVD Recommender API is running"}

@app.post("/recommend", response_model=RecommendationResponse)
async def get_recommendations(request: UserRequest):
    """Get recommendations for a user."""
    if recommender is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Load current interactions
        data_path = os.getenv('DATA_PATH', 'small_user_outfit_interactions.csv')
        user_interactions = pd.read_csv(data_path)
        
        # Get all unique outfit IDs
        all_outfits = set(user_interactions['outfit_id'])
        
        # Get outfits the user has already interacted with and their ratings
        user_rated_outfits = user_interactions[
            user_interactions['user_id'] == request.user_id
        ][['outfit_id', 'interaction']].set_index('outfit_id')['interaction'].to_dict()
        
        # Get predictions for all outfits with stronger penalties for previously rated ones
        predictions = []
        for outfit_id in all_outfits:
            pred = recommender.predict(request.user_id, outfit_id)
            # Apply stronger penalty if outfit was previously rated
            if outfit_id in user_rated_outfits:
                # Stronger penalties based on the original rating:
                # - For disliked outfits (-1), reduce score by 5.0
                # - For liked outfits (1), reduce score by 4.0
                # - For wishlisted outfits (3), reduce score by 4.5
                penalty = {
                    -1: 5.0,  # Stronger penalty for disliked items
                    1: 4.0,   # Stronger penalty for liked items
                    3: 4.5    # Stronger penalty for wishlisted items
                }.get(user_rated_outfits[outfit_id], 3.0)
                adjusted_score = pred.est - penalty
            else:
                adjusted_score = pred.est
            
            predictions.append((outfit_id, adjusted_score, outfit_id not in user_rated_outfits))
        
        # Sort by adjusted score and get top N
        predictions.sort(key=lambda x: x[1], reverse=True)
        top_n = predictions[:request.n_recommendations]
        
        # Format the response
        recommendations = [
            Recommendation(
                outfit_id=outfit_id,
                score=score,
                is_new=is_new
            )
            for outfit_id, score, is_new in top_n
        ]
        
        return RecommendationResponse(recommendations=recommendations)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/update-interactions")
async def update_interactions(request: InteractionRequest):
    """Save new user interactions to new_interactions.csv."""
    try:
        # Convert interactions to list of dictionaries
        interactions = [
            {
                'user_id': interaction.user_id,
                'outfit_id': interaction.outfit_id,
                'interaction': interaction.interaction
            }
            for interaction in request.interactions
        ]
        
        # Convert to DataFrame
        new_df = pd.DataFrame(interactions)
        
        # Append to new_interactions.csv
        if os.path.exists('new_interactions.csv'):
            existing_df = pd.read_csv('new_interactions.csv')
            new_df = pd.concat([existing_df, new_df], ignore_index=True)
        
        # Save to new_interactions.csv
        new_df.to_csv('new_interactions.csv', index=False)
        
        return {"status": "success", "message": f"Added {len(interactions)} interactions to new_interactions.csv"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Run SVD Recommender API')
    parser.add_argument('--model', type=str, default='trained_svd_model.pkl', help='Path to the trained model file')
    parser.add_argument('--data', type=str, default='small_user_outfit_interactions.csv', help='Path to the user interactions data file')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to run the API on')
    parser.add_argument('--port', type=int, default=8000, help='Port to run the API on')
    args = parser.parse_args()
    
    # Set environment variables for the API
    os.environ['MODEL_PATH'] = args.model
    os.environ['DATA_PATH'] = args.data
    
    # Run the API
    uvicorn.run(app, host=args.host, port=args.port) 