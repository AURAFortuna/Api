from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import List, Optional, Dict

import pickle
import os
import pandas as pd
from surprise import Dataset, Reader, SVD

# --- Firebase Admin Setup ---
import firebase_admin
from firebase_admin import credentials, firestore, auth

if not firebase_admin._apps:
    firebase_admin.initialize_app()
db = firestore.client()
# --- End Firebase Admin Setup ---

app = FastAPI(title="AURA Recommender API - Firestore & Auth (IDs Only)") # Updated title

# --- Authentication Setup ---
auth_scheme = HTTPBearer()

async def get_current_user(credentials: HTTPAuthorizationCredentials = Security(auth_scheme)) -> str:
    token = credentials.credentials
    try:
        decoded_token = auth.verify_id_token(token)
        return decoded_token['uid']
    except auth.ExpiredIdTokenError:
         raise HTTPException(status_code=401, detail="Token expired", headers={"WWW-Authenticate": "Bearer"})
    except auth.InvalidIdTokenError as e:
         print(f"Token verification failed: {e}")
         raise HTTPException(status_code=401, detail="Invalid token", headers={"WWW-Authenticate": "Bearer"})
    except Exception as e:
        print(f"Authentication error: {e}")
        raise HTTPException(status_code=401, detail="Could not validate credentials", headers={"WWW-Authenticate": "Bearer"})
# --- End Authentication Setup ---

# --- Data Models ---
class UserRequest(BaseModel):
    n_recommendations: Optional[int] = 10

class InteractionInput(BaseModel):
    outfit_id: str
    interaction: int

class InteractionRequest(BaseModel):
    interactions: List[InteractionInput]

# --- COMMENTED OUT: Model for returning full outfit details ---
# class OutfitDetails(BaseModel):
#     outfit_id: str
#     brand: Optional[str] = None
#     description: Optional[str] = None
#     image_path: Optional[str] = None
#     style: Optional[str] = None
#     # Add other fields needed by FlutterFlow

# --- MODIFIED: Response model now returns a list of strings (outfit IDs) ---
class RecommendationResponse(BaseModel):
    recommendations: List[str] # CHANGED: Returning list of outfit IDs now
# --- End Data Models ---


# --- Global Variables ---
recommender: Optional[SVD] = None
all_outfit_ids: set = set()
# --- End Global Variables ---


# --- COMMENTED OUT: Helper Function to fetch full details ---
# async def fetch_outfit_details(outfit_ids: List[str]) -> Dict[str, OutfitDetails]:
#     """Fetches outfit details from Firestore for a list of outfit IDs."""
#     outfit_details = {}
#     if not outfit_ids:
#         return outfit_details
#     # Use parallel fetches if possible, or batch reads for larger lists
#     for outfit_id in outfit_ids:
#         try:
#             doc_ref = db.collection('outfits').document(outfit_id)
#             # doc = await doc_ref.get() # Use async if uvicorn workers > 1
#             doc = doc_ref.get() # Use sync if uvicorn workers = 1
#             if doc.exists:
#                 data = doc.to_dict()
#                 outfit_details[outfit_id] = OutfitDetails(
#                     outfit_id=doc.id,
#                     brand=data.get('brand'),
#                     description=data.get('description'),
#                     image_path=data.get('image_path'),
#                     style=data.get('Style') # Note case difference
#                     # map other fields...
#                 )
#             else:
#                  print(f"Warning: Outfit document {outfit_id} not found.")
#         except Exception as e:
#             print(f"Error fetching details for outfit {outfit_id}: {e}")
#     return outfit_details
# --- End Commented Out Helper Function ---


# --- API Events ---
@app.on_event("startup")
async def startup_event():
    global recommender, all_outfit_ids
    print("API Startup: Loading resources...")
    try:
        # Load Model
        model_path = os.getenv('MODEL_PATH', 'trained_svd_model.pkl')
        if not os.path.exists(model_path):
            print(f"ERROR: Model file not found at {model_path}. API cannot serve recommendations.")
            return
        with open(model_path, 'rb') as f:
            recommender = pickle.load(f)
        print(f"Model loaded successfully from {model_path}!")

        # Load All Outfit IDs from 'outfits' collection
        outfits_ref = db.collection('outfits')
        docs = outfits_ref.select([]).stream()
        all_outfit_ids = {doc.id for doc in docs}
        if not all_outfit_ids:
             print("Warning: No outfit IDs found in 'outfits' collection.")
        else:
             print(f"Loaded {len(all_outfit_ids)} unique outfit IDs.")

    except Exception as e:
        print(f"FATAL ERROR during startup: {e}")
        recommender = None

@app.get("/")
async def root():
    global recommender
    status = "healthy" if recommender else "degraded (model not loaded/startup failed)"
    return {"status": status, "message": "AURA Recommender API"}


@app.post("/recommend", response_model=RecommendationResponse) # Response model expects List[str]
async def get_recommendations(
    request: UserRequest,
    user_id: str = Depends(get_current_user)
):
    global recommender, all_outfit_ids

    if recommender is None:
        raise HTTPException(status_code=503, detail="Recommendation model not available.")
    if not all_outfit_ids:
         raise HTTPException(status_code=404, detail="No outfits available for recommendations.")

    print(f"Generating recommendations for user: {user_id}")
    try:
        # --- Fetch User's Interactions from Firestore (Likes, Dislikes, Wishlist) ---
        user_ref = db.collection('users').document(user_id)
        user_rated_outfits: Dict[str, int] = {}

        # Fetch Likes
        likes_ref = db.collection('Likes').where('user_ref_like', '==', user_ref)
        like_docs = likes_ref.stream()
        for doc in like_docs:
            data = doc.to_dict(); outfit_ref = data.get('outfit_ref_like')
            if outfit_ref and hasattr(outfit_ref, 'id'): user_rated_outfits[outfit_ref.id] = 1

        # Fetch Dislikes
        dislikes_ref = db.collection('Dislikes').where('user_ref_dislike', '==', user_ref)
        dislike_docs = dislikes_ref.stream()
        for doc in dislike_docs:
             data = doc.to_dict(); outfit_ref = data.get('outfit_ref_dislike')
             if outfit_ref and hasattr(outfit_ref, 'id'): user_rated_outfits[outfit_ref.id] = -1

        # Fetch Wishlist
        wishlist_ref = db.collection('Wishlist').where('user_ref_wishlist', '==', user_ref)
        wish_docs = wishlist_ref.stream()
        for doc in wish_docs:
            data = doc.to_dict(); outfit_ref = data.get('outfit_ref_wishlist')
            if outfit_ref and hasattr(outfit_ref, 'id'): user_rated_outfits[outfit_ref.id] = 3

        print(f"Fetched {len(user_rated_outfits)} total interactions for user {user_id}")
        # --- End Firestore Interaction Fetch ---

        # Generate predictions using SVD model
        predictions = []
        for outfit_id in all_outfit_ids:
            if not outfit_id: continue

            pred = recommender.predict(uid=user_id, iid=outfit_id)
            adjusted_score = pred.est
            is_new = outfit_id not in user_rated_outfits

            if not is_new:
                original_rating = user_rated_outfits.get(outfit_id)
                penalty = 0.0
                if original_rating == 1: penalty = 4.0
                elif original_rating == -1: penalty = 5.0
                elif original_rating == 3: penalty = 4.5
                adjusted_score -= penalty

            predictions.append((outfit_id, adjusted_score, is_new))

        # Sort by score
        predictions.sort(key=lambda x: x[1], reverse=True)

        # --- MODIFIED: Extract only the top N outfit IDs ---
        recommended_outfit_ids = [
            outfit_id for outfit_id, score, is_new in predictions[:request.n_recommendations]
        ]
        print(f"Returning {len(recommended_outfit_ids)} recommended outfit IDs.")
        
        # --- COMMENTED OUT: Fetching and formatting full details ---
        # # --- Fetch Full Details for Recommended Outfits ---
        # outfit_details_map = await fetch_outfit_details(recommended_outfit_ids)
        # # --- Format final response ---
        # final_recommendations = []
        # for outfit_id in recommended_outfit_ids:
        #     details = outfit_details_map.get(outfit_id)
        #     if details:
        #         final_recommendations.append(details)
        # print(f"Returning {len(final_recommendations)} recommendations with details.")
        # return RecommendationResponse(recommendations=final_recommendations)
        # --- End Commented Out Detail Fetching ---

        # --- Return the list of IDs ---
        return RecommendationResponse(recommendations=recommended_outfit_ids) # Sends List[str]

    except Exception as e:
        print(f"ERROR generating recommendations for user {user_id}: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")


# --- /update-interactions endpoint remains the same as before ---
@app.post("/update-interactions")
async def update_interactions(
    request: InteractionRequest,
    user_id: str = Depends(get_current_user)
):
    # ... (Keep the previous implementation that writes to Likes/Dislikes/Wishlist) ...
    print(f"Updating interactions for user: {user_id}")
    try:
        batch = db.batch()
        count = 0
        user_ref = db.collection('users').document(user_id)

        for interaction in request.interactions:
            outfit_ref = db.collection('outfits').document(interaction.outfit_id)
            interaction_data = { "timestamp": firestore.SERVER_TIMESTAMP }
            target_collection = None

            if interaction.interaction == 1:
                target_collection = 'Likes'
                interaction_data['user_ref_like'] = user_ref
                interaction_data['outfit_ref_like'] = outfit_ref
                interaction_data['created_at_like'] = firestore.SERVER_TIMESTAMP
            elif interaction.interaction == -1:
                target_collection = 'Dislikes'
                interaction_data['user_ref_dislike'] = user_ref
                interaction_data['outfit_ref_dislike'] = outfit_ref
                interaction_data['created_at_dislike'] = firestore.SERVER_TIMESTAMP
            elif interaction.interaction == 3:
                target_collection = 'Wishlist'
                interaction_data['user_ref_wishlist'] = user_ref
                interaction_data['outfit_ref_wishlist'] = outfit_ref
                interaction_data['created_at_wishlist'] = firestore.SERVER_TIMESTAMP
            else:
                print(f"Warning: Skipping unknown interaction type {interaction.interaction} for outfit {interaction.outfit_id}")
                continue

            if target_collection:
                doc_ref = db.collection(target_collection).document()
                batch.set(doc_ref, interaction_data)
                count += 1

        if count > 0:
            batch.commit()
            print(f"Successfully added {count} interactions for user {user_id}")
            return {"status": "success", "message": f"Added {count} interactions to Firestore"}
        else:
            print(f"No valid interactions provided for user {user_id}")
            return {"status": "no_op", "message": "No valid interactions were processed"}

    except Exception as e:
        print(f"ERROR updating interactions for user {user_id}: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal server error while saving interactions: {e}")


# --- Main Execution (for testing locally) ---
if __name__ == "__main__":
    import uvicorn
    print("Starting Uvicorn server directly (for local testing)...")
    uvicorn.run("api:app", host='127.0.0.1', port=8000, reload=True)
