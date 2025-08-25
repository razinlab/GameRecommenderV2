import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os
import random

MODEL_PATH = "models/finetuned-bge-small-en-v1.5-best"
EMBEDDINGS_PATH = "data/game_embeddings.npy"
GAME_DATA_PATH = "data/GameData_final.csv"

GAME_TITLE_COLUMN = "name"
GAME_ID_COLUMN = "game_id"
IMAGE_URL_COLUMN = "image_url"

STEAM_LINK_COLUMN = "steam_link"
GOG_LINK_COLUMN = "gog_link"
XBOX_LINK_COLUMN = "xbox_link"
PLAYSTATION_LINK_COLUMN = "playstation_link"
EPIC_LINK_COLUMN = "epic_link"
NINTENDO_LINK_COLUMN = "nintendo_link"
OFFICIAL_LINK_COLUMN = "official_link"

TOP_N_TO_FETCH = 50
NUM_RANDOM_TO_DISPLAY = 10

model = None
game_embeddings = None
games_df = None
game_titles = None


def load_resources():
    global model, game_embeddings, games_df, game_titles

    print(f"Loading model from: {MODEL_PATH}")
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model directory not found at {MODEL_PATH}")
        raise FileNotFoundError(f"Model directory not found: {MODEL_PATH}")
    try:
        model = SentenceTransformer(MODEL_PATH)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading the sentence transformer model: {e}")
        raise

    print(f"\nLoading game embeddings from: {EMBEDDINGS_PATH}")
    if not os.path.exists(EMBEDDINGS_PATH):
        print(f"Error: Game embeddings file not found at {EMBEDDINGS_PATH}")
        raise FileNotFoundError(f"Game embeddings file not found: {EMBEDDINGS_PATH}")
    try:
        game_embeddings = np.load(EMBEDDINGS_PATH)
        print(f"Game embeddings loaded successfully. Shape: {game_embeddings.shape}")
    except Exception as e:
        print(f"Error loading game embeddings: {e}")
        raise

    print(f"\nLoading game data from: {GAME_DATA_PATH}")
    if not os.path.exists(GAME_DATA_PATH):
        print(f"Error: Game data CSV file not found at {GAME_DATA_PATH}")
        raise FileNotFoundError(f"Game data CSV file not found: {GAME_DATA_PATH}")
    try:
        games_df = pd.read_csv(GAME_DATA_PATH)

        required_columns = [GAME_TITLE_COLUMN, GAME_ID_COLUMN, IMAGE_URL_COLUMN]

        for col in required_columns:
            if col not in games_df.columns:
                print(f"Error: Column '{col}' not found in {GAME_DATA_PATH}.")
                print(f"Available columns are: {games_df.columns.tolist()}")
                print(
                    f"Please update the relevant column name variables (e.g., GAME_ID_COLUMN) at the top of recommender.py.")
                raise ValueError(f"Missing required column: {col}. Please configure it in recommender.py.")

        game_titles = games_df[GAME_TITLE_COLUMN]
        print(f"Game data loaded successfully. Found {len(games_df)} games.")
    except Exception as e:
        print(f"Error loading game data CSV: {e}")
        raise


try:
    load_resources()
except Exception as e:
    print(f"FATAL: Failed to initialize recommender resources: {e}")

    model = None
    game_embeddings = None
    games_df = None
    game_titles = None


def get_recommendations(query_prompt: str):
    """
    Performs a semantic search and returns detailed game information for a random selection from top results.
    """
    global model, game_embeddings, games_df, game_titles

    if model is None or game_embeddings is None or games_df is None or game_titles is None:
        print("Error: Recommender resources not loaded properly. Cannot perform search.")
        return []

    if query_prompt is None or not query_prompt.strip():
        print("Error: Query prompt cannot be empty.")
        return []

    try:
        query_embedding = model.encode(query_prompt, convert_to_tensor=False)
        query_embedding = query_embedding.reshape(1, -1)

        similarities = cosine_similarity(query_embedding, game_embeddings)[0]

        num_games_available = len(similarities)
        current_top_n_fetch = min(TOP_N_TO_FETCH, num_games_available)

        if current_top_n_fetch == 0:
            return []

        top_n_indices = np.argsort(-similarities)[:current_top_n_fetch]

        fetched_results = []
        for idx in top_n_indices:
            game_data_row = games_df.iloc[idx]

            def get_data_from_row(column_name, default=""):
                if column_name in game_data_row and pd.notna(game_data_row[column_name]):
                    return game_data_row[column_name]
                return default

            links = {
                "steam": get_data_from_row(STEAM_LINK_COLUMN),
                "gog": get_data_from_row(GOG_LINK_COLUMN),
                "xbox": get_data_from_row(XBOX_LINK_COLUMN),
                "playstation": get_data_from_row(PLAYSTATION_LINK_COLUMN),
                "epic": get_data_from_row(EPIC_LINK_COLUMN),
                "nintendo": get_data_from_row(NINTENDO_LINK_COLUMN),
                "official": get_data_from_row(OFFICIAL_LINK_COLUMN)
            }

            game_id_val = get_data_from_row(GAME_ID_COLUMN)

            if not game_id_val or not isinstance(game_id_val, (str, int, float)):
                game_id_val = f"generated_id_{idx}"

            fetched_results.append({
                "id": str(game_id_val),
                "text": game_titles.iloc[idx],
                "score": float(similarities[idx]),
                "image_url": get_data_from_row(IMAGE_URL_COLUMN),
                "links": links
            })

        if not fetched_results:
            return []

        if len(fetched_results) <= NUM_RANDOM_TO_DISPLAY:
            random_selection = fetched_results
        else:
            random_selection = random.sample(fetched_results, NUM_RANDOM_TO_DISPLAY)

        return random_selection

    except Exception as e:
        print(f"An error occurred during semantic search: {e}")
        return []


if __name__ == "__main__":

    if model is not None and game_embeddings is not None and games_df is not None:
        example_prompt = "cozy farming simulation with crafting elements"
        print(f"\nPerforming semantic search for prompt: '{example_prompt}'")

        recommendations = get_recommendations(example_prompt)

        if recommendations:
            print(f"\nFetched and randomly selected {len(recommendations)} games:")
            for i, game in enumerate(recommendations):
                print(f"{i + 1}. {game['text']} (ID: {game['id']}, Similarity: {game['score']:.4f})")
                print(f"   Image: {game['image_url']}")
                print(f"   Links: {game['links']}")
        else:
            print("No results found or an error occurred during direct testing.")
    else:
        print("Recommender resources not loaded. Cannot run test. Check errors during initial loading.")

