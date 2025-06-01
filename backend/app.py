import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import random
from flask import Flask, request, jsonify, g, send_from_directory
from flask_cors import CORS
import math
import json
import traceback
import sqlite3
import hashlib
import secrets
import jwt
import datetime
from functools import wraps

app = Flask(__name__)

DATABASE_PATH = "data/users.db"
SECRET_KEY = os.environ.get('JWT_SECRET_KEY', secrets.token_hex(16))
FRONTEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'frontend'))

CORS(app, resources={r"/*": {
    "origins": [
        "http://localhost:63342", "http://127.0.0.1:63342",
        "http://localhost:5500", "http://127.0.0.1:5500",
        "http://localhost:5000", "http://127.0.0.1:5000"
    ],
    "supports_credentials": True
}})

MODEL_PATH = "models/finetuned-bge-small-en-v1.5-best"
EMBEDDINGS_PATH = "data/game_embeddings.npy"
GAME_DATA_PATH = "data/GameData_final.csv"

GAME_ID_COLUMN = "game_id"
GAME_TITLE_COLUMN = "name"
GAME_COVER_URL_COLUMN = "cover_url"
GAME_STORYLINE_COLUMN = "storyline"
GAME_GENRES_COLUMN = "genres"
GAME_THEMES_COLUMN = "themes"
GAME_PLATFORMS_COLUMN = "platforms"
GAME_RELEASE_DATE_COLUMN = "release_date_human"
GAME_SCREENSHOTS_COLUMN = "screenshots"
GAME_LANGUAGES_COLUMN = "language_supports"
GAME_MODES_COLUMN = "game_modes"
GAME_MULTIPLAYER_MODES_COLUMN = "multiplayer_modes"
GAME_SIMILAR_GAMES_COLUMN = "similar_games"
GAME_STEAM_URL_COLUMN = "steam_url"
GAME_GOG_URL_COLUMN = "gog_url"
GAME_PLAYSTATION_URL_COLUMN = "playstation_url"
GAME_XBOX_URL_COLUMN = "xbox_url"
GAME_EPIC_URL_COLUMN = "epic_url"
GAME_NINTENDO_URL_COLUMN = "nintendo_url"
GAME_OFFICIAL_URL_COLUMN = "official_url"

TOP_N_TO_FETCH = 50
NUM_RANDOM_TO_DISPLAY = 10
MAX_SIMILAR_GAMES_IN_DETAIL = 5
DEFAULT_COVER_URL = "https://placehold.co/130x180/2c2c2e/e0e0e0?text=No+Cover"
DEFAULT_SCREENSHOT_URL = "https://placehold.co/600x338/2c2c2e/e0e0e0?text=No+Screenshot"
DEFAULT_SIMILAR_GAME_COVER = "https://placehold.co/100x140/2c2c2e/e0e0e0?text=N/A"
EXPECTED_DETAIL_COLUMNS = [
    GAME_ID_COLUMN, GAME_TITLE_COLUMN, GAME_COVER_URL_COLUMN, GAME_STORYLINE_COLUMN,
    GAME_GENRES_COLUMN, GAME_THEMES_COLUMN, GAME_PLATFORMS_COLUMN, GAME_RELEASE_DATE_COLUMN,
    GAME_SCREENSHOTS_COLUMN, GAME_LANGUAGES_COLUMN, GAME_MODES_COLUMN,
    GAME_MULTIPLAYER_MODES_COLUMN, GAME_SIMILAR_GAMES_COLUMN,
    GAME_STEAM_URL_COLUMN, GAME_GOG_URL_COLUMN, GAME_PLAYSTATION_URL_COLUMN,
    GAME_XBOX_URL_COLUMN, GAME_EPIC_URL_COLUMN, GAME_NINTENDO_URL_COLUMN,
    GAME_OFFICIAL_URL_COLUMN
]

recommender_model = None
game_embeddings = None
game_data_full_df = None


def convert_numpy_types_and_nan_to_none(data):
    if isinstance(data, dict):
        return {k: convert_numpy_types_and_nan_to_none(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_numpy_types_and_nan_to_none(item) for item in data]
    elif isinstance(data, (
            np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32,
            np.uint64)):
        return int(data)
    elif isinstance(data, (np.float64, np.float16, np.float32, np.float64)):
        return float(data) if not math.isnan(data) else None
    elif isinstance(data, (np.bool_)):
        return bool(data)
    elif pd.isna(data):
        return None
    return data


def safe_json_parse(data_string, default_value=None):
    if default_value is None: default_value = []
    cleaned_data_string = convert_numpy_types_and_nan_to_none(data_string)
    if cleaned_data_string is None: return default_value
    try:
        if isinstance(cleaned_data_string, (list, dict)): return cleaned_data_string
        if isinstance(cleaned_data_string, str):
            if cleaned_data_string.lower() == 'nan': return default_value
            processed_string = cleaned_data_string.replace("None", "null").replace("True", "true").replace("False",
                                                                                                           "false")
            if (processed_string.startswith('[') and processed_string.endswith(']')) or \
                    (processed_string.startswith('{') and processed_string.endswith('}')):
                try:
                    return json.loads(processed_string.replace("'", '"'))
                except json.JSONDecodeError:
                    try:
                        return json.loads(processed_string)
                    except json.JSONDecodeError:
                        return [processed_string]
            return [processed_string]
        return cleaned_data_string
    except (json.JSONDecodeError, TypeError):
        if isinstance(cleaned_data_string, str): return [cleaned_data_string]
        return default_value


def load_recommender_resources():
    global recommender_model, game_embeddings, game_data_full_df
    print("Attempting to load recommender resources...")
    resource_paths = {"Model": MODEL_PATH, "Embeddings": EMBEDDINGS_PATH, "Game Data CSV": GAME_DATA_PATH}
    all_paths_exist = True
    for name, path in resource_paths.items():
        abs_path = os.path.join(app.root_path, '..', path) if not os.path.isabs(path) else path

        if not os.path.exists(path):
            print(
                f"CRITICAL ERROR (recommender): Path for {name} NOT FOUND: {path} (Resolved to: {os.path.abspath(path)})")
            all_paths_exist = False
    if not all_paths_exist:
        return False

    try:
        recommender_model = SentenceTransformer(MODEL_PATH)
        print("✅ Recommender Model loaded successfully.")
    except Exception as e:
        print(f"Error loading recommender model from {MODEL_PATH}: {e}");
        traceback.print_exc();
        return False

    try:
        game_embeddings = np.load(EMBEDDINGS_PATH)
        print(f"✅ Recommender Embeddings loaded from {EMBEDDINGS_PATH}. Shape: {game_embeddings.shape}")
    except Exception as e:
        print(f"Error loading recommender embeddings from {EMBEDDINGS_PATH}: {e}");
        traceback.print_exc();
        return False

    try:
        temp_df = pd.read_csv(GAME_DATA_PATH, dtype={GAME_ID_COLUMN: str})
        for col in EXPECTED_DETAIL_COLUMNS:
            if col not in temp_df.columns: temp_df[col] = pd.NA

        if GAME_ID_COLUMN not in temp_df.columns or temp_df[GAME_ID_COLUMN].isnull().any():
            print(f"Error: Crucial column '{GAME_ID_COLUMN}' missing or contains null values in {GAME_DATA_PATH}.")
            return False

        try:
            numeric_ids = pd.to_numeric(temp_df[GAME_ID_COLUMN], errors='coerce')
            if numeric_ids.notnull().all():

                if (numeric_ids == numeric_ids.round()).all():
                    temp_df[GAME_ID_COLUMN] = numeric_ids.astype(int)

        except Exception:
            print(
                f"Warning: Some '{GAME_ID_COLUMN}' values in {GAME_DATA_PATH} could not be cleanly converted to int, kept as original type.")

        temp_df = temp_df.set_index(GAME_ID_COLUMN,
                                    drop=False)
        game_data_full_df = temp_df
        print(
            f"✅ Recommender Game data loaded from {GAME_DATA_PATH}. Found {len(game_data_full_df)} games. Index type: {game_data_full_df.index.dtype}, Column '{GAME_ID_COLUMN}' type: {game_data_full_df[GAME_ID_COLUMN].dtype}")
        return True
    except Exception as e:
        print(f"Error loading game data CSV from {GAME_DATA_PATH}: {e}");
        traceback.print_exc();
        return False


if not load_recommender_resources():
    print("FATAL: Recommender resources could not be loaded. Recommendation features will be unavailable.")


else:
    print("✅✅✅ Recommender resources successfully loaded at app initialization. ✅✅✅")


def get_minimal_game_details(game_id_orig):
    global game_data_full_df
    if game_data_full_df is None: return None
    try:
        game_id_converted = game_id_orig

        if game_data_full_df.index.dtype == 'object' and not isinstance(game_id_orig, str):
            game_id_converted = str(game_id_orig)
        elif pd.api.types.is_integer_dtype(game_data_full_df.index.dtype) and not isinstance(game_id_orig, int):
            try:
                game_id_converted = int(float(str(game_id_orig)))
            except ValueError:
                print(f"Cannot convert ID {game_id_orig} to int for index lookup."); return None

        if game_id_converted not in game_data_full_df.index:
            return None
        game_info_series = game_data_full_df.loc[game_id_converted]
        minimal_details = {
            GAME_ID_COLUMN: game_info_series.name,
            GAME_TITLE_COLUMN: game_info_series.get(GAME_TITLE_COLUMN),
            GAME_COVER_URL_COLUMN: game_info_series.get(GAME_COVER_URL_COLUMN) or DEFAULT_SIMILAR_GAME_COVER
        }
        return convert_numpy_types_and_nan_to_none(minimal_details)
    except Exception as e:
        print(
            f"Error in get_minimal_game_details for ID {game_id_orig} (converted: {game_id_converted if 'game_id_converted' in locals() else 'N/A'}): {e}");
        traceback.print_exc();
        return None


def get_full_game_details_by_id(game_id_orig):
    global game_data_full_df
    if game_data_full_df is None: return None
    try:
        main_game_id_converted = game_id_orig

        if game_data_full_df.index.dtype == 'object' and not isinstance(game_id_orig, str):
            main_game_id_converted = str(game_id_orig)
        elif pd.api.types.is_integer_dtype(game_data_full_df.index.dtype) and not isinstance(game_id_orig, int):
            try:
                main_game_id_converted = int(float(str(game_id_orig)))
            except ValueError:
                print(f"Cannot convert main ID {game_id_orig} to int for index lookup."); return None

        if main_game_id_converted not in game_data_full_df.index:
            return None

        game_info_series = game_data_full_df.loc[main_game_id_converted]
        game_info = {}
        for col in EXPECTED_DETAIL_COLUMNS:
            if col == GAME_SIMILAR_GAMES_COLUMN or col == "urls": continue
            game_info[col] = game_info_series.get(col)

        game_info[GAME_ID_COLUMN] = game_info_series.name

        for col in [GAME_GENRES_COLUMN, GAME_THEMES_COLUMN, GAME_PLATFORMS_COLUMN, GAME_LANGUAGES_COLUMN,
                    GAME_MODES_COLUMN, GAME_MULTIPLAYER_MODES_COLUMN]:
            game_info[col] = safe_json_parse(game_info_series.get(col), [])

        raw_screenshots = game_info_series.get(GAME_SCREENSHOTS_COLUMN)
        parsed_screenshots = safe_json_parse(raw_screenshots, [])
        game_info[GAME_SCREENSHOTS_COLUMN] = [
            f"https:{s}" if isinstance(s, str) and s.startswith("//") else str(s)
            for s in parsed_screenshots if isinstance(s, str) and s
        ]
        if not game_info[GAME_SCREENSHOTS_COLUMN]: game_info[GAME_SCREENSHOTS_COLUMN] = [DEFAULT_SCREENSHOT_URL]

        game_info["urls"] = {
            k: game_info_series.get(globals().get(f"GAME_{k.upper()}_URL_COLUMN"))
            for k in ["steam", "gog", "playstation", "xbox", "epic", "nintendo", "official"]
        }
        game_info["urls"] = {k: v for k, v in game_info["urls"].items() if pd.notna(v) and str(v).strip()}

        similar_game_names_raw_string = game_info_series.get(GAME_SIMILAR_GAMES_COLUMN)
        similar_game_names_list = safe_json_parse(similar_game_names_raw_string, [])
        detailed_similar_games = []
        if similar_game_names_list:
            for sim_name_raw in similar_game_names_list[:MAX_SIMILAR_GAMES_IN_DETAIL]:
                if sim_name_raw is None or not isinstance(sim_name_raw, str) or not sim_name_raw.strip(): continue
                try:
                    if GAME_TITLE_COLUMN in game_data_full_df.columns and pd.api.types.is_string_dtype(
                            game_data_full_df[GAME_TITLE_COLUMN]):
                        matched_games = game_data_full_df[
                            game_data_full_df[GAME_TITLE_COLUMN].fillna('').str.lower() == sim_name_raw.strip().lower()]
                    else:
                        matched_games = pd.DataFrame()

                    if not matched_games.empty:
                        found_game_id = matched_games.index[0]
                        if str(found_game_id) == str(main_game_id_converted): continue
                        sim_details = get_minimal_game_details(found_game_id)
                        if sim_details: detailed_similar_games.append(sim_details)
                except Exception as e_sim:
                    print(f"Error looking up similar game by name '{sim_name_raw}': {e_sim}");
                    traceback.print_exc();
                    pass
        game_info[GAME_SIMILAR_GAMES_COLUMN] = detailed_similar_games
        return convert_numpy_types_and_nan_to_none(game_info)
    except Exception as e_full:
        print(
            f"Error in get_full_game_details_by_id for ID {game_id_orig} (converted: {main_game_id_converted if 'main_game_id_converted' in locals() else 'N/A'}): {e_full}");
        traceback.print_exc();
        return None


def semantic_search_and_get_top_ids(query_prompt: str, top_n: int):
    global recommender_model, game_embeddings, game_data_full_df
    if not query_prompt or not query_prompt.strip(): return []
    if recommender_model is None or game_embeddings is None or game_data_full_df is None:
        print("Error in semantic_search: Recommender resources (model, embeddings, or game_data_full_df) not loaded.")
        return []
    try:
        query_embedding = recommender_model.encode(query_prompt, convert_to_tensor=False).reshape(1, -1)
        similarities = cosine_similarity(query_embedding, game_embeddings)[0]

        top_n_indices = np.argsort(-similarities)

        valid_sorted_indices = top_n_indices[~np.isnan(similarities[top_n_indices])]

        actual_top_n_indices = valid_sorted_indices[:top_n]

        results_ids = [convert_numpy_types_and_nan_to_none(game_data_full_df.index[idx]) for idx in actual_top_n_indices
                       if idx < len(game_data_full_df)]

        return results_ids
    except Exception as e:
        print(f"Error in semantic_search: {e}");
        traceback.print_exc();
        return []


def get_db():
    db = getattr(g, '_database', None)
    if db is None:
        db_dir = os.path.dirname(DATABASE_PATH)
        if not os.path.exists(db_dir) and db_dir:
            os.makedirs(db_dir, exist_ok=True)
        db = g._database = sqlite3.connect(DATABASE_PATH)
        db.row_factory = sqlite3.Row
    return db


@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()


def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()


def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        if 'Authorization' in request.headers:
            auth_header = request.headers['Authorization']
            token = auth_header.split(" ")[1] if auth_header.startswith("Bearer ") else auth_header
        if not token: return jsonify({'message': 'Token is missing!'}), 401
        try:
            jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        except jwt.ExpiredSignatureError:
            return jsonify({'message': 'Token has expired!'}), 401
        except jwt.InvalidTokenError:
            return jsonify({'message': 'Token is invalid!'}), 401
        return f(*args, **kwargs)

    return decorated


def get_user_from_token():
    auth_header = request.headers.get('Authorization')
    if not auth_header or not auth_header.startswith('Bearer '): return None
    token = auth_header.split(' ')[1]
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        return payload
    except (jwt.ExpiredSignatureError, jwt.InvalidTokenError):
        return None


@app.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    if not data or not data.get('username') or not data.get('password'):
        return jsonify({'message': 'Missing username or password'}), 400
    username = data['username'];
    password = data['password']
    hashed_password = hash_password(password)
    db = get_db();
    cursor = db.cursor()
    try:
        cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
        if cursor.fetchone(): return jsonify({'message': 'Username already exists'}), 409
        cursor.execute("INSERT INTO users (username, password_hash) VALUES (?, ?)", (username, hashed_password))
        db.commit()
        return jsonify({'message': 'User registered successfully'}), 201
    except sqlite3.Error as e:
        db.rollback();
        print(f"SQLite registration error: {e}")
        return jsonify({'message': f'Database error during registration: {e}'}), 500
    finally:
        if cursor: cursor.close()


@app.route('/login', methods=['POST'])
def login_api():
    data = request.get_json()
    if not data or not data.get('username') or not data.get('password'):
        return jsonify({'message': 'Missing username or password'}), 400
    username = data['username'];
    password = data['password']
    db = get_db();
    cursor = db.cursor()
    try:
        cursor.execute("SELECT user_id, username, password_hash FROM users WHERE username = ?", (username,))
        user_row = cursor.fetchone()
        if user_row and user_row['password_hash'] == hash_password(password):
            token_payload = {'user_id': user_row['user_id'], 'username': user_row['username'],
                             'exp': datetime.datetime.utcnow() + datetime.timedelta(days=7)}
            token = jwt.encode(token_payload, SECRET_KEY, algorithm="HS256")
            return jsonify({'token': token, 'username': user_row['username']}), 200
        else:
            return jsonify({'message': 'Invalid username or password'}), 401
    except sqlite3.Error as e:
        print(f"SQLite login error: {e}")
        return jsonify({'message': f'Database error during login: {e}'}), 500
    finally:
        if cursor: cursor.close()


@app.route('/recommend', methods=['POST'])
def recommend_games_route():
    if not request.is_json: return jsonify({"error": "Request must be JSON"}), 400
    data = request.get_json()
    query_prompt = data.get('query_prompt') or data.get('query')
    if not query_prompt: return jsonify({"error": "Missing 'query_prompt' or 'query'"}), 400

    if recommender_model is None or game_embeddings is None or game_data_full_df is None:
        return jsonify({"error": "Recommender system not ready. Resources not loaded."}), 503

    top_n_game_ids = semantic_search_and_get_top_ids(query_prompt, TOP_N_TO_FETCH)
    if not top_n_game_ids: return jsonify({"error": "No similar game IDs found by semantic search."}), 404

    detailed_top_n_games = [details for game_id in top_n_game_ids if (details := get_full_game_details_by_id(game_id))]
    if not detailed_top_n_games: return jsonify(
        {"error": "Found IDs from search but failed to get full details for any."}), 404

    recommendations_to_display = random.sample(detailed_top_n_games,
                                               min(len(detailed_top_n_games), NUM_RANDOM_TO_DISPLAY))
    return jsonify(recommendations_to_display)


@app.route('/game_details/<game_id_str>', methods=['GET'])
def get_single_game_details_route(game_id_str):
    if not game_id_str: return jsonify({"error": "Missing game_id"}), 400
    if game_data_full_df is None: return jsonify({"error": "Recommender system not ready. Game data not loaded."}), 503
    details = get_full_game_details_by_id(game_id_str)
    return jsonify(details) if details else jsonify(
        {"error": f"Could not retrieve details for game ID {game_id_str}."}), 404


@app.route('/random_games', methods=['GET'])
def random_games_route():
    global game_data_full_df
    if game_data_full_df is None or game_data_full_df.empty:
        print("Error in /random_games: game_data_full_df is None or empty.")
        return jsonify({"error": "Game data not available."}), 500
    try:
        all_game_ids = game_data_full_df.index.tolist()
        if not all_game_ids: return jsonify({"error": "No games available to choose from."}), 404

        num_to_select = min(NUM_RANDOM_TO_DISPLAY, len(all_game_ids))
        random_game_ids_selected = random.sample(all_game_ids, num_to_select)

        random_games_details = [details for game_id in random_game_ids_selected if
                                (details := get_full_game_details_by_id(game_id))]

        if not random_games_details and random_game_ids_selected:
            return jsonify({"error": "Could not retrieve details for randomly selected games."}), 500
        return jsonify(random_games_details)
    except Exception as e:
        print(f"CRITICAL Error in /random_games: {e}");
        traceback.print_exc()
        return jsonify({"error": "An internal error occurred while fetching random games."}), 500


@app.route('/saved-games', methods=['POST'])
@token_required
def save_game():
    user_info = get_user_from_token();
    if not user_info: return jsonify({'message': 'Invalid token or user not found'}), 401
    user_id = user_info['user_id'];
    data = request.get_json()
    if not data or not data.get('game_id') or not data.get('game_name'):
        return jsonify({'message': 'Missing game_id or game_name'}), 400
    game_id_str = str(data['game_id']);
    game_name = data['game_name']
    image_url = data.get('image_url', DEFAULT_COVER_URL)
    game_urls_json = json.dumps(data.get('links_list', []))
    db = get_db();
    cursor = db.cursor()
    try:
        cursor.execute("SELECT save_id FROM saved_games WHERE user_id = ? AND game_id = ?", (user_id, game_id_str))
        if cursor.fetchone(): return jsonify({'message': 'Game already saved'}), 409
        cursor.execute(
            "INSERT INTO saved_games (user_id, game_id, game_name, image_url, game_url) VALUES (?, ?, ?, ?, ?)",
            (user_id, game_id_str, game_name, image_url, game_urls_json))
        db.commit();
        return jsonify({'message': 'Game saved successfully'}), 201
    except sqlite3.Error as e:
        db.rollback();
        print(f"SQLite save_game error: {e}")
        return jsonify({'message': f'Database error saving game: {e}'}), 500
    finally:
        if cursor: cursor.close()


@app.route('/saved-games', methods=['GET'])
@token_required
def get_saved_games():
    user_info = get_user_from_token();
    if not user_info: return jsonify({'message': 'Invalid token or user not found'}), 401
    user_id = user_info['user_id']
    db = get_db();
    cursor = db.cursor()
    try:
        cursor.execute("SELECT game_id, game_name, image_url, game_url FROM saved_games WHERE user_id = ?", (user_id,))
        games_list = []
        for row in cursor.fetchall():
            game_dict = dict(row)
            try:
                game_dict['links_list'] = json.loads(game_dict['game_url']) if game_dict['game_url'] else []
            except json.JSONDecodeError:
                game_dict['links_list'] = []
            games_list.append(game_dict)
        return jsonify({'games': games_list}), 200
    except sqlite3.Error as e:
        print(f"SQLite get_saved_games error: {e}")
        return jsonify({'message': f'Database error fetching saved games: {e}'}), 500
    finally:
        if cursor: cursor.close()


@app.route('/saved-games/<game_id_str>', methods=['DELETE'])
@token_required
def remove_saved_game(game_id_str):
    user_info = get_user_from_token();
    if not user_info: return jsonify({'message': 'Invalid token or user not found'}), 401
    user_id = user_info['user_id']
    db = get_db();
    cursor = db.cursor()
    try:
        cursor.execute("DELETE FROM saved_games WHERE user_id = ? AND game_id = ?", (user_id, game_id_str))
        db.commit()
        return jsonify({
            'message': 'Game removed successfully' if cursor.rowcount > 0 else 'Game not found'}), 200 if cursor.rowcount > 0 else 404
    except sqlite3.Error as e:
        db.rollback();
        print(f"SQLite remove_saved_game error: {e}")
        return jsonify({'message': f'Database error removing game: {e}'}), 500
    finally:
        if cursor: cursor.close()


@app.route('/')
def serve_index(): return send_from_directory(FRONTEND_DIR, 'index.html')


@app.route('/login.html')
def serve_login_page(): return send_from_directory(FRONTEND_DIR, 'login.html')


@app.route('/my_games.html')
def serve_my_games_page(): return send_from_directory(FRONTEND_DIR, 'my_games.html')


@app.route('/<path:filename>')
def serve_frontend_static(filename): return send_from_directory(FRONTEND_DIR, filename)


if __name__ == '__main__':
    db_dir = os.path.dirname(DATABASE_PATH)
    if not os.path.exists(db_dir) and db_dir:
        os.makedirs(db_dir, exist_ok=True)
        print(f"Created directory for database: {db_dir}")
    if not os.path.exists(DATABASE_PATH):
        print(f"Database not found at {DATABASE_PATH}. If you have a setup script, run it.")
    else:
        print(f"Using SQLite database at: {DATABASE_PATH}")

    if recommender_model and game_embeddings is not None and game_data_full_df is not None:
        print("Recommender resources previously loaded. Starting Flask server (debug mode)...")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:

        print("Recommender resources not loaded at init, attempting load for direct run...")
        if load_recommender_resources():
            print("Recommender resources loaded. Starting Flask server (debug mode)...")
            app.run(debug=True, host='0.0.0.0', port=5000)
        else:
            print("Flask server not started due to failure in loading recommender resources.")
