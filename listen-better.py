# Read instructions here: https://x.com/burkov/status/1921303279562064098
# Make sure to install the Google AI SDK: pip install -q google-generativeai

import os
import json
import random
import time
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import google.generativeai as genai # For Google AI API
from dotenv import load_dotenv

# --- Configuration ---
load_dotenv()

SPOTIFY_CLIENT_ID = os.getenv("SPOTIPY_CLIENT_ID")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIPY_CLIENT_SECRET")
SPOTIFY_REDIRECT_URI = os.getenv("SPOTIPY_REDIRECT_URI")
GOOGLE_AI_API_KEY = os.getenv("GOOGLE_AI_API_KEY") # Using Google AI API Key

SCOPES = "user-library-read playlist-modify-public playlist-read-private playlist-read-collaborative"
NEW_PLAYLIST_NAME = "New AI Deep Cuts"
ALL_RECS_PLAYLIST_NAME = "All AI Deep Cuts History"

TARGET_NEW_SONGS_COUNT = 20
MAX_GEMINI_ATTEMPTS = 10
MAX_SONGS_TO_GEMINI_PROMPT = 150

# Model specified by user for Google AI API
GEMINI_MODEL = "gemini-2.5-flash-preview-04-17"
# If "gemini-2.5-flash-preview" doesn't work, try "gemini-1.5-flash-latest" or "gemini-1.5-pro-latest"

# --- New Configuration for Recommendation Quality ---
MAX_POPULARITY_THRESHOLD = 60
MIN_RELEASE_YEAR = 2021

FAVORED_STYLES_HINT = (
    "I'm looking for challenging, thoughtful, and inspiring music. Think along the lines of: "
    "experimental electronic, art pop, progressive rock, contemporary classical, avant-garde jazz, "
    "intelligent indie folk, post-rock, or songs with unique instrumentation, complex lyrical themes, "
    "and innovative soundscapes. Prioritize lesser-known artists, hidden gems, and newer releases. "
    "I appreciate music that pushes boundaries or offers a fresh perspective."
)
DISFAVORED_ELEMENTS_HINT = (
    "Please AVOID very mainstream, chart-topping pop, generic EDM, adult contemporary, "
    "or artists like Michael Bublé, Sade, Ed Sheeran, Maroon 5, or similar easy-listening, "
    "highly popular acts unless they have specific tracks that are genuinely experimental or "
    "fit the niche described in my favored styles. Avoid songs that are primarily TikTok trends, "
    "overly sentimental ballads, or background music. I want to be actively engaged by the music."
)

# --- Helper Functions ---

def get_spotify_client():
    auth_manager = SpotifyOAuth(
        client_id=SPOTIFY_CLIENT_ID,
        client_secret=SPOTIFY_CLIENT_SECRET,
        redirect_uri=SPOTIFY_REDIRECT_URI,
        scope=SCOPES,
        cache_path=".spotify_cache_token_info" # More specific cache file name
    )
    sp = spotipy.Spotify(auth_manager=auth_manager)
    print("Successfully authenticated with Spotify.")
    return sp

def get_all_liked_songs_details(sp):
    print("Fetching all liked songs details...")
    liked_songs_details = []
    offset = 0
    limit = 50
    while True:
        try:
            results = sp.current_user_saved_tracks(limit=limit, offset=offset)
            if not results or not results['items']:
                break
            for item in results['items']:
                track = item.get('track')
                if track and track.get('name') and track.get('artists'):
                    if track['artists']:
                        artist_name = track['artists'][0]['name']
                        liked_songs_details.append({
                            "track": track['name'],
                            "artist": artist_name,
                        })
            offset += limit
            print(f"Fetched {len(liked_songs_details)} liked songs so far...")
            if not results.get('next'):
                break
            time.sleep(0.05)
        except Exception as e:
            print(f"Error fetching liked songs page: {e}")
            break
    print(f"Total liked songs details fetched: {len(liked_songs_details)}")
    return liked_songs_details

def get_playlist_by_name(sp, playlist_name, user_id):
    playlists = sp.current_user_playlists(limit=50)
    while playlists:
        for playlist in playlists['items']:
            if playlist['name'] == playlist_name and playlist['owner']['id'] == user_id:
                return playlist
        if playlists['next']:
            playlists = sp.next(playlists)
            time.sleep(0.05)
        else:
            playlists = None
    return None

def get_or_create_playlist_id(sp, user_id, playlist_name, public=True):
    playlist_object = get_playlist_by_name(sp, playlist_name, user_id)
    if playlist_object:
        print(f"Found existing playlist: '{playlist_name}' (ID: {playlist_object['id']})")
        return playlist_object['id']
    else:
        print(f"Playlist '{playlist_name}' not found. Creating it...")
        try:
            new_playlist = sp.user_playlist_create(user=user_id, name=playlist_name, public=public)
            print(f"Successfully created playlist: '{playlist_name}' (ID: {new_playlist['id']})")
            return new_playlist['id']
        except Exception as e:
            print(f"Error creating playlist '{playlist_name}': {e}")
            return None

def get_playlist_tracks_simplified(sp, playlist_id):
    if not playlist_id: return []
    print(f"Fetching tracks from playlist ID: {playlist_id}...")
    playlist_tracks = []
    offset = 0
    limit = 100
    while True:
        try:
            results = sp.playlist_items(playlist_id, limit=limit, offset=offset,
                                        fields="items(track(id,name,artists(name))),next")
            if not results or not results['items']: break
            for item in results['items']:
                track_info = item.get('track')
                if track_info and track_info.get('id') and track_info.get('name') and track_info.get('artists'):
                    if track_info['artists']:
                        artist_name = track_info['artists'][0]['name']
                        playlist_tracks.append({
                            "track": track_info['name'],
                            "artist": artist_name,
                            "id": track_info['id']
                        })
            offset += limit
            print(f"Fetched {len(playlist_tracks)} tracks from playlist ID {playlist_id} so far...")
            if not results.get('next'): break
            time.sleep(0.05)
        except Exception as e:
            print(f"Error fetching playlist items for {playlist_id}: {e}")
            break
    print(f"Total tracks fetched from playlist ID {playlist_id}: {len(playlist_tracks)}")
    return playlist_tracks

def get_gemini_recommendations_google_ai(api_key, conversation_history_raw, model_name_to_use):
    """
    Sends the conversation history to Google AI Gemini and requests recommendations.
    Returns a tuple: (parsed_recommendations_list, raw_assistant_response_content_string)
    """
    print(f"\nSending request to Google AI ({model_name_to_use}) with {len(conversation_history_raw)} messages in history...")

    if not api_key:
        print("Error: Google AI API Key is not provided.")
        return [], None

    genai.configure(api_key=api_key)

    adapted_history = []
    for msg in conversation_history_raw:
        role = "model" if msg["role"] == "assistant" else msg["role"]
        adapted_history.append({'role': role, 'parts': [msg['content']]})

    if not adapted_history or adapted_history[-1]["role"] != "user":
        print("Error: Conversation history is empty or does not end with a user role message after adaptation.")
        return [], None

    try:
        # Initialize the model. You can set a default temperature here if you want,
        # but response_mime_type is best set per generate_content call for clarity and reliability.
        model = genai.GenerativeModel(
            model_name_to_use
            # You could still set a default temperature for the model instance if desired:
            # generation_config=genai.types.GenerationConfig(temperature=0.7)
        )

        # The entire adapted history is passed. The API handles the turns.
        # Pass the GenerationConfig with response_mime_type HERE:
        response = model.generate_content(
            adapted_history,
            generation_config=genai.types.GenerationConfig(
                temperature=0.7, # You can also set temperature here
                response_mime_type="application/json" # Crucial for JSON output
            ),
            request_options={"timeout": 120} # Increased timeout
        )

        raw_assistant_response_content = response.text

        recommendations = []
        try:
            parsed_content = json.loads(raw_assistant_response_content)
            if isinstance(parsed_content, list):
                recommendations = parsed_content
            elif isinstance(parsed_content, dict) and len(parsed_content.keys()) == 1:
                key = list(parsed_content.keys())[0]
                if isinstance(parsed_content[key], list):
                    recommendations = parsed_content[key]
                else:
                    print(f"Google AI returned a dictionary but the value under key '{key}' was not a list.")
            else:
                 print("Google AI response was JSON, but not in the expected list or single-key-list format.")

        except json.JSONDecodeError as e_json:
            print(f"Error: Google AI response could not be parsed as JSON: {e_json}")
            print(f"Google AI Raw Response Content:\n{raw_assistant_response_content}")
            return [], raw_assistant_response_content

        valid_recommendations = []
        for rec in recommendations:
            if isinstance(rec, dict) and "track" in rec and "artist" in rec:
                valid_recommendations.append({"track": str(rec["track"]), "artist": str(rec["artist"])})
            else:
                print(f"Warning: Skipping invalid recommendation format from Google AI: {rec}")

        print(f"Received {len(valid_recommendations)} validly structured recommendations from Google AI.")
        return valid_recommendations, raw_assistant_response_content

    except Exception as e:
        print(f"Error calling Google AI API with model {model_name_to_use}: {e}")
        if 'response' in locals() and hasattr(response, 'prompt_feedback'):
             print(f"Google AI Prompt Feedback: {response.prompt_feedback}")
        return [], None


def verify_and_filter_songs_on_spotify(sp, recommended_songs_details):
    print("\nVerifying Gemini recommendations on Spotify and applying filters...")
    enriched_songs_info = []
    for song_detail in recommended_songs_details:
        track_name = song_detail.get('track')
        artist_name = song_detail.get('artist')
        if not track_name or not artist_name:
            print(f"  Skipping malformed Gemini suggestion: {song_detail}")
            continue

        query = f"track:{track_name} artist:{artist_name}"
        try:
            results = sp.search(q=query, type="track", limit=1)
            time.sleep(0.05)
            if results and results['tracks']['items']:
                found_track = results['tracks']['items'][0]
                
                release_date_str = found_track.get('album', {}).get('release_date')
                release_year = None
                if release_date_str:
                    try:
                        release_year = int(release_date_str.split('-')[0])
                    except ValueError:
                        if len(release_date_str) == 4 and release_date_str.isdigit():
                            release_year = int(release_date_str)
                        else:
                            print(f"  Warning: Could not parse year from release_date: {release_date_str} for track {found_track['name']}")
                
                song_info = {
                    "uri": found_track['uri'],
                    "id": found_track['id'],
                    "track": found_track['name'],
                    "artist": found_track['artists'][0]['name'],
                    "popularity": found_track.get('popularity'),
                    "release_year": release_year
                }
                enriched_songs_info.append(song_info)
            else:
                print(f"  Spotify Lookup: Gemini suggestion '{track_name}' by '{artist_name}' not found on Spotify.")
        except Exception as e:
            print(f"  Error searching Spotify for '{track_name}' by '{artist_name}': {e}")
    
    print(f"Found {len(enriched_songs_info)} Gemini suggestions on Spotify. Now applying custom filters...")
    return enriched_songs_info


def update_playlist_items(sp, playlist_id, track_uris, replace=False):
    if not playlist_id: return False
    if not track_uris and not replace: return True
    if not track_uris and replace:
        try:
            sp.playlist_replace_items(playlist_id, [])
            print(f"Cleared all items from playlist ID {playlist_id}.")
            return True
        except Exception as e: print(f"Error clearing playlist {playlist_id}: {e}"); return False

    action = "Replacing items in" if replace else "Adding items to"
    print(f"{action} playlist ID {playlist_id} with {len(track_uris)} songs...")
    try:
        if replace:
            if len(track_uris) <= 100:
                 sp.playlist_replace_items(playlist_id, track_uris)
            else:
                sp.playlist_replace_items(playlist_id, [])
                for i in range(0, len(track_uris), 100):
                    sp.playlist_add_items(playlist_id, track_uris[i:i + 100])
                    if i + 100 < len(track_uris): time.sleep(0.2)
        else:
            for i in range(0, len(track_uris), 100):
                sp.playlist_add_items(playlist_id, track_uris[i:i + 100])
                if i + 100 < len(track_uris): time.sleep(0.2)
        print(f"Successfully updated playlist ID {playlist_id}.")
        return True
    except Exception as e:
        print(f"Error updating playlist {playlist_id}: {e}")
        return False

# --- Main Execution ---
if __name__ == "__main__":
    if not all([SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET, SPOTIFY_REDIRECT_URI, GOOGLE_AI_API_KEY]):
        print("Error: Missing environment variables (Spotify or Google AI). Please check .env file."); exit(1)

    sp_client = get_spotify_client()
    if not sp_client: exit(1)

    user_info = sp_client.me()
    user_id = user_info['id']
    print(f"Logged in as: {user_info.get('display_name', user_id)}")

    all_my_liked_songs_details = get_all_liked_songs_details(sp_client)
    if not all_my_liked_songs_details:
        print("No liked songs found. Exiting."); exit()
    
    all_my_liked_songs_set = set()
    for song_detail in all_my_liked_songs_details:
        track = song_detail.get('track', "").strip().lower()
        artist = song_detail.get('artist', "").strip().lower()
        if track and artist:
            all_my_liked_songs_set.add((track, artist))
    print(f"Created set of {len(all_my_liked_songs_set)} unique liked songs (name/artist) for de-duplication.")

    random.shuffle(all_my_liked_songs_details)
    sample_liked_songs_for_gemini_prompt = all_my_liked_songs_details[:MAX_SONGS_TO_GEMINI_PROMPT]

    all_recs_playlist_id = get_or_create_playlist_id(sp_client, user_id, ALL_RECS_PLAYLIST_NAME)
    existing_all_recs_songs_details = []
    if all_recs_playlist_id:
        existing_all_recs_songs_details = get_playlist_tracks_simplified(sp_client, all_recs_playlist_id)
    
    all_recs_history_track_ids_set = set()
    for song_detail in existing_all_recs_songs_details:
        if song_detail.get('id'):
            all_recs_history_track_ids_set.add(song_detail['id'])
    print(f"Found {len(all_recs_history_track_ids_set)} unique track IDs in '{ALL_RECS_PLAYLIST_NAME}' history.")

    collected_new_songs_for_playlist_uris = []
    collected_new_songs_for_playlist_details = [] 
    
    conversation_history = [] # Will store messages for Google AI API
    all_gemini_suggestions_this_session_raw_details = [] 

    liked_songs_prompt_str = "\n".join([f"- \"{s['track']}\" by {s['artist']}" for s in sample_liked_songs_for_gemini_prompt])
    
    initial_user_prompt_content = f"""You are a highly discerning music recommendation expert with a deep knowledge of diverse genres and a knack for unearthing hidden gems.
I'm looking for {TARGET_NEW_SONGS_COUNT} new song recommendations based on the following songs I already like.
My goal is to find music that is:
- Challenging: Not afraid to be experimental, complex, or unconventional.
- Thoughtful: Lyrically rich, thematically deep, or emotionally resonant in a non-superficial way.
- Inspiring: Music that sparks new ideas, emotions, or perspectives.
- NEW: Prioritize songs released in the last few years (ideally from {MIN_RELEASE_YEAR or 'any year'} onwards, if possible) and by lesser-known or emerging artists. I want to avoid tracks that are already massively popular (e.g., popularity score above {MAX_POPULARITY_THRESHOLD or 'any score'} on a 0-100 scale on Spotify).

Please consider these preferences: {FAVORED_STYLES_HINT}
And try to avoid these elements: {DISFAVORED_ELEMENTS_HINT}

Your response MUST be ONLY a valid JSON array of objects. Each object must have a "track" key (song title) and an "artist" key (artist name).
Do not include any other text, explanations, or markdown formatting outside of the JSON array.
Ensure variety in the artists you recommend in this batch; try not to recommend multiple songs by the same artist unless they are exceptionally fitting.

Here are some songs I like:
{liked_songs_prompt_str}

Please provide {TARGET_NEW_SONGS_COUNT} new song recommendations in the specified JSON format, keeping all the above criteria in mind."""
    # For Google AI, the first message is "user"
    conversation_history.append({"role": "user", "content": initial_user_prompt_content})


    for attempt in range(MAX_GEMINI_ATTEMPTS):
        if len(collected_new_songs_for_playlist_uris) >= TARGET_NEW_SONGS_COUNT:
            print("\nTarget number of new songs reached.")
            break

        print(f"\n--- Gemini Request Attempt {attempt + 1}/{MAX_GEMINI_ATTEMPTS} ---")
        
        # The conversation_history already has the user prompt from the previous iteration or the initial one.
        # If it's not the first attempt, the last message in history is the model's previous response.
        # We need to add a NEW user prompt for the follow-up.
        if attempt > 0:
            songs_suggested_by_gemini_this_session_str = "\n".join(
                [f"- \"{s['track']}\" by {s['artist']}" for s in all_gemini_suggestions_this_session_raw_details]
            )
            if not songs_suggested_by_gemini_this_session_str:
                songs_suggested_by_gemini_this_session_str = "(None previously suggested in this session)"

            follow_up_user_prompt_content = f"""Thank you. Now, please provide {TARGET_NEW_SONGS_COUNT} MORE unique song recommendations.
It's crucial that these new recommendations are different from any songs you've already suggested to me in this conversation. For reference, here are the songs you've suggested so far (please avoid these entirely):
{songs_suggested_by_gemini_this_session_str}

Also, ensure these new recommendations are different from the initial list of liked songs I provided (at the very start of our conversation).
Remember my core preferences for music that is challenging, thoughtful, inspiring, and NEW (ideally released {MIN_RELEASE_YEAR or 'any year'} or later, by lesser-known or emerging artists, and not overly popular - e.g. Spotify popularity < {MAX_POPULARITY_THRESHOLD or 'any score'}).
My style preferences: {FAVORED_STYLES_HINT}
Elements to avoid: {DISFAVORED_ELEMENTS_HINT}
Ensure artist variety in this new batch.

Your response must be ONLY a valid JSON array of objects, with "track" and "artist" keys, as before."""
            conversation_history.append({"role": "user", "content": follow_up_user_prompt_content})

        gemini_batch_recs_parsed, raw_model_response_str = get_gemini_recommendations_google_ai(
            GOOGLE_AI_API_KEY,
            conversation_history, # Pass the current history
            GEMINI_MODEL
        )

        if raw_model_response_str:
            # Add model's response to history for the next turn
            conversation_history.append({"role": "model", "content": raw_model_response_str})
        
        if not gemini_batch_recs_parsed:
            print("Gemini returned no validly structured recommendations or there was an API error.")
            if attempt < MAX_GEMINI_ATTEMPTS - 1: time.sleep(5)
            continue
        
        for rec in gemini_batch_recs_parsed:
            all_gemini_suggestions_this_session_raw_details.append(rec)
        
        enriched_spotify_songs_this_batch = verify_and_filter_songs_on_spotify(sp_client, gemini_batch_recs_parsed)
        
        newly_added_this_turn_count = 0
        for song_info in enriched_spotify_songs_this_batch:
            if len(collected_new_songs_for_playlist_uris) >= TARGET_NEW_SONGS_COUNT:
                break

            spotify_track_name_lower = song_info['track'].lower()
            spotify_artist_name_lower = song_info['artist'].lower()
            
            is_liked = (spotify_track_name_lower, spotify_artist_name_lower) in all_my_liked_songs_set
            is_in_all_recs_playlist_history = song_info['id'] in all_recs_history_track_ids_set
            is_already_collected_this_session = any(
                s['id'] == song_info['id'] for s in collected_new_songs_for_playlist_details
            )
            is_too_popular = False
            if MAX_POPULARITY_THRESHOLD is not None and song_info['popularity'] is not None:
                if song_info['popularity'] > MAX_POPULARITY_THRESHOLD:
                    is_too_popular = True
            is_too_old = False
            if MIN_RELEASE_YEAR is not None and song_info['release_year'] is not None:
                if song_info['release_year'] < MIN_RELEASE_YEAR:
                    is_too_old = True

            if not is_liked and \
               not is_in_all_recs_playlist_history and \
               not is_already_collected_this_session and \
               not is_too_popular and \
               not is_too_old:
                
                collected_new_songs_for_playlist_uris.append(song_info['uri'])
                collected_new_songs_for_playlist_details.append(song_info)
                newly_added_this_turn_count +=1
                print(f"  ++ COLLECTED: '{song_info['track']}' by {song_info['artist']} (Pop: {song_info['popularity']}, Year: {song_info['release_year']})")
            else:
                reasons = []
                if is_liked: reasons.append("is liked")
                if is_in_all_recs_playlist_history: reasons.append("in history playlist")
                if is_already_collected_this_session: reasons.append("already collected this session")
                if is_too_popular: reasons.append(f"too popular (Pop: {song_info['popularity']} > {MAX_POPULARITY_THRESHOLD})")
                if is_too_old: reasons.append(f"too old (Year: {song_info['release_year']} < {MIN_RELEASE_YEAR})")
                print(f"  -- SKIPPED: '{song_info['track']}' by {song_info['artist']} (Pop: {song_info['popularity']}, Year: {song_info['release_year']}). Reasons: {', '.join(reasons) or 'unknown'}")

        print(f"Added {newly_added_this_turn_count} new songs this turn.")
        print(f"Total collected for new playlist so far: {len(collected_new_songs_for_playlist_uris)}/{TARGET_NEW_SONGS_COUNT}")
        
        if len(collected_new_songs_for_playlist_uris) >= TARGET_NEW_SONGS_COUNT:
            break 
        elif attempt < MAX_GEMINI_ATTEMPTS -1 :
            time.sleep(3)

    final_uris_for_new_playlist = collected_new_songs_for_playlist_uris[:TARGET_NEW_SONGS_COUNT]
    final_details_for_all_recs_update = collected_new_songs_for_playlist_details[:TARGET_NEW_SONGS_COUNT]

    if not final_uris_for_new_playlist:
        print("\nNo new, suitable songs were collected from Gemini after all attempts. Exiting.")
        exit()

    print(f"\nCollected {len(final_uris_for_new_playlist)} final new songs for '{NEW_PLAYLIST_NAME}'.")
    for i, song in enumerate(final_details_for_all_recs_update):
        print(f"  {i+1}. \"{song['track']}\" by {song['artist']} (Pop: {song['popularity']}, Year: {song['release_year']})")

    new_playlist_id = get_or_create_playlist_id(sp_client, user_id, NEW_PLAYLIST_NAME)
    if new_playlist_id:
        print(f"\nUpdating playlist '{NEW_PLAYLIST_NAME}' by replacing items...")
        if update_playlist_items(sp_client, new_playlist_id, final_uris_for_new_playlist, replace=True):
            playlist_url_new = sp_client.playlist(new_playlist_id)['external_urls']['spotify']
            print(f"Successfully updated '{NEW_PLAYLIST_NAME}'. URL: {playlist_url_new}")
    else:
        print(f"Could not create or find playlist '{NEW_PLAYLIST_NAME}'.")

    if all_recs_playlist_id and final_details_for_all_recs_update:
        uris_to_add_to_all_recs = [song['uri'] for song in final_details_for_all_recs_update]
        print(f"\nAppending {len(uris_to_add_to_all_recs)} songs to '{ALL_RECS_PLAYLIST_NAME}'...")
        if update_playlist_items(sp_client, all_recs_playlist_id, uris_to_add_to_all_recs, replace=False):
            playlist_url_all = sp_client.playlist(all_recs_playlist_id)['external_urls']['spotify']
            print(f"Successfully appended songs to '{ALL_RECS_PLAYLIST_NAME}'. URL: {playlist_url_all}")
    elif not all_recs_playlist_id:
         print(f"Could not find or create playlist '{ALL_RECS_PLAYLIST_NAME}' to append songs.")

    print("\nScript finished.")
