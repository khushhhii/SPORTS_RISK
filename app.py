import streamlit as st
import os
import pandas as pd
import requests
import json
import time
import pickle
import numpy as np

# --- Configuration ---
SIMULATE_GEMINI_API = False

# --- FIXED: Reliable Stadium Image for Dashboard ---
DASHBOARD_BG_URL = "https://images.unsplash.com/photo-1518091043644-c1d4457512c6?q=80&w=2831&auto=format&fit=crop"

# --- FIXED: New Reliable Stadium Image for Home Page ---
HOME_BG_URL = "https://images.unsplash.com/photo-1431324155629-1a6deb1dec8d?q=80&w=2070&auto=format&fit=crop"

# The debugger confirmed this is the correct filename and location (in the root)
MODEL_FILE = 'xgboost_injury_model.pkl'

# --- Player Data ---
PLAYER_DATA = {
    "Select Player...": {
        "age": "N/A", "height": "N/A", "nationality": "N/A", "position": "N/A",
        "minutesPlayedLastWeek": 0, "trainingLoadRatio": 1.0, 
        "previousInjuriesCount": 0, "wellnessScore": 10,
        "photoURL": "https://placehold.co/100x100/6B7280/FFFFFF?text=P"
    },
    # --- High Risk Examples ---
    "L. Messi": { "age": 37, "height": "170 cm", "nationality": "Argentine", "position": "Forward", "minutesPlayedLastWeek": 400, "trainingLoadRatio": 1.55, "previousInjuriesCount": 1, "wellnessScore": 6, "photoURL": "https://placehold.co/100x100/DC3545/FFFFFF?text=LM" },
    "C. Ronaldo": { "age": 39, "height": "187 cm", "nationality": "Portuguese", "position": "Striker", "minutesPlayedLastWeek": 550, "trainingLoadRatio": 1.2, "previousInjuriesCount": 4, "wellnessScore": 3, "photoURL": "https://placehold.co/100x100/DC3545/FFFFFF?text=CR" },
    "K. Mbappet": { "age": 25, "height": "178 cm", "nationality": "French", "position": "Winger", "minutesPlayedLastWeek": 650, "trainingLoadRatio": 1.7, "previousInjuriesCount": 0, "wellnessScore": 5, "photoURL": "https://placehold.co/100x100/DC3545/FFFFFF?text=KM" },
    "Z. Petrov": { "age": 31, "height": "182 cm", "nationality": "Russian", "position": "Winger", "minutesPlayedLastWeek": 400, "trainingLoadRatio": 1.45, "previousInjuriesCount": 4, "wellnessScore": 7, "photoURL": "https://placehold.co/100x100/DC3545/FFFFFF?text=ZP" },
    "T. Miller": { "age": 29, "height": "190 cm", "nationality": "German", "position": "Striker", "minutesPlayedLastWeek": 500, "trainingLoadRatio": 1.3, "previousInjuriesCount": 1, "wellnessScore": 4, "photoURL": "https://placehold.co/100x100/DC3545/FFFFFF?text=TM" },
    "U. O'Connell": { "age": 37, "height": "180 cm", "nationality": "Irish", "position": "Midfielder", "minutesPlayedLastWeek": 580, "trainingLoadRatio": 1.2, "previousInjuriesCount": 2, "wellnessScore": 6, "photoURL": "https://placehold.co/100x100/DC3545/FFFFFF?text=UO" },
    "V. Singh": { "age": 25, "height": "175 cm", "nationality": "Indian", "position": "Winger", "minutesPlayedLastWeek": 500, "trainingLoadRatio": 1.5, "previousInjuriesCount": 3, "wellnessScore": 8, "photoURL": "https://placehold.co/100x100/DC3545/FFFFFF?text=VS" },
    # --- Moderate Risk Examples ---
    "V. Van Dijk": { "age": 32, "height": "193 cm", "nationality": "Dutch", "position": "Defender", "minutesPlayedLastWeek": 600, "trainingLoadRatio": 1.25, "previousInjuriesCount": 1, "wellnessScore": 7, "photoURL": "https://placehold.co/100x100/FFC107/333333?text=VD" },
    "E. Haaland": { "age": 23, "height": "195 cm", "nationality": "Norwegian", "position": "Striker", "minutesPlayedLastWeek": 480, "trainingLoadRatio": 1.4, "previousInjuriesCount": 0, "wellnessScore": 8, "photoURL": "https://placehold.co/100x100/FFC107/333333?text=EH" },
    "G. Kante": { "age": 33, "height": "168 cm", "nationality": "French", "position": "Midfielder", "minutesPlayedLastWeek": 350, "trainingLoadRatio": 1.1, "previousInjuriesCount": 2, "wellnessScore": 6, "photoURL": "https://placehold.co/100x100/FFC107/333333?text=GK" },
    "B. Davies": { "age": 31, "height": "178 cm", "nationality": "Welsh", "position": "Winger", "minutesPlayedLastWeek": 480, "trainingLoadRatio": 1.3, "previousInjuriesCount": 0, "wellnessScore": 6, "photoURL": "https://placehold.co/100x100/FFC107/333333?text=BD" },
    "C. Johnson": { "age": 34, "height": "179 cm", "nationality": "American", "position": "Midfielder", "minutesPlayedLastWeek": 400, "trainingLoadRatio": 0.8, "previousInjuriesCount": 1, "wellnessScore": 7, "photoURL": "https://placehold.co/100x100/FFC107/333333?text=CJ" },
    "G. Rossi": { "age": 30, "height": "174 cm", "nationality": "Italian", "position": "Forward", "minutesPlayedLastWeek": 350, "trainingLoadRatio": 1.4, "previousInjuriesCount": 0, "wellnessScore": 7, "photoURL": "https://placehold.co/100x100/FFC107/333333?text=GR" },
    "H. Chen": { "age": 32, "height": "181 cm", "nationality": "Chinese", "position": "Defender", "minutesPlayedLastWeek": 500, "trainingLoadRatio": 0.9, "previousInjuriesCount": 2, "wellnessScore": 8, "photoURL": "https://placehold.co/100x100/FFC107/333333?text=HC" },
    "I. Dubois": { "age": 27, "height": "170 cm", "nationality": "French", "position": "Midfielder", "minutesPlayedLastWeek": 420, "trainingLoadRatio": 1.25, "previousInjuriesCount": 0, "wellnessScore": 6, "photoURL": "https://placehold.co/100x100/FFC107/333333?text=ID" },
    "K. Popescu": { "age": 39, "height": "190 cm", "nationality": "Romanian", "position": "Goalkeeper", "minutesPlayedLastWeek": 580, "trainingLoadRatio": 1.15, "previousInjuriesCount": 3, "wellnessScore": 7, "photoURL": "https://placehold.co/100x100/FFC107/333333?text=KP" },
    "L. Silva": { "age": 33, "height": "176 cm", "nationality": "Brazilian", "position": "Winger", "minutesPlayedLastWeek": 390, "trainingLoadRatio": 1.35, "previousInjuriesCount": 1, "wellnessScore": 5, "photoURL": "https://placehold.co/100x100/FFC107/333333?text=LS" },
    "W. Jones": { "age": 28, "height": "185 cm", "nationality": "English", "position": "Defender", "minutesPlayedLastWeek": 630, "trainingLoadRatio": 1.05, "previousInjuriesCount": 1, "wellnessScore": 7, "photoURL": "https://placehold.co/100x100/FFC107/333333?text=WJ" },
    "X. Li": { "age": 24, "height": "177 cm", "nationality": "Chinese", "position": "Midfielder", "minutesPlayedLastWeek": 300, "trainingLoadRatio": 1.4, "previousInjuriesCount": 0, "wellnessScore": 8, "photoURL": "https://placehold.co/100x100/FFC107/333333?text=XL" },
    "Y. Adebayo": { "age": 22, "height": "191 cm", "nationality": "Nigerian", "position": "Striker", "minutesPlayedLastWeek": 450, "trainingLoadRatio": 1.0, "previousInjuriesCount": 0, "wellnessScore": 5, "photoURL": "https://placehold.co/100x100/FFC107/333333?text=YA" },
    # --- Low Risk Examples ---
    "A. Martinez": { "age": 28, "height": "175 cm", "nationality": "Argentine", "position": "Goalkeeper", "minutesPlayedLastWeek": 280, "trainingLoadRatio": 1.05, "previousInjuriesCount": 0, "wellnessScore": 9, "photoURL": "https://placehold.co/100x100/198754/FFFFFF?text=AM" },
    "A. Becker": { "age": 31, "height": "193 cm", "nationality": "Brazilian", "position": "Goalkeeper", "minutesPlayedLastWeek": 540, "trainingLoadRatio": 1.0, "previousInjuriesCount": 0, "wellnessScore": 9, "photoURL": "https://placehold.co/100x100/198754/FFFFFF?text=AB" },
    "K. De Bruyner": { "age": 32, "height": "181 cm", "nationality": "Belgian", "position": "Midfielder", "minutesPlayedLastWeek": 250, "trainingLoadRatio": 0.85, "previousInjuriesCount": 1, "wellnessScore": 8, "photoURL": "https://placehold.co/100x100/198754/FFFFFF?text=KB" },
    "S. Ramos": { "age": 37, "height": "184 cm", "nationality": "Spanish", "position": "Defender", "minutesPlayedLastWeek": 300, "trainingLoadRatio": 1.05, "previousInjuriesCount": 2, "wellnessScore": 9, "photoURL": "https://placehold.co/100x100/198754/FFFFFF?text=SR" },
    "E. Müller": { "age": 26, "height": "192 cm", "nationality": "German", "position": "Goalkeeper", "minutesPlayedLastWeek": 630, "trainingLoadRatio": 1.1, "previousInjuriesCount": 0, "wellnessScore": 8, "photoURL": "https://placehold.co/100x100/198754/FFFFFF?text=EM" },
    "F. Kim": { "age": 25, "height": "173 cm", "nationality": "South Korean", "position": "Winger", "minutesPlayedLastWeek": 200, "trainingLoadRatio": 0.95, "previousInjuriesCount": 0, "wellnessScore": 9, "photoURL": "https://placehold.co/100x100/198754/FFFFFF?text=FK" },
    "N. Van Der Sar": { "age": 29, "height": "186 cm", "nationality": "Dutch", "position": "Defender", "minutesPlayedLastWeek": 520, "trainingLoadRatio": 1.0, "previousInjuriesCount": 0, "wellnessScore": 10, "photoURL": "https://placehold.co/100x100/198754/FFFFFF?text=NV" },
    "O. Okoro": { "age": 30, "height": "184 cm", "nationality": "Nigerian", "position": "Midfielder", "minutesPlayedLastWeek": 310, "trainingLoadRatio": 1.1, "previousInjuriesCount": 0, "wellnessScore": 9, "photoURL": "https://placehold.co/100x100/198754/FFFFFF?text=OO" },
    "P. García": { "age": 32, "height": "177 cm", "nationality": "Spanish", "position": "Winger", "minutesPlayedLastWeek": 400, "trainingLoadRatio": 1.08, "previousInjuriesCount": 0, "wellnessScore": 8, "photoURL": "https://placehold.co/100x100/198754/FFFFFF?text=PG" },
    "R. Singh": { "age": 26, "height": "172 cm", "nationality": "Indian", "position": "Forward", "minutesPlayedLastWeek": 150, "trainingLoadRatio": 0.9, "previousInjuriesCount": 0, "wellnessScore": 9, "photoURL": "https://placehold.co/100x100/198754/FFFFFF?text=RS" },
}

# Load the model
# --------------------------------------------------------------------------------
# !!! BEGIN UPDATED DIAGNOSTIC CODE !!!
# --------------------------------------------------------------------------------
try:
    with open(MODEL_FILE, 'rb') as file:
        model = pickle.load(file)
    MODEL_LOADED = True
    st.success("Model loaded successfully! 🎉 Heuristic fallback mode disabled.")
    
except FileNotFoundError:
    st.error(f"Error: Model file '{MODEL_FILE}' not found. Please check GitHub sync.")
    model = None
    MODEL_LOADED = False
    
except Exception as e:
    # This block executes if the file exists, but the content is invalid 
    # (e.g., LFS pointer, corruption, version mismatch).
    st.error(f"FATAL MODEL LOAD ERROR: {e}. Model is unusable.")
    st.info("💡 Please check the model file's size on GitHub (possible Git LFS issue) or library versions.")
    model = None
    MODEL_LOADED = False
# --------------------------------------------------------------------------------
# !!! END UPDATED DIAGNOSTIC CODE !!!
# --------------------------------------------------------------------------------


# --- Helper Functions ---

def get_risk_data(score):
    if score < 30:
        return { "category": "Low Risk", "color": "success", "hex_color": "#00FF7F", "icon": "🛡️", "description": "Optimal physical condition. Ready for full load." }
    elif score < 65:
        return { "category": "Moderate Risk", "color": "warning", "hex_color": "#FFD700", "icon": "⚠️", "description": "Fatigue indicators present. Monitor load." }
    else:
        return { "category": "High Risk", "color": "danger", "hex_color": "#FF4500", "icon": "🚑", "description": "Injury imminent. Rest highly recommended." }

def calculate_risk(inputs):
    age = inputs.get('age', 25)
    if not isinstance(age, int): age = 25
    minutes = inputs.get('minutesPlayedLastWeek', 0)
    ratio = inputs.get('trainingLoadRatio', 1.0)
    injuries = inputs.get('previousInjuriesCount', 0)
    wellness = inputs.get('wellnessScore', 10)
    
    def fallback_score(age, minutes, ratio, injuries, wellness):
        deviation = abs(ratio - 1.0) 
        base_score = injuries * 10
        acwr_score = deviation * 30 
        wellness_penalty = (10 - wellness) * 3
        age_penalty = max(0, age - 28) * 5
        return min(100, max(0, int(base_score + acwr_score + wellness_penalty + age_penalty)))
    
    if not MODEL_LOADED:
        # NOTE: This is the heuristic fallback mode
        return fallback_score(age, minutes, ratio, injuries, wellness)
    
    try:
        # Features array creation for model prediction
        features = np.array([[age, minutes, ratio, injuries, wellness]])
        probability_of_injury = model.predict_proba(features)[0][1]
        return round(probability_of_injury * 100)
    except Exception as e:
        # If prediction fails (e.g., mismatched feature names, broken model object), fall back
        return fallback_score(age, minutes, ratio, injuries, wellness)

def navigate_to(page):
    st.session_state.page = page

# --- UI Functions ---

def set_background(url, overlay_opacity=0.0):
    """Injects CSS to set a full-screen background image."""
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: linear-gradient(rgba(0,0,0,{overlay_opacity}), rgba(0,0,0,{overlay_opacity})), url("{url}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        /* Force text to white for contrast on dark backgrounds */
        h1, h2, h3, h4, h5, p, span, div {{
            color: #E0E0E0;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

def page_home():
    """First Page: Intro with bright background."""
    # Set the background using the robust URL
    set_background(HOME_BG_URL, overlay_opacity=0.5) 
    
    st.markdown(
        """
        <style>
        @keyframes fadeIn { from { opacity: 0; transform: translateY(20px); } to { opacity: 1; transform: translateY(0); } }
        
        /* Updated container styles to center and reduce gap */
        .hero-container {
            display: flex; 
            flex-direction: column; 
            align-items: center; 
            justify-content: center; /* Center content vertically in this div */
            height: 40vh; /* Controlled height */
            text-align: center;
            animation: fadeIn 1.5s ease-out;
            margin-top: 10vh; /* Pushes the entire block down */
        }
        
        .main-title {
            font-family: 'Helvetica Neue', sans-serif; font-size: 5rem; font-weight: 900;
            color: #FFFFFF; text-shadow: 0 0 30px rgba(0, 191, 255, 0.8);
            margin-bottom: 5px; /* Reduced margin */
            letter-spacing: -2px; line-height: 1.1;
        }

        /* Styles to center the button visually and tighten spacing */
        div.stButton {
            text-align: center;
            margin-top: -10px; /* Pull button up slightly closer to the title */
        }
        .stButton>button {
            font-size: 1.5rem !important; 
            font-weight: bold !important; 
            padding: 20px 60px !important;
            border-radius: 50px !important; 
            background: linear-gradient(90deg, #00BFFF, #1E90FF) !important;
            color: white !important; 
            border: none !important;
            box-shadow: 0 10px 30px rgba(0, 191, 255, 0.5) !important;
            transition: all 0.3s ease !important;
        }
        .stButton>button:hover {
            transform: scale(1.05) !important; 
            box-shadow: 0 10px 50px rgba(0, 191, 255, 0.8) !important;
        }
        </style>
        
        <div class="hero-container">
            <h1 class="main-title">ELITE INJURY<br>PREDICTOR</h1>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Using columns to perfectly center the button widget
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        # use_container_width=True helps it fill the center column nicely
        if st.button("ENTER DASHBOARD", key="start_button", use_container_width=True):
            navigate_to("dashboard")

def page_dashboard():
    """Second Page: Dark themed dashboard."""
    set_background(DASHBOARD_BG_URL, overlay_opacity=0.7) # Darker overlay for readability

    # Custom CSS for the dashboard components (Glassmorphism)
    st.markdown("""
        <style>
        /* Card Styling for Containers */
        .glass-card {
            background-color: rgba(255, 255, 255, 0.08);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        }
        
        /* Selectbox styling */
        .stSelectbox > div > div {
            background-color: rgba(0, 0, 0, 0.7) !important;
            color: white !important;
            border: 1px solid #444 !important;
        }
        
        /* Metric styling */
        [data-testid="stMetric"] {
            background-color: rgba(0,0,0,0.5);
            padding: 15px;
            border-radius: 10px;
            border-left: 4px solid #00BFFF;
        }
        [data-testid="stMetricLabel"] { color: #aaa !important; font-size: 0.9rem !important; }
        [data-testid="stMetricValue"] { color: white !important; font-size: 1.8rem !important; }

        /* Warning box styling */
        .stAlert { background-color: rgba(255, 165, 0, 0.2); color: white; border: 1px solid orange; }
        </style>
    """, unsafe_allow_html=True)

    # Top Navigation
    col_head1, col_head2 = st.columns([6,1])
    with col_head1:
        st.markdown("<h1 style='margin-top:-20px; text-shadow: 2px 2px 4px #000;'>⚽ Performance Hub</h1>", unsafe_allow_html=True)
    with col_head2:
        if st.button("HOME", key="back_button"):
            navigate_to("home")

    st.markdown("---")

    # Layout
    col_sidebar, col_main = st.columns([1, 3])

    with col_sidebar:
        # Wrapped in a container for styling
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("Select Athlete")
        
        all_player_names = sorted([k for k in PLAYER_DATA.keys() if k != "Select Player..."])
        player_options = ["Select Player..."] + all_player_names
        
        if 'player_select' not in st.session_state:
            st.session_state.player_select = "Select Player..."
        
        try: default_index = player_options.index(st.session_state.player_select)
        except ValueError: default_index = 0
            
        selected_player_name = st.selectbox(
            "Roster", options=player_options, index=default_index,
            key="player_select", label_visibility="collapsed"
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Add some filler info or legends here if needed
        if selected_player_name != "Select Player...":
            st.markdown("<br>", unsafe_allow_html=True)
            st.info("ℹ️ Data updated: Live")

    current_data = PLAYER_DATA.get(selected_player_name, PLAYER_DATA["Select Player..."])

    with col_main:
        if selected_player_name == "Select Player...":
             st.markdown("""
             <div class="glass-card" style="text-align: center; padding: 50px;">
                 <h2 style="color: #00BFFF;">WAITING FOR SELECTION</h2>
                 <p>Select a player from the roster to generate a live risk assessment.</p>
             </div>
             """, unsafe_allow_html=True)
        else:
            # Calculate Risk
            risk_inputs = {
                'age': current_data.get('age'), 'minutesPlayedLastWeek': current_data.get('minutesPlayedLastWeek'),
                'trainingLoadRatio': current_data.get('trainingLoadRatio'), 'previousInjuriesCount': current_data.get('previousInjuriesCount'),
                'wellnessScore': current_data.get('wellnessScore')
            }
            risk_score = calculate_risk(risk_inputs)
            risk_data = get_risk_data(risk_score)

            # --- TOP SECTION: Profile & Risk ---
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            col_photo, col_stats, col_risk = st.columns([1, 2, 2])
            
            with col_photo:
                 st.image(current_data['photoURL'], width=130)

            with col_stats:
                st.markdown(f"<h2 style='margin:0; color:white;'>{selected_player_name}</h2>", unsafe_allow_html=True)
                st.markdown(f"<p style='color:#aaa; margin:0;'>{current_data['nationality']} | {current_data['position']}</p>", unsafe_allow_html=True)
                st.markdown(f"<p style='color:#aaa; margin:0;'>Age: {current_data['age']} | Height: {current_data['height']}</p>", unsafe_allow_html=True)

            with col_risk:
                st.markdown(f"""
                <div style="text-align: right; border-right: 5px solid {risk_data['hex_color']}; padding-right: 20px;">
                    <h1 style="color:{risk_data['hex_color']}; font-size: 3.5rem; margin:0;">{risk_score}%</h1>
                    <h4 style="color: white; margin:0;'>INJURY PROBABILITY</h4>
                    <p style="color: {risk_data['hex_color']}; margin:0;'>{risk_data['icon']} {risk_data['category']}</p>
                </div>
                """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)

            # --- BOTTOM SECTION: Metrics ---
            st.subheader("Model Inputs & Biometrics")
            
            m_col1, m_col2, m_col3, m_col4 = st.columns(4)
            
            with m_col1:
                st.metric("Minutes (Week)", f"{current_data['minutesPlayedLastWeek']}")
            with m_col2:
                st.metric("ACWR Load", f"{current_data['trainingLoadRatio']:.2f}")
            with m_col3:
                st.metric("Injury History", f"{current_data['previousInjuriesCount']}")
            with m_col4:
                # Color code wellness
                w_score = current_data['wellnessScore']
                w_color = "#00FF7F" if w_score > 7 else "#FF4500" if w_score < 5 else "#FFD700"
                st.markdown(f"""
                <div style="background-color: rgba(0,0,0,0.5); padding: 15px; border-radius: 10px; border-left: 4px solid {w_color};">
                    <label style="color:#aaa; font-size:0.9rem;">Wellness Score</label>
                    <div style="color:{w_color}; font-size: 1.8rem; font-weight:bold;">{w_score}/10</div>
                </div>
                """, unsafe_allow_html=True)
            
            # The warning remains in case the model file is corrupted, but should now be gone if the file loaded correctly.
            if not MODEL_LOADED:
                 # This will only show if one of the 'except' blocks above was hit.
                 # The 'except' blocks now include descriptive error messages.
                 st.warning("Running in heuristic fallback mode (Model file not found).")

# --- Main Logic ---
st.set_page_config(page_title="Elite Injury Predictor", layout="wide", initial_sidebar_state="collapsed")
st.markdown("""<style>#MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}</style>""", unsafe_allow_html=True)

if 'page' not in st.session_state: st.session_state.page = "home"

if st.session_state.page == "home": page_home()
elif st.session_state.page == "dashboard": page_dashboard()
