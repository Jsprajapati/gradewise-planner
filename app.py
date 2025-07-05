# app.py
import streamlit as st
import pandas as pd
# Import the functions from your original script
from process import analyze_text, process_file_content

# st.set_page_config(layout="centered") # Or "wide" for more space
st.set_page_config(
    page_title="My Multi-Page App",
    page_icon="🏠",
    layout="centered"
)

st.title("Welcome to My Awesome App! 👋")
st.write("""
This is the main landing page.
Use the navigation in the sidebar to explore different sections of the application.
""")

st.info("👈 Select a page from the sidebar to begin!")

# Initialize session state for API key if not already present
if "api_key_configured" not in st.session_state:
    st.session_state.api_key_configured = False
if "api_key" not in st.session_state:
    st.session_state.api_key = ""

# --- Check for API key (developer's secret first, then user's input) ---
# 1. Try to load from st.secrets (for deployed apps / developer's default)
try:
    if st.secrets.get("API_KEY"):
        st.session_state.api_key = st.secrets["API_KEY"]
        st.session_state.api_key_configured = True
        st.success("API Key loaded from secrets! 🎉")
        st.info("You can now navigate to the other pages.")
except Exception:
    pass # st.secrets will raise an error if secrets.toml is missing or key not found. Handle silently.

# 2. If not from secrets, check if user already provided it in the current session
if st.session_state.api_key_configured:
    st.write("You are all set! Explore the app using the sidebar.")
else:
    # If API key is still not configured
    st.warning("No API Key detected. Please configure it on the 'Config' page.")
    # Provide a link/button to the config page or suggest sidebar navigation
    st.markdown("👈 Navigate to **'Config'** in the sidebar to set your API Key.")

