# pages/99_Config.py
import streamlit as st
import time 
st.set_page_config(
    page_title="Configuration",
    page_icon="⚙️",
    layout="centered"
)

st.title("⚙️ API Key Configuration")
st.write("Please enter your API Key to enable full functionality of the application.")
st.session_state.api_key_configured = False
# Use a form to collect the API key
with st.form("api_key_submission_form"):
    new_api_key = st.text_input(
        "Your API Key:",
        type="password",
        help="This key will be stored in your current browser session only and not permanently saved.",
        value=st.session_state.get("api_key", "") # Pre-fill if already set in session state
    )
    model= st.text_input("Model", value=st.session_state.get("model", "gemini-2.0-flash-001"))
    submit_button = st.form_submit_button("Save API Key")

if submit_button:
    if new_api_key:
        st.session_state.api_key = new_api_key
        st.session_state.model = model
        st.session_state.api_key_configured = True
        st.success("API Key saved for this session! You can now use the other pages.")
        time.sleep(2) # Give user time to read the message
        # TODO: Add option to go to back page(prev page)
        # Option 2: Suggest navigation (works for all Streamlit versions)
        st.info("👈 Please navigate to another page using the sidebar.")

    else:
        st.error("API Key cannot be empty. Please enter a valid key.")

# Display current status
if st.session_state.api_key_configured:
    st.write("Current API Key status: **Configured** (first 5 chars: `", st.session_state.api_key[:5], "...`)")
else:
    st.write("Current API Key status: **Not Configured**")

st.markdown("---")
st.write("⚠️ Your API key is stored only in your browser's session state and will be lost when you close the tab.")
st.write("For permanent storage, consider using `st.secrets` for your own deployment.")