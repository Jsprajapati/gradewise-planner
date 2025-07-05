# pages/2_Data_Input.py
import streamlit as st
# from your_backend_script import process_data # Assuming you have a function here

st.set_page_config(page_title="Syllabi Gen", page_icon="⚙️")

st.title("⚙️ Generate Syllabus using AI")

st.write("Upload a file or enter data to process with your Python backend script.")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### Data Preview:")
    st.dataframe(df.head())

    # Example of calling your backend script's function
    if st.button("Process Data"):
        with st.spinner("Processing..."):
            # Replace with your actual script's function call
            # processed_results = process_data(df)
            st.success("Data processed! (Placeholder)")
            st.write("Here would be the results from your backend script.")
            # st.write(processed_results) # Display actual results

st.text_input("Subject title")
text_to_process = st.session_state.get('text_to_process', '') # Using session_state for persistence

if st.button("Submit Text"):
    if text_to_process:
        # Replace with your actual script's function call
        # text_analysis_results = analyze_text(text_to_process)
        st.success(f"Text '{text_to_process}' submitted! (Placeholder)")
        # st.write(text_analysis_results)
    else:
        st.warning("Please enter some text.")

# Update session state on input change
def update_text_input():
    st.session_state.text_to_process = st.session_state.text_input_widget

st.text_input("Type something here:", key='text_input_widget', on_change=update_text_input)