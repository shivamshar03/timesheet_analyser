import streamlit as st

# Set page configuration
st.set_page_config(page_title="Timesheet Upload", layout="wide")
st.title("📤 Upload Timesheet & Generate Report")

# File uploader for .txt files
uploaded_file = st.file_uploader("Upload your `.txt` timesheet file", type=["txt"])

if uploaded_file is not None:
    # Read the content of the uploaded file
    file_content = uploaded_file.getvalue().decode("utf-8")

    # Show the raw content in a text area
    edited_content = st.text_area("📄 File Preview", value=file_content, height=300)
    upload = st.button("Upload")
    if upload :
        st.session_state["uploaded_file_name"] = uploaded_file.name
        st.session_state["uploaded_file_bytes"] = edited_content
        st.success("✅ File uploaded and stored in session successfully.")

