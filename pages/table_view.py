import streamlit as st
import pandas as pd

st.set_page_config(page_title="📊 Timesheet Tables", layout="wide")
st.title("📄 View Timesheet as Table")

# Check for uploaded file from main page
if "uploaded_file_bytes" not in st.session_state:
    st.warning("Please upload a timesheet file from the Upload & Analyze page first.")
    st.stop()

# Decode the uploaded content
timesheet_data = st.session_state["uploaded_file_bytes"].decode("utf-8")

# Attempt to extract and parse visualization CSV data
try:
    csv_start = timesheet_data.index("--RAW DATA FOR VISUALIZATION--") + len("--RAW DATA FOR VISUALIZATION--")
    csv_end = timesheet_data.index("--END OF DATA--")
    raw_csv = timesheet_data[csv_start:csv_end].strip()

    from io import StringIO
    df = pd.read_csv(StringIO(raw_csv))

    st.subheader("👥 Employee Performance Table")
    st.dataframe(df, use_container_width=True)

    st.markdown("---")
    st.subheader("📈 Summary Statistics")
    st.write(df.describe(include='all'))

except Exception as e:
    st.error(f"❌ Could not parse tabular data: {e}")
