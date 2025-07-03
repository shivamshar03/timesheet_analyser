import streamlit as st
import pandas as pd
import os
import re
from io import StringIO
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_groq import ChatGroq

# Streamlit setup
st.set_page_config(page_title="📊 Timesheet Dashboard", layout="wide")
st.title("📄 Developer Timesheet Analyzer")

# Check if file is uploaded via session state
if "uploaded_file_bytes" not in st.session_state:
    st.warning("Please upload a raw timesheet file on the Upload & Analyze page first.")
    st.stop()

# Load uploaded raw timesheet text
timesheet_data = st.session_state["uploaded_file_bytes"]

# Load Groq API key
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# Initialize Groq LLM
llm = ChatGroq(model="llama3-8b-8192", api_key=groq_api_key)

# Strict Prompt for converting raw text to CSV
csv_prompt = PromptTemplate.from_template("""
You are an expert timesheet parser.

Convert the raw daily report logs below into a **valid CSV table** with exactly the following columns:

Developer,Date,Login Time,Logout Time,Total Hours,Projects,Tasks

🧠 Rules:
- First line MUST be the exact column headers.
- All values must be comma-separated.
- Wrap fields that contain commas in double quotes.
- Calculate Total Hours as a decimal (e.g. 9:00 AM to 6:30 PM = 9.5).
- If data is missing, leave the field blank but keep all 7 columns.
- Output ONLY the CSV — no explanations, no markdown, no headings.

✅ Example:
Developer,Date,Login Time,Logout Time,Total Hours,Projects,Tasks
"Shubhanshu Shrimali",2025-06-20,9:30 AM,9:25 PM,11.92,"Warvox, Intelliverse-X","Resolved targeting issue, Updated APIs"
"Bhavesh Agarwal",2025-06-20,8:33 AM,9:09 PM,12.6,"Gods Gang","Fixed dungeon issue, Combined levels"

Now convert the following raw logs into that format:

{raw_text}
""")

# Run LangChain LLMChain
chain = LLMChain(llm=llm, prompt=csv_prompt)

# CSV cleaner to remove unwanted output
def clean_llm_csv(llm_output):
    # Remove code block markers like ```csv or ```
    cleaned = re.sub(r"^```(?:csv)?|```$", "", llm_output.strip(), flags=re.MULTILINE)

    # Find the actual header line
    lines = cleaned.splitlines()
    for i, line in enumerate(lines):
        if all(keyword in line.lower() for keyword in [
            "developer", "date", "login", "logout", "total", "project", "task"
        ]):
            return "\n".join(lines[i:])

    raise ValueError("No valid CSV header found in LLM output.")

# Run LLM to convert raw to CSV
with st.spinner("🧠 Analyzing raw timesheet and generating structured CSV..."):
    try:
        llm_output = chain.run(raw_text=timesheet_data)

        # Debug view: raw LLM output
        with st.expander("📝 Raw LLM Output"):
            st.code(llm_output)

        # Clean and parse CSV
        cleaned_csv = clean_llm_csv(llm_output)
        df = pd.read_csv(StringIO(cleaned_csv))

        st.success("✅ Timesheet converted and parsed successfully!")
    except Exception as e:
        st.error(f"❌ Failed to process CSV: {e}")
        st.stop()

# Show full table
st.subheader("👥 Full Timesheet Table")
st.dataframe(df, use_container_width=True)

# Developer Performance Table
st.markdown("---")
st.subheader("💼 Developer Performance Summary")
try:
    performance_df = df.groupby("Developer").agg(
        Total_Hours=("Total Hours", "sum"),
        Task_Count=("Tasks", lambda x: sum(len(str(i).split(",")) for i in x)),
        Project_Count=("Projects", lambda x: sum(len(str(i).split(",")) for i in x))
    ).reset_index()
    st.dataframe(performance_df, use_container_width=True)
except:
    st.warning("⚠️ Unable to compute performance summary. Check column formatting.")

# Project Summary Table
st.markdown("---")
st.subheader("📦 Project Summary")
try:
    project_exploded = df.assign(Projects=df["Projects"].str.split(",")).explode("Projects")
    project_exploded["Projects"] = project_exploded["Projects"].str.strip()
    project_summary = project_exploded.groupby("Projects").agg(
        Total_Hours=("Total Hours", "sum"),
        Contributors=("Developer", pd.Series.nunique)
    ).reset_index()
    st.dataframe(project_summary, use_container_width=True)
except:
    st.warning("⚠️ Could not extract project-wise summary.")

# Visualizations
st.markdown("---")
st.subheader("📊 Visualizations")
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📈 Project Contribution (Bar Chart)")
    try:
        st.bar_chart(project_summary.set_index("Projects")["Total_Hours"])
    except:
        st.warning("⚠️ Project chart failed to load.")

with col2:
    st.markdown("### 🥧 Project Distribution (Pie Chart)")
    try:
        st.plotly_chart({
            "data": [{
                "type": "pie",
                "labels": project_summary["Projects"],
                "values": project_summary["Total_Hours"]
            }]
        })
    except:
        st.warning("⚠️ Pie chart visualization failed.")

# Summary Stats
st.markdown("---")
st.subheader("📋 Summary Statistics")
try:
    st.write(df.describe(include='all'))
except:
    st.warning("⚠️ Summary statistics could not be computed.")
