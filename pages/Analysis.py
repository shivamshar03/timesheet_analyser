import streamlit as st
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import pandas as pd
import plotly.express as px

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# Initialize LLM
llm = ChatGroq(temperature=0.9, model_name="llama3-8b-8192", groq_api_key=groq_api_key)

# Define the prompt template
prompt_template = PromptTemplate(
    input_variables=["timesheet_data"],
    template="""
You are a Project Manager reviewing daily timesheet reports submitted by a team of game developers.

Based on the data provided below, generate the following:

1. Project Summary:
   - List all ongoing projects.
   - Summarize each project’s key updates and progress in bullet points , it should be like main heading (project name ) then poject highlighted  subpoints.
   - Mention contributors for each project.

2. Employee Performance Analysis:
   For each employee:
   - Mention their name and project contributions.
   - Highlight key achievements or issues.
   - Provide a very honest performance rating (out of 5 stars).
   - Note login and logout times, and calculate average working hours.

3. Work Hours Table:
   A table showing average login duration per day per employee with observations.

4. Important Highlights and Recommendations:
   - Mention technical breakthroughs, collaboration insights, and any blockers.
   - Suggest next steps, improvements, or adjustments in work assignments.

5. 📊 At the end, output chart data as CSV like this:Add commentMore actions
--RAW DATA FOR VISUALIZATION--
employee_name,average_hours,rating
Shubhanshu Shrimali,9.5,4.5
...
--END OF DATA--

perl
Copy
Edit
⚠️ Only include numeric values. Omit rows with missing or "varies" data.

-- RAW TIMESHEET DATA START --
{timesheet_data}
-- RAW TIMESHEET DATA END --

Note : Do not consider saturday and sunday and also chek the human errors .
"""
)

# LangChain LLM Chain
chain = LLMChain(llm=llm, prompt=prompt_template)

# Streamlit UI
st.set_page_config(page_title="Timesheet Dashboard - Upload & Analyze", layout="wide")
st.title("📤 Upload Timesheet & Generate Report")

if "uploaded_file_bytes" not in st.session_state:
    st.warning("Please upload a timesheet file from the Upload & Analyze page first.")
    st.stop()

# Read text data
timesheet_text = st.session_state["uploaded_file_bytes"]

if timesheet_text is not None:
    with st.spinner("Generating report..."):
        result = chain.invoke({"timesheet_data": timesheet_text})

    # Extract raw report and CSV data
    report_text = result.get("text", "")
    st.write(report_text)
    st.subheader("📋 Generated Report")
    st.markdown(report_text)

    # Attempt to parse RAW DATA section
    try:
        csv_start = report_text.index("--RAW DATA FOR VISUALIZATION--") + len("--RAW DATA FOR VISUALIZATION--")
        csv_end = report_text.index("--END OF DATA--")
        raw_csv = report_text[csv_start:csv_end].strip()

        from io import StringIO
        df = pd.read_csv(StringIO(raw_csv))

        st.subheader("📊 Visual Insights")

        # Example usage of tabs
        tab1, tab2 = st.tabs(["Working Hours", "Performance Rating"])

        with tab1:
            fig1 = px.bar(df, x="employee_name", y="average_hours", color="employee_name",
                          title="Average Working Hours")
            st.plotly_chart(fig1, use_container_width=True)

        with tab2:
            fig2 = px.bar(df, x="employee_name", y="rating", color="employee_name", title="Performance Ratings")
            st.plotly_chart(fig2, use_container_width=True)

    except Exception as e:
        st.warning(f"⚠️ Could not parse visualization data: {e}")

    st.download_button(
        label="📥 Download Full Report",
        data=report_text,
        file_name="project_report.txt",
        mime="text/plain"
    )
else:
    st.info("Please upload a valid `.txt` file containing daily report logs.")
