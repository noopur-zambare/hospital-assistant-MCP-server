import sqlite3
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

DB_PATH = Path(__file__).parent / "expanded_hospital.db"
BLUE = "#4C78A8"


@st.cache_data(show_spinner="Loading database...")
def load_data():
    conn = sqlite3.connect(DB_PATH)
    data = {
        "patients": pd.read_sql("SELECT * FROM patients", conn),
        "symptoms": pd.read_sql("SELECT * FROM symptoms", conn),
        "diagnoses": pd.read_sql("SELECT * FROM diagnoses", conn),
        "labs": pd.read_sql("SELECT * FROM lab_results", conn),
        "meds": pd.read_sql("SELECT * FROM medications", conn),
        "notes": pd.read_sql("SELECT * FROM clinical_notes", conn),
    }
    conn.close()
    return data


def blue_bar(df, x, y, title):
    fig = px.bar(df, x=x, y=y, text=y, title=title, template="plotly_dark")
    fig.update_traces(textposition="outside", marker_color=BLUE, cliponaxis=False)
    fig.update_layout(
        title_font_size=20,
        xaxis_title=x,
        yaxis_title=y,
        uniformtext_minsize=6,
        uniformtext_mode="hide",
        xaxis_tickangle=-45,
    )
    return fig


def _counts(series, names):
    df = series.value_counts().reset_index()
    df.columns = names
    return df


def plot_age(d):
    return px.histogram(d["patients"], x="age", nbins=20, title="Age Distribution",
                        template="plotly_dark", color_discrete_sequence=[BLUE])

def plot_gender(d):
    return blue_bar(_counts(d["patients"]["gender"], ["gender", "count"]),
                    "gender", "count", "Gender Distribution")

def plot_symptoms(d):
    return blue_bar(_counts(d["symptoms"]["symptom"], ["symptom", "count"]),
                    "symptom", "count", "All Symptoms Frequency")

def plot_severity(d):
    return blue_bar(_counts(d["symptoms"]["severity"], ["severity", "count"]),
                    "severity", "count", "Severity Distribution")

def plot_diagnoses(d):
    return blue_bar(_counts(d["diagnoses"]["condition_name"], ["condition", "count"]),
                    "condition", "count", "All Diagnoses")

def plot_icd10(d):
    return blue_bar(_counts(d["diagnoses"]["icd10_code"], ["icd10", "count"]),
                    "icd10", "count", "ICD-10 Distribution")

def plot_labs(d):
    return blue_bar(_counts(d["labs"]["test_name"], ["test", "count"]),
                    "test", "count", "All Lab Tests")

def plot_meds(d):
    return blue_bar(_counts(d["meds"]["drug_name"], ["drug", "count"]),
                    "drug", "count", "All Medications")

def plot_visits_per_patient(d):
    df = d["notes"]["patient_id"].value_counts().value_counts().sort_index().reset_index()
    df.columns = ["visits", "patients"]
    return blue_bar(df, "visits", "patients", "Visits per Patient")

def plot_note_length(d):
    notes = d["notes"].copy()
    notes["note_length"] = notes["note"].str.len()
    return px.histogram(notes, x="note_length", nbins=30,
                        title="Clinical Note Length Distribution",
                        template="plotly_dark", color_discrete_sequence=[BLUE])

def plot_timeline(d):
    notes = d["notes"].copy()
    notes["visit_date"] = pd.to_datetime(notes["visit_date"])
    timeline = notes.groupby("visit_date").size().reset_index(name="visits")
    fig = px.line(timeline, x="visit_date", y="visits",
                  title="Hospital Visits Over Time", template="plotly_dark")
    fig.update_traces(line_color=BLUE)
    return fig


PLOTS = {
    "Age Distribution": plot_age,
    "Gender Distribution": plot_gender,
    "Symptoms Frequency": plot_symptoms,
    "Severity Distribution": plot_severity,
    "All Diagnoses": plot_diagnoses,
    "ICD-10 Distribution": plot_icd10,
    "Lab Tests": plot_labs,
    "Medications": plot_meds,
    # "Visits per Patient": plot_visits_per_patient,
    # "Note Length Distribution": plot_note_length,
    "Visits Over Time": plot_timeline,
}


def render_analysis():
    data = load_data()

    with st.sidebar:
        st.header("Plots")
        selected = st.multiselect(
            "Select plots to display",
            options=list(PLOTS.keys()),
            default=["Age Distribution", "All Diagnoses"],
        )

    if not selected:
        st.info("Select plots from the left pane.")
        return

    cols = st.columns(2)
    for i, name in enumerate(selected):
        with cols[i % 2]:
            st.plotly_chart(PLOTS[name](data), use_container_width=True)