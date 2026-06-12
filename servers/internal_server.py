from mcp.server.fastmcp import FastMCP
import sqlite3

mcp = FastMCP("InternalHospitalDB", host="0.0.0.0", port=8001)

DB_FILE = "expanded_hospital.db"


def get_connection():
    return sqlite3.connect(DB_FILE)


@mcp.tool()
async def get_patient(patient_id: str):

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT *
        FROM patients
        WHERE patient_id = ?
    """, (patient_id,))

    patient = cursor.fetchone()

    conn.close()

    if not patient:
        return {"error": "Patient not found"}

    return {
        "patient_id": patient[0],
        "first_name": patient[1],
        "last_name": patient[2],
        "age": patient[3],
        "gender": patient[4]
    }


@mcp.tool()
async def get_patient_diagnosis(patient_id: str):

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT condition_name, icd10_code
        FROM diagnoses
        WHERE patient_id = ?
    """, (patient_id,))

    rows = cursor.fetchall()

    conn.close()

    diagnoses = []

    for row in rows:
        diagnoses.append({
            "condition": row[0],
            "icd10_code": row[1]
        })

    return {
        "patient_id": patient_id,
        "diagnoses": diagnoses
    }


@mcp.tool()
async def get_lab_results(patient_id: str):

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT test_name, test_value, unit
        FROM lab_results
        WHERE patient_id = ?
    """, (patient_id,))

    rows = cursor.fetchall()

    conn.close()

    labs = []

    for row in rows:
        labs.append({
            "test_name": row[0],
            "value": row[1],
            "unit": row[2]
        })

    return {
        "patient_id": patient_id,
        "lab_results": labs
    }


@mcp.tool()
async def get_medications(patient_id: str):

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT drug_name, dosage, frequency
        FROM medications
        WHERE patient_id = ?
    """, (patient_id,))

    rows = cursor.fetchall()

    conn.close()

    medications = []

    for row in rows:
        medications.append({
            "drug_name": row[0],
            "dosage": row[1],
            "frequency": row[2]
        })

    return {
        "patient_id": patient_id,
        "medications": medications
    }


@mcp.tool()
async def search_patients_by_symptom(symptom: str):

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT patients.patient_id,
               patients.first_name,
               patients.last_name,
               symptoms.symptom,
               diagnoses.condition_name
        FROM patients
        JOIN symptoms
            ON patients.patient_id = symptoms.patient_id
        LEFT JOIN diagnoses
            ON patients.patient_id = diagnoses.patient_id
        WHERE LOWER(symptoms.symptom) = LOWER(?)
    """, (symptom,))

    rows = cursor.fetchall()

    conn.close()

    results = []

    for row in rows:
        results.append({
            "patient_id": row[0],
            "name": f"{row[1]} {row[2]}",
            "symptom": row[3],
            "diagnosis": row[4]
        })

    return {
        "symptom": symptom,
        "matches": results
    }


@mcp.tool()
async def risk_flag_patient(patient_id: str):

    conn = get_connection()
    cursor = conn.cursor()

    score = 0

    cursor.execute("""
        SELECT age
        FROM patients
        WHERE patient_id = ?
    """, (patient_id,))

    patient = cursor.fetchone()

    if not patient:
        conn.close()
        return {"error": "Patient not found"}

    age = patient[0]

    if age >= 65:
        score += 2

    cursor.execute("""
        SELECT test_name, test_value
        FROM lab_results
        WHERE patient_id = ?
    """, (patient_id,))

    labs = cursor.fetchall()

    for lab in labs:

        test_name = lab[0]
        value = str(lab[1])

        if test_name == "Oxygen Saturation":
            try:
                if float(value) < 92:
                    score += 3
            except:
                pass

        if test_name == "Glucose":
            try:
                if float(value) > 200:
                    score += 2
            except:
                pass

        if test_name == "WBC":
            try:
                if float(value) > 12:
                    score += 2
            except:
                pass

    conn.close()

    level = "LOW"

    if score >= 5:
        level = "HIGH"
    elif score >= 3:
        level = "MEDIUM"

    return {
        "patient_id": patient_id,
        "risk_score": score,
        "risk_level": level
    }


if __name__ == "__main__":
    mcp.run(transport="streamable-http")