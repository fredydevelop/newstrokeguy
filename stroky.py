# ==========================================
# STROKE PREDICTION SYSTEM - STREAMLIT APP
# ==========================================

import numpy as np
import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu
import base64
import pickle as pk


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Stroke Prediction System",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="expanded"
)


# ==========================================
# CUSTOM CSS
# ==========================================

st.markdown(
    """
    <style>

    /* Overall page */
    .stApp {
        background-color: #f5f8fb;
    }

    /* Main content width */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1000px;
    }

    /* Hide Streamlit default menu/footer */
    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }


    /* ===============================
       HEADER
    =============================== */

    .main-header {
        background: linear-gradient(
            135deg,
            #0f766e,
            #0d9488
        );

        padding: 28px 30px;
        border-radius: 18px;
        margin-bottom: 25px;
        color: white;

        box-shadow:
            0 8px 24px rgba(0,0,0,0.08);
    }

    .main-header h1 {
        margin: 0;
        font-size: 30px;
        font-weight: 700;
    }

    .main-header p {
        margin-top: 8px;
        margin-bottom: 0;
        font-size: 15px;
        opacity: 0.92;
    }


    /* ===============================
       SECTION HEADER
    =============================== */

    .section-title {
        font-size: 20px;
        font-weight: 700;
        color: #17324d;
        margin-top: 5px;
        margin-bottom: 4px;
    }

    .section-description {
        color: #64748b;
        font-size: 14px;
        margin-bottom: 20px;
    }


    /* ===============================
       INFORMATION BOX
    =============================== */

    .info-box {
        background-color: #eaf6f5;
        border-left: 4px solid #0d9488;
        border-radius: 10px;
        padding: 14px 18px;
        margin-bottom: 22px;
        color: #334155;
        font-size: 14px;
    }


    /* ===============================
       RESULT CARDS
    =============================== */

    .result-positive {
        background-color: #fff4f4;
        border: 1px solid #fecaca;
        border-left: 5px solid #dc2626;
        padding: 22px;
        border-radius: 14px;
        margin-top: 20px;
    }

    .result-positive h3 {
        color: #b91c1c;
        margin: 0 0 5px 0;
    }

    .result-positive p {
        color: #7f1d1d;
        margin: 0;
    }


    .result-negative {
        background-color: #f0fdf4;
        border: 1px solid #bbf7d0;
        border-left: 5px solid #16a34a;
        padding: 22px;
        border-radius: 14px;
        margin-top: 20px;
    }

    .result-negative h3 {
        color: #15803d;
        margin: 0 0 5px 0;
    }

    .result-negative p {
        color: #166534;
        margin: 0;
    }


    /* ===============================
       MEDICAL DISCLAIMER
    =============================== */

    .disclaimer {
        margin-top: 15px;
        background-color: #fff7ed;
        border: 1px solid #fed7aa;
        border-radius: 10px;
        padding: 12px 16px;
        color: #9a3412;
        font-size: 13px;
    }


    /* ===============================
       BUTTON
    =============================== */

    div.stButton > button {
        width: 100%;
        background-color: #0d9488;
        color: white;
        border: none;
        border-radius: 10px;
        padding: 12px;
        height: 48px;
        font-size: 16px;
        font-weight: 600;
    }

    div.stButton > button:hover {
        background-color: #0f766e;
        color: white;
        border: none;
    }


    /* Form button */

    div[data-testid="stFormSubmitButton"] button {
        width: 100%;
        background-color: #0d9488;
        color: white;
        border: none;
        border-radius: 10px;
        height: 48px;
        font-size: 16px;
        font-weight: 600;
    }

    div[data-testid="stFormSubmitButton"] button:hover {
        background-color: #0f766e;
        color: white;
    }


    /* ===============================
       INPUTS
    =============================== */

    div[data-baseweb="select"] > div {
        border-radius: 10px;
    }

    div[data-testid="stNumberInput"] input {
        border-radius: 10px;
    }


    /* ===============================
       SIDEBAR
    =============================== */

    section[data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e2e8f0;
    }


    /* ===============================
       UPLOAD BOX
    =============================== */

    div[data-testid="stFileUploader"] {
        background-color: white;
        padding: 16px;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
    }


    /* ===============================
       TABLE
    =============================== */

    div[data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ==========================================
# SIDEBAR
# ==========================================

with st.sidebar:

    st.markdown(
        """
        <div style="
            padding:15px 5px 20px 5px;
        ">
            <h2 style="
                color:#0f766e;
                margin-bottom:2px;
            ">
                🧠 StrokeDetect
            </h2>

            <p style="
                color:#64748b;
                font-size:13px;
                margin-top:0;
            ">
                Machine Learning Prediction System
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )


    selection = option_menu(
        menu_title=None,

        options=[
            "Single Prediction",
            "Multi Prediction"
        ],

        icons=[
            "person",
            "file-earmark-spreadsheet"
        ],

        default_index=0,

        styles={

            "container": {
                "padding": "0!important",
                "background-color": "#ffffff"
            },

            "icon": {
                "color": "#0d9488",
                "font-size": "18px"
            },

            "nav-link": {
                "font-size": "15px",
                "text-align": "left",
                "margin": "5px 0",
                "padding": "11px 13px",
                "border-radius": "9px",
                "--hover-color": "#eaf6f5"
            },

            "nav-link-selected": {
                "background-color": "#0d9488",
                "color": "white"
            }
        }
    )


    st.markdown("---")

    st.caption(
        "Stroke prediction using machine learning."
    )


# ==========================================
# MODEL FILES
# ==========================================

MODEL_FILE = "strokedetect.sav"

SCALER_FILE = "my_standard_scaler.pkl"


# ==========================================
# FEATURE ORDER
# ==========================================

FEATURE_COLUMNS = [
    "gender",
    "age",
    "hypertension",
    "heart_disease",
    "ever_married",
    "work_type",
    "Residence_type",
    "avg_glucose_level",
    "bmi",
    "smoking_status"
]


# ==========================================
# CATEGORICAL ENCODINGS
# ==========================================

SEX_MAP = {
    "Male": 1,
    "Female": 0
}


HYPERTENSION_MAP = {
    "Hypertensive": 1,
    "Not hypertensive": 0
}


HEART_DISEASE_MAP = {
    "Has heart disease": 1,
    "No heart disease": 0
}


MARRIED_MAP = {
    "Yes": 1,
    "No": 0
}


WORK_TYPE_MAP = {
    "Govt_job": 0,
    "Never_worked": 1,
    "Private": 2,
    "Self-employed": 3,
    "children": 4
}


RESIDENCE_MAP = {
    "Rural": 0,
    "Urban": 1
}


SMOKING_MAP = {
    "never smoked": 0,
    "formerly smoked": 1,
    "smokes": 2,
    "Unknown": 3
}


# ==========================================
# LOAD MODEL
# ==========================================

@st.cache_resource
def load_model():

    with open(
        MODEL_FILE,
        "rb"
    ) as file:

        model = pk.load(file)

    return model


# ==========================================
# LOAD SCALER
# ==========================================

@st.cache_resource
def load_scaler():

    with open(
        SCALER_FILE,
        "rb"
    ) as file:

        scaler = pk.load(file)

    return scaler


# ==========================================
# DOWNLOAD FUNCTION
# ==========================================

def filedownload(df):

    csv = df.to_csv(
        index=False
    )

    b64 = base64.b64encode(
        csv.encode()
    ).decode()

    href = (
        f'<a href="data:file/csv;base64,{b64}" '
        f'download="stroke_prediction.csv">'
        f'Download Prediction Results'
        f'</a>'
    )

    return href


# ==========================================
# SINGLE PREDICTION FUNCTION
# ==========================================

def stroke_detect(given_data):

    model = load_model()

    scaler = load_scaler()


    input_data = np.asarray(
        given_data,
        dtype=float
    )


    input_data_reshaped = (
        input_data.reshape(
            1,
            -1
        )
    )


    if (
        input_data_reshaped.shape[1]
        != 10
    ):

        raise ValueError(
            "The model requires exactly "
            "10 input features."
        )


    if not np.all(
        np.isfinite(
            input_data_reshaped
        )
    ):

        raise ValueError(
            "Invalid numeric input detected."
        )


    scaled_input = scaler.transform(
        input_data_reshaped
    )


    prediction = model.predict(
        scaled_input
    )[0]


    prediction = int(
        prediction
    )


    return prediction


# ==========================================
# SINGLE PREDICTION PAGE
# ==========================================

def single_prediction_page():

    # HEADER

    st.markdown(
        """
        <div class="main-header">

            <h1>
                🧠 Stroke Prediction System
            </h1>

            <p>
                Enter the patient's health information
                to generate a machine-learning prediction.
            </p>

        </div>
        """,
        unsafe_allow_html=True
    )


    st.markdown(
        """
        <div class="info-box">

            <strong>Patient Assessment</strong><br>

            Complete all fields below.
            Categorical fields must be selected
            before a prediction can be generated.

        </div>
        """,
        unsafe_allow_html=True
    )


    st.markdown(
        '<div class="section-title">Patient Information</div>',
        unsafe_allow_html=True
    )


    st.markdown(
        """
        <div class="section-description">
            Provide the patient's demographic and
            clinical information.
        </div>
        """,
        unsafe_allow_html=True
    )


    # ======================================
    # FORM
    # ======================================

    with st.form(
        "stroke_prediction_form"
    ):


        # ROW 1

        col1, col2 = st.columns(2)


        with col1:

            age = st.number_input(
                "Patient Age",
                min_value=0,
                max_value=120,
                value=None,
                step=1,
                placeholder="Enter age"
            )


        with col2:

            option1 = st.selectbox(
                "Sex",
                (
                    "Male",
                    "Female"
                ),
                index=None,
                placeholder="Choose an option"
            )


        # ROW 2

        col1, col2 = st.columns(2)


        with col1:

            option4 = st.selectbox(
                "Hypertension",
                (
                    "Hypertensive",
                    "Not hypertensive"
                ),
                index=None,
                placeholder="Choose an option"
            )


        with col2:

            option5 = st.selectbox(
                "Heart Disease",
                (
                    "Has heart disease",
                    "No heart disease"
                ),
                index=None,
                placeholder="Choose an option"
            )


        # ROW 3

        col1, col2 = st.columns(2)


        with col1:

            marriage = st.selectbox(
                "Ever Married?",
                (
                    "Yes",
                    "No"
                ),
                index=None,
                placeholder="Choose an option"
            )


        with col2:

            option6 = st.selectbox(
                "Work Type",
                (
                    "children",
                    "Govt_job",
                    "Never_worked",
                    "Private",
                    "Self-employed"
                ),
                index=None,
                placeholder="Choose an option"
            )


        # ROW 4

        col1, col2 = st.columns(2)


        with col1:

            option7 = st.selectbox(
                "Residence Type",
                (
                    "Rural",
                    "Urban"
                ),
                index=None,
                placeholder="Choose an option"
            )


        with col2:

            smoking_status = st.selectbox(
                "Smoking Status",
                (
                    "never smoked",
                    "formerly smoked",
                    "smokes",
                    "Unknown"
                ),
                index=None,
                placeholder="Choose an option"
            )


        # ROW 5

        col1, col2 = st.columns(2)


        with col1:

            glucose = st.number_input(
                "Average Glucose Level",
                min_value=0.0,
                value=None,
                step=0.01,
                placeholder="Enter glucose level"
            )


        with col2:

            bmi = st.number_input(
                "Body Mass Index (BMI)",
                min_value=0.0,
                value=None,
                step=0.01,
                placeholder="Enter BMI"
            )


        st.write("")


        submitted = (
            st.form_submit_button(
                "Generate Prediction"
            )
        )


    # ======================================
    # VALIDATION
    # ======================================

    if submitted:


        fields = {

            "Patient Age":
                age,

            "Sex":
                option1,

            "Hypertension":
                option4,

            "Heart Disease":
                option5,

            "Ever Married":
                marriage,

            "Work Type":
                option6,

            "Residence Type":
                option7,

            "Average Glucose Level":
                glucose,

            "BMI":
                bmi,

            "Smoking Status":
                smoking_status
        }


        missing_fields = [

            name

            for name, value
            in fields.items()

            if value is None

        ]


        if missing_fields:

            st.error(
                "Please complete the following field(s): "
                + ", ".join(
                    missing_fields
                )
            )

            return


        # ==================================
        # NUMERIC VALIDATION
        # ==================================

        validation_errors = []


        if glucose <= 0:

            validation_errors.append(
                "Average glucose level "
                "must be greater than 0."
            )


        if bmi <= 0:

            validation_errors.append(
                "BMI must be greater than 0."
            )


        if validation_errors:

            for error in validation_errors:

                st.error(
                    error
                )

            return


        # ==================================
        # ENCODE CATEGORIES
        # ==================================

        sex = SEX_MAP[
            option1
        ]


        hyperten = HYPERTENSION_MAP[
            option4
        ]


        heartDis = HEART_DISEASE_MAP[
            option5
        ]


        married = MARRIED_MAP[
            marriage
        ]


        work_type = WORK_TYPE_MAP[
            option6
        ]


        resident = RESIDENCE_MAP[
            option7
        ]


        smoking_stat = SMOKING_MAP[
            smoking_status
        ]


        # ==================================
        # MODEL INPUT
        # ==================================

        patient_data = [

            sex,

            age,

            hyperten,

            heartDis,

            married,

            work_type,

            resident,

            glucose,

            bmi,

            smoking_stat
        ]


        # ==================================
        # PREDICTION
        # ==================================

        try:

            prediction = stroke_detect(
                patient_data
            )


            if prediction == 1:

                st.markdown(
                    """
                    <div class="result-positive">

                        <h3>
                            ⚠ Stroke Risk Detected
                        </h3>

                        <p>
                            The machine-learning model
                            classified this patient's
                            input as indicating stroke risk.
                        </p>

                    </div>
                    """,
                    unsafe_allow_html=True
                )


            else:

                st.markdown(
                    """
                    <div class="result-negative">

                        <h3>
                            ✓ No Stroke Risk Detected
                        </h3>

                        <p>
                            The machine-learning model
                            classified this patient's
                            input as not indicating
                            stroke risk.
                        </p>

                    </div>
                    """,
                    unsafe_allow_html=True
                )


            st.markdown(
                """
                <div class="disclaimer">

                    <strong>Medical Disclaimer:</strong>

                    This prediction is generated by
                    a machine-learning model and
                    should not be considered a
                    medical diagnosis.

                </div>
                """,
                unsafe_allow_html=True
            )


        except Exception as error:

            st.error(
                "The prediction could not "
                "be completed."
            )

            st.exception(
                error
            )


# ==========================================
# MULTI PREDICTION
# ==========================================

def multi_prediction(
    uploaded_file
):


    model = load_model()

    scaler = load_scaler()


    try:

        dfinput = pd.read_csv(
            uploaded_file
        )


    except Exception:

        st.error(
            "The uploaded file could "
            "not be read."
        )

        return


    # Remove target

    if "stroke" in dfinput.columns:

        dfinput.drop(
            "stroke",
            axis=1,
            inplace=True
        )


    # Remove ID

    if "id" in dfinput.columns:

        dfinput.drop(
            "id",
            axis=1,
            inplace=True
        )


    dfinput.reset_index(
        drop=True,
        inplace=True
    )


    st.markdown(
        '<div class="section-title">Uploaded Patient Data</div>',
        unsafe_allow_html=True
    )


    st.write(
        f"{len(dfinput)} patient record(s) detected."
    )


    st.dataframe(
        dfinput,
        use_container_width=True
    )


    # ======================================
    # COLUMN VALIDATION
    # ======================================

    if (
        list(dfinput.columns)
        != FEATURE_COLUMNS
    ):

        st.error(
            "The uploaded dataset columns "
            "do not match the features "
            "required by the model."
        )


        st.write(
            "Required column order:"
        )


        st.code(
            ", ".join(
                FEATURE_COLUMNS
            )
        )

        return


    # ======================================
    # MISSING VALUES
    # ======================================

    if dfinput.isnull().any().any():

        missing_columns = (

            dfinput.columns[
                dfinput.isnull().any()
            ]

            .tolist()
        )


        st.error(
            "Missing values detected in: "
            + ", ".join(
                missing_columns
            )
        )

        return


    # ======================================
    # NUMERIC CONVERSION
    # ======================================

    try:

        numeric_data = (
            dfinput.astype(
                float
            )
        )


    except ValueError:

        st.error(
            "The uploaded CSV must contain "
            "encoded numeric values."
        )

        return


    # ======================================
    # SCALE DATA
    # ======================================

    scaled_data = scaler.transform(
        numeric_data.values
    )


    # ======================================
    # PREDICT
    # ======================================

    if st.button(
        "Generate Predictions"
    ):


        prediction = model.predict(
            scaled_data
        )


        prediction = np.asarray(
            prediction
        ).reshape(-1)


        labels = []


        for result in prediction:

            if int(result) == 1:

                labels.append(
                    "Stroke risk detected"
                )

            else:

                labels.append(
                    "No stroke risk detected"
                )


        result_dataframe = pd.DataFrame({

            "Patient ID":
                np.arange(
                    1,
                    len(labels) + 1
                ),

            "Prediction":
                labels

        })


        st.markdown(
            '<div class="section-title">Prediction Results</div>',
            unsafe_allow_html=True
        )


        st.dataframe(
            result_dataframe,
            use_container_width=True
        )


        st.markdown(
            filedownload(
                result_dataframe
            ),
            unsafe_allow_html=True
        )


# ==========================================
# MULTI PREDICTION PAGE
# ==========================================

def multi_prediction_page():


    st.markdown(
        """
        <div class="main-header">

            <h1>
                📊 Multiple Patient Prediction
            </h1>

            <p>
                Upload a CSV dataset to generate
                stroke predictions for multiple
                patients simultaneously.
            </p>

        </div>
        """,
        unsafe_allow_html=True
    )


    st.markdown(
        '<div class="section-title">Upload Dataset</div>',
        unsafe_allow_html=True
    )


    st.markdown(
        """
        <div class="section-description">
            Your CSV file must contain the ten
            input features required by the model.
        </div>
        """,
        unsafe_allow_html=True
    )


    uploaded_file = st.file_uploader(
        "Upload CSV file",
        type=[
            "csv"
        ]
    )


    if uploaded_file is not None:

        multi_prediction(
            uploaded_file
        )


    else:

        st.info(
            "Select a CSV file to begin "
            "multiple-patient prediction."
        )


# ==========================================
# ROUTING
# ==========================================

if selection == "Single Prediction":

    single_prediction_page()


elif selection == "Multi Prediction":

    multi_prediction_page()
