# ==========================================
# STROKE PREDICTION SYSTEM
# STREAMLIT APPLICATION
# ==========================================

import numpy as np
import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu
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
    with open(MODEL_FILE, "rb") as file:
        model = pk.load(file)

    return model


# ==========================================
# LOAD SCALER
# ==========================================

@st.cache_resource
def load_scaler():
    with open(SCALER_FILE, "rb") as file:
        scaler = pk.load(file)

    return scaler


# ==========================================
# CLEAR OLD PREDICTION
# WHEN AN INPUT CHANGES
# ==========================================

def clear_prediction():
    st.session_state.prediction_result = None


# ==========================================
# INITIALISE SESSION STATE
# ==========================================

if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None


# ==========================================
# SIDEBAR
# ==========================================

with st.sidebar:

    selection = option_menu(
        menu_title="Main Menu",

        options=[
            "Single Prediction",
            "Multi Prediction"
        ],

        icons=[
            "person",
            "file-earmark-spreadsheet"
        ],

        menu_icon="house",

        default_index=0
    )


# ==========================================
# SINGLE PREDICTION FUNCTION
# ==========================================

def stroke_detect(given_data):

    model = load_model()
    scaler = load_scaler()

    # Convert input into NumPy array
    input_data = np.asarray(
        given_data,
        dtype=float
    )

    # Reshape for a single patient
    input_data = input_data.reshape(
        1,
        -1
    )

    # Validate number of features
    if input_data.shape[1] != 10:

        raise ValueError(
            "The model requires exactly 10 input features."
        )

    # Validate numeric values
    if not np.all(
        np.isfinite(input_data)
    ):

        raise ValueError(
            "The patient data contains an invalid numeric value."
        )

    # Scale the patient information
    scaled_input = scaler.transform(
        input_data
    )

    # Make prediction
    prediction = model.predict(
        scaled_input
    )

    # Extract prediction
    prediction = np.asarray(
        prediction
    ).reshape(-1)[0]

    return int(prediction)


# ==========================================
# SINGLE PREDICTION PAGE
# ==========================================

def single_prediction_page():

    st.title(
        "🧠 Stroke Prediction System"
    )

    st.subheader(
        "Patient Information"
    )

    st.caption(
        "Provide the patient's information in the fields below."
    )


    # ======================================
    # ROW 1
    # ======================================

    col1, col2 = st.columns(2)


    with col1:

        age = st.number_input(
            "Patient Age",
            min_value=0,
            max_value=120,
            value=None,
            step=1,
            placeholder="Enter age",
            key="patient_age",
            on_change=clear_prediction
        )


    with col2:

        sex_option = st.selectbox(
            "Sex",
            (
                "Male",
                "Female"
            ),
            index=None,
            placeholder="Choose an option",
            key="patient_sex",
            on_change=clear_prediction
        )


    # ======================================
    # ROW 2
    # ======================================

    col1, col2 = st.columns(2)


    with col1:

        hypertension_option = st.selectbox(
            "Hypertension",
            (
                "Hypertensive",
                "Not hypertensive"
            ),
            index=None,
            placeholder="Choose an option",
            key="patient_hypertension",
            on_change=clear_prediction
        )


    with col2:

        heart_disease_option = st.selectbox(
            "Heart Disease",
            (
                "Has heart disease",
                "No heart disease"
            ),
            index=None,
            placeholder="Choose an option",
            key="patient_heart_disease",
            on_change=clear_prediction
        )


    # ======================================
    # ROW 3
    # ======================================

    col1, col2 = st.columns(2)


    with col1:

        marriage_option = st.selectbox(
            "Ever Married?",
            (
                "Yes",
                "No"
            ),
            index=None,
            placeholder="Choose an option",
            key="patient_marriage",
            on_change=clear_prediction
        )


    with col2:

        work_option = st.selectbox(
            "Work Type",
            (
                "children",
                "Govt_job",
                "Never_worked",
                "Private",
                "Self-employed"
            ),
            index=None,
            placeholder="Choose an option",
            key="patient_work_type",
            on_change=clear_prediction
        )


    # ======================================
    # ROW 4
    # ======================================

    col1, col2 = st.columns(2)


    with col1:

        residence_option = st.selectbox(
            "Residence Type",
            (
                "Rural",
                "Urban"
            ),
            index=None,
            placeholder="Choose an option",
            key="patient_residence",
            on_change=clear_prediction
        )


    with col2:

        smoking_option = st.selectbox(
            "Smoking Status",
            (
                "never smoked",
                "formerly smoked",
                "smokes",
                "Unknown"
            ),
            index=None,
            placeholder="Choose an option",
            key="patient_smoking",
            on_change=clear_prediction
        )


    # ======================================
    # ROW 5
    # ======================================

    col1, col2 = st.columns(2)


    with col1:

        glucose = st.number_input(
            "Average Glucose Level",
            min_value=0.0,
            value=None,
            step=0.01,
            placeholder="Enter glucose level",
            key="patient_glucose",
            on_change=clear_prediction
        )


    with col2:

        bmi = st.number_input(
            "Body Mass Index (BMI)",
            min_value=0.0,
            value=None,
            step=0.01,
            placeholder="Enter BMI",
            key="patient_bmi",
            on_change=clear_prediction
        )


    st.write("")


    # ======================================
    # GENERATE PREDICTION BUTTON
    # ======================================

    predict_button = st.button(
        "Generate Prediction",
        type="primary",
        use_container_width=True
    )


    # ======================================
    # VALIDATION AND PREDICTION
    # ======================================

    if predict_button:

        fields = {

            "Patient Age": age,

            "Sex": sex_option,

            "Hypertension": hypertension_option,

            "Heart Disease": heart_disease_option,

            "Ever Married": marriage_option,

            "Work Type": work_option,

            "Residence Type": residence_option,

            "Average Glucose Level": glucose,

            "BMI": bmi,

            "Smoking Status": smoking_option
        }


        # ----------------------------------
        # REQUIRED FIELD VALIDATION
        # ----------------------------------

        missing_fields = [
            field_name
            for field_name, value in fields.items()
            if value is None
        ]


        if missing_fields:

            st.session_state.prediction_result = None

            st.error(
                "Please complete the following field(s): "
                + ", ".join(missing_fields),
                icon="⚠️"
            )

            return


        # ----------------------------------
        # NUMERIC VALIDATION
        # ----------------------------------

        validation_errors = []


        if glucose <= 0:

            validation_errors.append(
                "Average glucose level must be greater than 0."
            )


        if bmi <= 0:

            validation_errors.append(
                "BMI must be greater than 0."
            )


        if validation_errors:

            st.session_state.prediction_result = None

            for error in validation_errors:

                st.error(
                    error,
                    icon="⚠️"
                )

            return


        # ----------------------------------
        # ENCODE CATEGORICAL VALUES
        # ----------------------------------

        sex = SEX_MAP[
            sex_option
        ]


        hypertension = HYPERTENSION_MAP[
            hypertension_option
        ]


        heart_disease = HEART_DISEASE_MAP[
            heart_disease_option
        ]


        married = MARRIED_MAP[
            marriage_option
        ]


        work_type = WORK_TYPE_MAP[
            work_option
        ]


        residence = RESIDENCE_MAP[
            residence_option
        ]


        smoking_status = SMOKING_MAP[
            smoking_option
        ]


        # ----------------------------------
        # CREATE MODEL INPUT
        # ----------------------------------

        patient_data = [
            sex,
            age,
            hypertension,
            heart_disease,
            married,
            work_type,
            residence,
            glucose,
            bmi,
            smoking_status
        ]


        # ----------------------------------
        # MAKE PREDICTION
        # ----------------------------------

        try:

            prediction = stroke_detect(
                patient_data
            )

            st.session_state.prediction_result = prediction


        except Exception as error:

            st.session_state.prediction_result = None

            st.error(
                "The prediction could not be completed.",
                icon="❌"
            )

            st.exception(
                error
            )

            return


    # ======================================
    # DISPLAY PREDICTION RESULT
    # ======================================

    if (
        st.session_state.prediction_result
        is not None
    ):

        st.subheader(
            "Prediction Result"
        )


        if (
            st.session_state.prediction_result
            == 1
        ):

            st.error(
                "Stroke Risk Detected",
                icon="⚠️"
            )


            st.write(
                "The machine-learning model classified "
                "the patient's information as indicating "
                "stroke risk."
            )


        else:

            st.success(
                "No Stroke Risk Detected",
                icon="✅"
            )


            st.write(
                "The machine-learning model classified "
                "the patient's information as not "
                "indicating stroke risk."
            )



# ==========================================
# MULTI PREDICTION FUNCTION
# ==========================================

def multi_prediction(uploaded_file):

    model = load_model()
    scaler = load_scaler()


    # ======================================
    # READ CSV FILE
    # ======================================

    try:

        dfinput = pd.read_csv(
            uploaded_file
        )


    except Exception:

        st.error(
            "The uploaded CSV file could not be read.",
            icon="❌"
        )

        return


    # ======================================
    # REMOVE TARGET COLUMN
    # ======================================

    if "stroke" in dfinput.columns:

        dfinput.drop(
            "stroke",
            axis=1,
            inplace=True
        )


    # ======================================
    # REMOVE ID COLUMN
    # ======================================

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


    # ======================================
    # DISPLAY UPLOADED DATA
    # ======================================

    st.subheader(
        "Uploaded Patient Data"
    )


    col1, col2 = st.columns(2)


    with col1:

        st.metric(
            "Patient Records",
            len(dfinput)
        )


    with col2:

        st.metric(
            "Features",
            len(dfinput.columns)
        )


    st.dataframe(
        dfinput,
        use_container_width=True
    )


    # ======================================
    # VALIDATE COLUMN ORDER
    # ======================================

    if (
        list(dfinput.columns)
        != FEATURE_COLUMNS
    ):

        st.error(
            "The uploaded dataset does not contain "
            "the required columns in the correct order.",
            icon="❌"
        )


        st.write(
            "Required column order:"
        )


        for number, column in enumerate(
            FEATURE_COLUMNS,
            start=1
        ):

            st.write(
                f"{number}. {column}"
            )


        return


    # ======================================
    # CHECK MISSING VALUES
    # ======================================

    if dfinput.isnull().any().any():

        missing_columns = (
            dfinput.columns[
                dfinput.isnull().any()
            ].tolist()
        )


        st.error(
            "Missing values were detected in: "
            + ", ".join(missing_columns),
            icon="⚠️"
        )


        return


    # ======================================
    # CONVERT TO NUMERIC DATA
    # ======================================

    try:

        numeric_data = dfinput.astype(
            float
        )


    except ValueError:

        st.error(
            "The uploaded file contains values "
            "that cannot be converted to numbers.",
            icon="❌"
        )

        return


    # ======================================
    # CHECK INVALID NUMERIC VALUES
    # ======================================

    if not np.all(
        np.isfinite(
            numeric_data.values
        )
    ):

        st.error(
            "The uploaded dataset contains invalid numeric values.",
            icon="❌"
        )

        return


    # ======================================
    # SCALE DATA
    # ======================================

    try:

        scaled_data = scaler.transform(
            numeric_data.values
        )


    except Exception:

        st.error(
            "The dataset could not be processed "
            "using the model scaler.",
            icon="❌"
        )

        return


    # ======================================
    # GENERATE MULTIPLE PREDICTIONS
    # ======================================

    if st.button(
        "Generate Predictions",
        type="primary",
        use_container_width=True
    ):

        try:

            prediction = model.predict(
                scaled_data
            )


            prediction = np.asarray(
                prediction
            ).reshape(-1)


            prediction_labels = []


            for result in prediction:

                if int(result) == 1:

                    prediction_labels.append(
                        "Stroke risk detected"
                    )

                else:

                    prediction_labels.append(
                        "No stroke risk detected"
                    )


            # ==================================
            # BUILD RESULT DATAFRAME
            # ==================================

            result_dataframe = pd.DataFrame({

                "Patient ID":
                    np.arange(
                        1,
                        len(prediction_labels) + 1
                    ),

                "Prediction":
                    prediction_labels
            })


            # ==================================
            # DISPLAY RESULTS
            # ==================================

            st.subheader(
                "Prediction Results"
            )


            stroke_count = (
                result_dataframe["Prediction"]
                ==
                "Stroke risk detected"
            ).sum()


            no_stroke_count = (
                len(result_dataframe)
                -
                stroke_count
            )


            col1, col2, col3 = st.columns(3)


            with col1:

                st.metric(
                    "Total Patients",
                    len(result_dataframe)
                )


            with col2:

                st.metric(
                    "Stroke Risk",
                    stroke_count
                )


            with col3:

                st.metric(
                    "No Stroke Risk",
                    no_stroke_count
                )


            st.dataframe(
                result_dataframe,
                use_container_width=True
            )


            # ==================================
            # DOWNLOAD RESULTS
            # ==================================

            csv = result_dataframe.to_csv(
                index=False
            ).encode(
                "utf-8"
            )


            st.download_button(
                label="Download Prediction Results",
                data=csv,
                file_name="stroke_predictions.csv",
                mime="text/csv",
                use_container_width=True
            )


        except Exception as error:

            st.error(
                "The predictions could not be completed.",
                icon="❌"
            )

            st.exception(
                error
            )


# ==========================================
# MULTI PREDICTION PAGE
# ==========================================

def multi_prediction_page():

    st.title(
        "📊 Multiple Patient Prediction"
    )


    uploaded_file = st.file_uploader(
        "Upload Patient Dataset",
        type=["csv"]
    )


    if uploaded_file is not None:

        multi_prediction(
            uploaded_file
        )


# ==========================================
# PAGE ROUTING
# ==========================================

if selection == "Single Prediction":

    single_prediction_page()


elif selection == "Multi Prediction":

    multi_prediction_page()
