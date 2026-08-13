# ==========================================
# STROKE PREDICTION SYSTEM
# STREAMLIT APPLICATION
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
# MODEL FILES
# ==========================================

MODEL_FILE = "strokedetect.sav"
SCALER_FILE = "my_standard_scaler.pkl"


# ==========================================
# FEATURE ORDER
# Must match model training
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
# DOWNLOAD FUNCTION
# ==========================================

def filedownload(df):

    csv = df.to_csv(index=False)

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
# STROKE PREDICTION FUNCTION
# ==========================================

def stroke_detect(given_data):

    model = load_model()
    scaler = load_scaler()

    # Convert patient data to NumPy array
    input_data = np.asarray(
        given_data,
        dtype=float
    )

    # Reshape for one patient
    input_data = input_data.reshape(
        1,
        -1
    )

    # Ensure correct number of features
    if input_data.shape[1] != 10:

        raise ValueError(
            "The model requires exactly 10 input features."
        )

    # Check for invalid values
    if not np.all(np.isfinite(input_data)):

        raise ValueError(
            "The patient information contains invalid numeric values."
        )

    # Scale using the same scaler used during training
    scaled_input = scaler.transform(
        input_data
    )

    # Prediction
    prediction = model.predict(
        scaled_input
    )[0]

    return int(prediction)


# ==========================================
# SIDEBAR
# ==========================================

with st.sidebar:

    st.title("🧠 StrokeDetect")

    st.caption(
        "Machine Learning Stroke Prediction System"
    )

    st.divider()

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

        default_index=0
    )

    st.divider()

    st.caption(
        "Prediction results are generated "
        "by a machine-learning model."
    )


# ==========================================
# SINGLE PREDICTION PAGE
# ==========================================

def single_prediction_page():

    # ======================================
    # HEADER
    # ======================================

    st.title("🧠 Stroke Prediction System")

    st.caption(
        "Enter the patient's demographic and clinical "
        "information to generate a stroke prediction."
    )

    st.divider()


    # ======================================
    # INFORMATION
    # ======================================

    st.info(
        "Complete all patient information below. "
        "All fields are required before a prediction "
        "can be generated.",
        icon="ℹ️"
    )


    # ======================================
    # PATIENT FORM
    # ======================================

    st.subheader("Patient Information")

    st.caption(
        "Provide the patient's information in the fields below."
    )


    with st.form(
        "stroke_prediction_form"
    ):


        # ==================================
        # SECTION 1
        # DEMOGRAPHIC INFORMATION
        # ==================================

        st.markdown("#### Demographic Information")


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


        st.divider()


        # ==================================
        # SECTION 2
        # MEDICAL INFORMATION
        # ==================================

        st.markdown("#### Medical Information")


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


        submitted = st.form_submit_button(
            "Generate Prediction",
            type="primary",
            use_container_width=True
        )


    # ======================================
    # VALIDATION
    # ======================================

    if submitted:

        fields = {

            "Patient Age": age,

            "Sex": option1,

            "Hypertension": option4,

            "Heart Disease": option5,

            "Ever Married": marriage,

            "Work Type": option6,

            "Residence Type": option7,

            "Average Glucose Level": glucose,

            "BMI": bmi,

            "Smoking Status": smoking_status
        }


        missing_fields = [

            field

            for field, value in fields.items()

            if value is None
        ]


        # ==================================
        # MISSING FIELD VALIDATION
        # ==================================

        if missing_fields:

            st.error(
                "Please complete the following field(s): "
                + ", ".join(missing_fields),
                icon="⚠️"
            )

            return


        # ==================================
        # NUMERIC VALIDATION
        # ==================================

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

            for error in validation_errors:

                st.error(
                    error,
                    icon="⚠️"
                )

            return


        # ==================================
        # ENCODE CATEGORICAL VARIABLES
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
        # BUILD MODEL INPUT
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
        # MAKE PREDICTION
        # ==================================

        try:

            prediction = stroke_detect(
                patient_data
            )


            st.divider()

            st.subheader(
                "Prediction Result"
            )


            if prediction == 1:

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


            st.warning(
                "This result is generated by a machine-learning "
                "model and should not be considered a medical diagnosis.",
                icon="⚕️"
            )


        except Exception as error:

            st.error(
                "The prediction could not be completed.",
                icon="❌"
            )

            st.exception(
                error
            )


# ==========================================
# MULTI PREDICTION FUNCTION
# ==========================================

def multi_prediction(
    uploaded_file
):

    model = load_model()
    scaler = load_scaler()


    # ======================================
    # READ DATA
    # ======================================

    try:

        dfinput = pd.read_csv(
            uploaded_file
        )

    except Exception as error:

        st.error(
            "The uploaded CSV file could not be read.",
            icon="❌"
        )

        return


    # ======================================
    # REMOVE STROKE COLUMN
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
    # DATASET SUMMARY
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

    if list(dfinput.columns) != FEATURE_COLUMNS:

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
    # CONVERT TO NUMERIC
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
    # INVALID NUMBERS
    # ======================================

    if not np.all(
        np.isfinite(
            numeric_data.values
        )
    ):

        st.error(
            "The uploaded dataset contains invalid "
            "numeric values.",
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


    except Exception as error:

        st.error(
            "The dataset could not be processed "
            "using the model scaler.",
            icon="❌"
        )

        return


    # ======================================
    # PREDICT
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
            # RESULTS
            # ==================================

            result_dataframe = pd.DataFrame({

                "Patient ID":
                    np.arange(
                        1,
                        len(
                            prediction_labels
                        ) + 1
                    ),

                "Prediction":
                    prediction_labels

            })


            st.divider()

            st.subheader(
                "Prediction Results"
            )


            # ==================================
            # SUMMARY
            # ==================================

            stroke_count = (
                result_dataframe[
                    "Prediction"
                ]
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
            # DOWNLOAD
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


            st.warning(
                "These results are generated by a machine-learning "
                "model and should not be considered medical diagnoses.",
                icon="⚕️"
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

    st.caption(
        "Upload patient data and generate predictions "
        "for multiple patients at once."
    )

    st.divider()


    st.info(
        "Upload a CSV file containing the 10 features "
        "required by the stroke prediction model.",
        icon="ℹ️"
    )


    uploaded_file = st.file_uploader(

        "Patient Dataset",

        type=[
            "csv"
        ],

        help=(
            "Upload a CSV file containing patient "
            "information."
        )

    )


    if uploaded_file is not None:

        multi_prediction(
            uploaded_file
        )


    else:

        st.write("")

        st.subheader(
            "No dataset uploaded"
        )

        st.caption(
            "Choose a CSV file above to start "
            "multiple-patient prediction."
        )


# ==========================================
# ROUTING
# ==========================================

if selection == "Single Prediction":

    single_prediction_page()


elif selection == "Multi Prediction":

    multi_prediction_page()
