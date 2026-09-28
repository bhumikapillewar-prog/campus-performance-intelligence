import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix


# -------------------------------------------------
# PAGE CONFIGURATION
# -------------------------------------------------

st.set_page_config(
    page_title="Campus Performance Intelligence",
    page_icon="🎓",
    layout="wide"
)


# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------

@st.cache_data
def load_data():
    return pd.read_csv("student_data.csv")


df = load_data()


# -------------------------------------------------
# TRAIN MACHINE LEARNING MODEL
# -------------------------------------------------

features = [
    "Attendance",
    "Study_Hours",
    "Previous_Marks",
    "Assignment_Score",
    "Internal_Marks",
    "Participation"
]

X = df[features]

encoder = LabelEncoder()
y = encoder.fit_transform(df["Performance"])

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = DecisionTreeClassifier(
    max_depth=4,
    random_state=42
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)


# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------

st.sidebar.title("🎓 Campus Intelligence")

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Student Prediction",
        "Data Analysis",
        "Model Insights"
    ]
)


# -------------------------------------------------
# HEADER
# -------------------------------------------------

st.title("🎓 Campus Performance Intelligence")

st.markdown(
    """
    **Student Performance Prediction & Analytics System**

    This application uses data mining and machine learning
    to analyze student performance and predict performance categories.
    """
)


# =================================================
# DASHBOARD
# =================================================

if page == "Dashboard":

    st.subheader("📊 Performance Dashboard")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Students",
        len(df)
    )

    col2.metric(
        "Average Attendance",
        f"{df['Attendance'].mean():.1f}%"
    )

    col3.metric(
        "Average Marks",
        f"{df['Previous_Marks'].mean():.1f}"
    )

    col4.metric(
        "Model Accuracy",
        f"{accuracy * 100:.1f}%"
    )

    st.divider()

    col1, col2 = st.columns(2)

    with col1:

        performance_count = (
            df["Performance"]
            .value_counts()
            .reset_index()
        )

        performance_count.columns = [
            "Performance",
            "Students"
        ]

        fig = px.bar(
            performance_count,
            x="Performance",
            y="Students",
            title="Students by Performance Category",
            text="Students"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        fig = px.scatter(
            df,
            x="Study_Hours",
            y="Previous_Marks",
            color="Performance",
            hover_data=["Student_ID"],
            title="Study Hours vs Previous Marks"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.subheader("📋 Student Dataset")

    st.dataframe(
        df,
        use_container_width=True
    )


# =================================================
# STUDENT PREDICTION
# =================================================

elif page == "Student Prediction":

    st.subheader("🤖 Student Performance Prediction")

    st.write(
        "Enter student information to predict the performance category."
    )

    col1, col2 = st.columns(2)

    with col1:

        attendance = st.slider(
            "Attendance (%)",
            0,
            100,
            75
        )

        study_hours = st.slider(
            "Study Hours per Day",
            0.0,
            12.0,
            4.0,
            0.5
        )

        previous_marks = st.slider(
            "Previous Marks",
            0,
            100,
            70
        )

    with col2:

        assignment_score = st.slider(
            "Assignment Score",
            0,
            100,
            70
        )

        internal_marks = st.slider(
            "Internal Assessment Marks",
            0,
            100,
            70
        )

        participation = st.slider(
            "Class Participation",
            0,
            10,
            6
        )

    if st.button(
        "🔮 Predict Performance",
        use_container_width=True
    ):

        input_data = pd.DataFrame(
            [[
                attendance,
                study_hours,
                previous_marks,
                assignment_score,
                internal_marks,
                participation
            ]],
            columns=features
        )

        prediction = model.predict(input_data)[0]

        probability = model.predict_proba(input_data)[0]

        result = encoder.inverse_transform(
            [prediction]
        )[0]

        confidence = probability.max() * 100

        st.divider()

        if result == "Excellent":

            st.success(
                f"🎉 Predicted Performance: {result}"
            )

        elif result == "Good":

            st.info(
                f"👍 Predicted Performance: {result}"
            )

        elif result == "Average":

            st.warning(
                f"📚 Predicted Performance: {result}"
            )

        else:

            st.error(
                f"⚠️ Predicted Performance: {result}"
            )

        st.metric(
            "Prediction Confidence",
            f"{confidence:.1f}%"
        )


# =================================================
# DATA ANALYSIS
# =================================================

elif page == "Data Analysis":

    st.subheader("📈 Student Data Analysis")

    st.write(
        "Explore relationships between different student attributes."
    )

    selected_feature = st.selectbox(
        "Select feature",
        [
            "Attendance",
            "Study_Hours",
            "Previous_Marks",
            "Assignment_Score",
            "Internal_Marks",
            "Participation"
        ]
    )

    fig = px.histogram(
        df,
        x=selected_feature,
        color="Performance",
        title=f"{selected_feature} Distribution"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader("📊 Attendance vs Performance")

    fig = px.box(
        df,
        x="Performance",
        y="Attendance",
        color="Performance",
        title="Attendance Distribution by Performance"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader("📚 Study Hours vs Performance")

    fig = px.box(
        df,
        x="Performance",
        y="Study_Hours",
        color="Performance",
        title="Study Hours Distribution by Performance"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# =================================================
# MODEL INSIGHTS
# =================================================

elif page == "Model Insights":

    st.subheader("🧠 Decision Tree Model Insights")

    st.write(
        "The model identifies which student factors contribute "
        "most to the prediction."
    )

    importance = pd.DataFrame({
        "Feature": features,
        "Importance": model.feature_importances_
    })

    importance = importance.sort_values(
        "Importance",
        ascending=False
    )

    fig = px.bar(
        importance,
        x="Importance",
        y="Feature",
        orientation="h",
        title="Feature Importance"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader("📋 Feature Importance")

    st.dataframe(
        importance,
        use_container_width=True
    )

    st.metric(
        "Decision Tree Accuracy",
        f"{accuracy * 100:.2f}%"
    )

    st.info(
        """
        **Algorithm Used:** Decision Tree Classification

        The model learns patterns from student attendance,
        study hours, previous marks, assignment scores,
        internal marks and participation.
        """
    )


# -------------------------------------------------
# FOOTER
# -------------------------------------------------

st.divider()

st.caption(
    "Campus Performance Intelligence | Python + Scikit-learn + Streamlit"
)