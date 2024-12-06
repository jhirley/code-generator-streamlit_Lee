import streamlit as st

from forms.contact import contact_form


@st.dialog("📨 Contact Me")
def show_contact_form():
    contact_form()


# --- HERO SECTION ---
col1, col2 = st.columns(2, gap="small", vertical_alignment="center")
with col1:
    st.image("https://avatars.githubusercontent.com/u/8917484?s=96&v=2", width=230)

with col2:
    st.title("Jay Fonte", anchor=False)
    st.write(
        "Changing the world with data-driven decision-making."
    )
    if st.button("📨 Contact Me"):
        show_contact_form()


# --- EXPERIENCE & QUALIFICATIONS ---
st.write("\n")
st.subheader("Experience & Qualifications", anchor=False)
st.write(
    """
    - 3 Years experience extracting actionable insights from data
    - Strong hands-on experience and knowledge in Python and R
    - Good understanding of statistical principles and their respective applications
    - Team First mentality, Self starter, strong sense of initiative and ownership.
    """
)

# --- SKILLS ---
st.write("\n")
st.subheader("Hard Skills", anchor=False)
st.write(
    """
    - Programming: Python (LangChain, Scikit-learn, Tensorflow, Pandas), SQL, R
    - Platform: jypyter notebook, Google Colab, databricks 
    - Data Visualization: PowerBi, Tableau, Matplotlib, Seaborn
    - Modeling: Neural networks, Clustering, Logistic regression, linear regression, decision trees, 
    - Databases: Postgres, MongoDB, Mariadb
    - Cloud: AWS, GCP, kubernetes, Docker
    """
)

st.write("\nhttps://github.com/Sven-Bo/python-multipage-webapp/")