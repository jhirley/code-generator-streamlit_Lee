import streamlit as st


# --- PAGE SETUP ---
about_page = st.Page(
    "pages/About_me.py",
    title="About Me",
    icon='🔥',
    default=True,
)
project_1_page = st.Page(
    "pages/Code_gen.py",
    title="CodeLlama Code Generator",
    icon="👋",
)
project_2_page = st.Page(
    "pages/Summaries.py",
    title="Llama-3.1 8B Video Summaries",
    icon='🛡️',
)
project_3_page = st.Page(
    "pages/local_rag.py",
    title="Local RAG agent with LLaMA3",
    icon='🎁',
)

# --- NAVIGATION SETUP [WITHOUT SECTIONS] ---
# pg = st.navigation(pages=[about_page, project_1_page, project_2_page])

# --- NAVIGATION SETUP [WITH SECTIONS]---
pg = st.navigation(
    {
        "Info": [about_page],
        "LLM Projects": [project_1_page, project_2_page]#, project_3_page],
    }
)

# --- SHARED ON ALL PAGES ---
st.logo("https://avatars.githubusercontent.com/u/8917484?s=96&v=4")
st.sidebar.markdown("Created by [Jhirley](https://github.com/jhirley/)")

# --- RUN NAVIGATION ---
pg.run()