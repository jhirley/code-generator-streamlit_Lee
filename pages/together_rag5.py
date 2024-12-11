#"Local RAG agent with LLaMA3"
import sys
import os
import re
import json
import streamlit as st
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from together import Together
import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer


from src.keys.keys import get_api_key
# from langchain_ollama import ChatOllama

from together import Together
# get the api key
together_api_key = get_api_key('TOGETHER_API_KEY')
nomic_api_key = get_api_key('NOMIC_API_KEY')

# Initialize Together client
client = Together()

model  = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"

   # Function to extract URLs from text
def extract_urls(text):
    url_pattern = re.compile(r'https?://\S+')
    return url_pattern.findall(text) 

# Function to fetch and extract text from URL
def extract_text_from_url(url):
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, "html.parser")
        # Extract the text from the page, excluding non-content parts (like scripts, ads)
        return ' '.join([p.get_text() for p in soup.find_all('p')])
    else:
        print(f"Failed to fetch {url}")
        return ""

st.title("Together.ai RAG w/ meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo")
st.write("This is a remote Retrieval-Augmented Generation (RAG) agent with LLaMA3. The agent can answer questions using a vectorstore or web search.")
st.write("This work is based off of the paper: 'Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks' Authors: Patrick Lewis, Ethan Perez, Aleksandra Piktus, et al. Published: 2020. URL: https://arxiv.org/abs/2005.11401")
st.write('This work contribute to the #AIforSocalGood initiative by providing a tool that can help people get information about the country they want to travel to.')
st.write("Helps travelers and attorneys create a travel plan based on safety, health, and environmental conditions and evaluate risk for the traveler. ")

# Input box for the user to enter a description
the_question = st.text_area("Ask the AI", placeholder="Ask the AI about the country your client is considering travel?")

# Button to trigger code generation
if st.button("Submit"):

    def _set_env(var: str):
        if not os.environ.get(var):
            key = get_api_key(var)
            st.write(f"Setting {var} environment variable to {key}")
            os.environ[var] = key


    _set_env("TAVILY_API_KEY")
    # _set_env("LANGSMITH_API_KEY")
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = "local-llama32-rag"



    ### Search
    from langchain_community.tools.tavily_search import TavilySearchResults

    web_search_tool = TavilySearchResults(k=3)

    ### LLM

    ## Vector store
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.document_loaders import WebBaseLoader
    from langchain_community.vectorstores import SKLearnVectorStore
    from langchain_nomic.embeddings import NomicEmbeddings

    urls = [
        "https://travel.state.gov/content/travel/en/international-travel.html",
        "https://travel.state.gov/content/travel/en/traveladvisories/traveladvisories.html",
        "https://www.osac.gov",
        "https://www.osac.gov/Country",
        "https://www.cbp.gov",
        "https://www.iata.org",
        "https://www.travel-advisory.info",
        "https://travel.state.gov/content/travel/en/international-travel/International-Travel-Country-Information-Pages/Canada.html", 
        "https://travel.state.gov/content/travel/en/international-travel/International-Travel-Country-Information-Pages/Brazil.html",
        "https://travel.state.gov/content/travel/en/international-travel/International-Travel-Country-Information-Pages/Egypt.html",
        "https://travel.state.gov/content/travel/en/international-travel/International-Travel-Country-Information-Pages/IsraeltheWestBankandGaza.html?wcmmode=disabled",
        "https://travel.state.gov/content/travel/en/international-travel/International-Travel-Country-Information-Pages/Thailand.html",
        "https://travel.state.gov/content/travel/en/international-travel/International-Travel-Country-Information-Pages/Ukraine.html",
        # "",
        # "",
        # "",
    ]

    # Load documents
    documents = []
    for url in urls:
        text = extract_text_from_url(url)
        if text:
            documents.append(Document(text=text))

    # Step 2: Initialize the TF-IDF vectorizer
    vectorizer = TfidfVectorizer()

    # Step 3: Transform the documents into TF-IDF vectors
    doc_texts = [doc.text for doc in documents]  # Extract the text from the Document objects
    X = vectorizer.fit_transform(doc_texts).toarray()  # Convert the documents to vectors

    # Step 4: Create the SKLearnVectorStore
    vector_store = SKLearnVectorStore.from_documents(documents, vectorizer)

    # Step 5: Query the vector store
    query = "What is machine learning?"
    query_vector = vectorizer.transform([query]).toarray()

    # Perform similarity search (find the most similar document to the query)
    retriever = vector_store.similarity_search(query_vector[0], k=3)
    # Create retriever
    

    # st.write(retriever.invoke("agent memory"))

    # Router, bases on a classifier
    ### Router
    import json
    from langchain_core.messages import HumanMessage, SystemMessage

    # Prompt
    router_instructions = """You are an expert at routing a user question to a vectorstore or web search.

    The vectorstore contains documents related to agents, prompt engineering, and adversarial attacks.

    Use the vectorstore for questions on these topics. For all else, and especially for current events, use web-search.

    Return JSON with single key, datasource, that is 'websearch' or 'vectorstore' depending on the question."""



    # llm_json_mode = ChatOllama(model=local_llm, temperature=0, format="json")


    # Test router

    ### Retrieval Grader

    # Doc grader instructions
    doc_grader_instructions = """You are a grader assessing relevance of a retrieved document to a user question.

    If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant."""

    # Grader prompt
    doc_grader_prompt = """Here is the retrieved document: \n\n {document} \n\n Here is the user question: \n\n {question}. 

    This carefully and objectively assess whether the document contains at least some information that is relevant to the question.

    Return JSON with single key, binary_score, that is 'yes' or 'no' score to indicate whether the document contains at least some information that is relevant to the question."""

    # Test
    question = the_question
    docs = retriever.invoke(question)
    doc_txt = docs[1].page_content
    doc_grader_prompt_formatted = doc_grader_prompt.format(
        document=doc_txt, question=question
    )



    ### Generate
    context = "you are evauleate the safety, the country conditions, all the travel adivisories but consider them from most recent as most important and oldest as the least important and the Travel Advisory Levels for the country requested"

    # Prompt
    rag_prompt = """You are an assistant for question-answering tasks. 

    Here is the context to use to answer the question:

    {context} 

    Think carefully about the above context. 

    Now, review the user question:

    {question}

    Provide an answer to this questions using only the above context. 

    Use three sentences maximum and keep the answer concise.
    Provide the source of the answer if possible and a URL.
    Answer:"""


    # Post-processing
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Test
    docs = retriever.invoke(question)
    docs_txt = format_docs(docs)
    rag_prompt_formatted = rag_prompt.format(context=docs_txt, question=question)
 
    llm_response = client.chat.completions.create(
        model=model,
        temperature=0.0,
        messages=[
            {"role": "system", "content": rag_prompt_formatted},
            {"role": "user", "content": the_question},
        ],
    )

    generation = llm_response.choices[0].message.content
    
    st.write('\n\n')
    st.write('### Generate')
    st.write(generation)

    ### Function to extract URLs from text
    urls_to_add = []
    urls_to_add = extract_urls(generation)

    if urls_to_add:
        print("URLs extracted from the answer:")
        for url in urls_to_add:
            print(url)

    ### Hallucination Grader

    # Hallucination grader instructions
    hallucination_grader_instructions = """

    You are a teacher grading a quiz. 

    You will be given FACTS and a ANSWER. 

    Here is the grade criteria to follow:

    (1) Ensure the ANSWER is grounded in the FACTS. 

    (2) Ensure the ANSWER does not contain "hallucinated" information outside the scope of the FACTS.

    Score:

    A score of yes means that the answer meets all of the criteria. This is the highest (best) score. 

    A score of no means that the answer does not meet all of the criteria. This is the lowest possible score you can give.

    Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. 

    Avoid simply stating the correct answer at the outset."""

    # Grader prompt
    hallucination_grader_prompt = """FACTS: \n\n {documents} \n\n ANSWER: {generation}. 

    Return JSON with two two keys, binary_score is 'yes' or 'no' score to indicate whether the ANSWER is grounded in the FACTS. And a key, explanation, that contains an explanation of the score."""

    # Test using documents and generation from above
    hallucination_grader_prompt_formatted = hallucination_grader_prompt.format(
        documents=docs_txt, generation=generation
    )
    llm_response = client.chat.completions.create(
        model=model,
        temperature=0.0,
        messages=[
            {"role": "system", "content": hallucination_grader_instructions},
            {"role": "user", "content": hallucination_grader_prompt_formatted},
        ],
    )

    result = llm_response.choices[0].message.content

    st.write('\n\n')
    st.write('### Hallucination Grader')
    st.write(result)