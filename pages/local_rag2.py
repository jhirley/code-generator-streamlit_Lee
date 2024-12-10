#"Local RAG agent with LLaMA3"
import sys
import os
import re
import streamlit as st
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from src.keys.keys import get_api_key

   # Function to extract URLs from text
def extract_urls(text):
    url_pattern = re.compile(r'https?://\S+')
    return url_pattern.findall(text) 


st.title("Local RAG w/ LLaMA3")
st.write("This is a local RAG agent with LLaMA3. The agent can answer questions using a vectorstore or web search.")

# Input box for the user to enter a description
the_question = st.text_area("Ask the AI", placeholder="Ask the AI about the country you to travel to?")

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
    from langchain_ollama import ChatOllama

    local_llm = "llama3.2:3b-instruct-fp16"
    llm = ChatOllama(model=local_llm, temperature=0)
    llm_json_mode = ChatOllama(model=local_llm, temperature=0, format="json")

    ## Vector store
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.document_loaders import WebBaseLoader
    from langchain_community.vectorstores import SKLearnVectorStore
    from langchain_nomic.embeddings import NomicEmbeddings

    urls = [
        "https://travel.state.gov/content/travel/en/international-travel.html",
        "https://travel.state.gov/content/travel/en/traveladvisories/traveladvisories.html",
        "https://travel.state.gov/content/travel/en/international-travel/International-Travel-Country-Information-Pages/Canada.html", 
    ]

    # Load documents
    docs = [WebBaseLoader(url).load() for url in urls]
    docs_list = [item for sublist in docs for item in sublist]

    # Split documents
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000, chunk_overlap=200
    )
    doc_splits = text_splitter.split_documents(docs_list)

    # Add to vectorDB
    vectorstore = SKLearnVectorStore.from_documents(
        documents=doc_splits,
        embedding=NomicEmbeddings(model="nomic-embed-text-v1.5", inference_mode="local"),
    )

    # Create retriever
    retriever = vectorstore.as_retriever(k=3)

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





    # Test router
    test_web_search = llm_json_mode.invoke(
        [SystemMessage(content=router_instructions)]
        + [
            HumanMessage(
                content=the_question
            )
        ]
    )
    
    st.write('### Router')
    st.write(
        json.loads(test_web_search.content),

    )
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
    result = llm_json_mode.invoke(
        [SystemMessage(content=doc_grader_instructions)]
        + [HumanMessage(content=doc_grader_prompt_formatted)]
    )
    st.write('### Retrieval Grader')
    st.write(json.loads(result.content) )



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
    generation = llm.invoke([HumanMessage(content=rag_prompt_formatted)])
    st.write('\n\n')
    st.write('### Generate')
    st.write(generation.content)

    ### Function to extract URLs from text
    urls_to_add = []
    urls_to_add = extract_urls(generation.content)

    if urls_to_add:
        print("URLs extracted from the answer:")
        for url in urls_to_add:
            print(url)

    ### Hallucination Grader

    # Hallucination grader instructions
    hallucination_grader_instructions = """

    You are a teacher grading a quiz. 

    You will be given FACTS and a STUDENT ANSWER. 

    Here is the grade criteria to follow:

    (1) Ensure the STUDENT ANSWER is grounded in the FACTS. 

    (2) Ensure the STUDENT ANSWER does not contain "hallucinated" information outside the scope of the FACTS.

    Score:

    A score of yes means that the student's answer meets all of the criteria. This is the highest (best) score. 

    A score of no means that the student's answer does not meet all of the criteria. This is the lowest possible score you can give.

    Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. 

    Avoid simply stating the correct answer at the outset."""

    # Grader prompt
    hallucination_grader_prompt = """FACTS: \n\n {documents} \n\n STUDENT ANSWER: {generation}. 

    Return JSON with two two keys, binary_score is 'yes' or 'no' score to indicate whether the STUDENT ANSWER is grounded in the FACTS. And a key, explanation, that contains an explanation of the score."""

    # Test using documents and generation from above
    hallucination_grader_prompt_formatted = hallucination_grader_prompt.format(
        documents=docs_txt, generation=generation.content
    )
    result = llm_json_mode.invoke(
        [SystemMessage(content=hallucination_grader_instructions)]
        + [HumanMessage(content=hallucination_grader_prompt_formatted)]
    )
    st.write('\n\n')
    st.write('### Hallucination Grader')
    st.write(json.loads(result.content))

    # ### Answer Grader

    # # Answer grader instructions
    # answer_grader_instructions = """You are a teacher grading a quiz. 

    # You will be given a QUESTION and a STUDENT ANSWER. 

    # Here is the grade criteria to follow:

    # (1) The STUDENT ANSWER helps to answer the QUESTION

    # Score:

    # A score of yes means that the student's answer meets all of the criteria. This is the highest (best) score. 

    # The student can receive a score of yes if the answer contains extra information that is not explicitly asked for in the question.

    # A score of no means that the student's answer does not meet all of the criteria. This is the lowest possible score you can give.

    # Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. 

    # Avoid simply stating the correct answer at the outset."""

    # # Grader prompt
    # answer_grader_prompt = """QUESTION: \n\n {question} \n\n STUDENT ANSWER: {generation}. 

    # Return JSON with two two keys, binary_score is 'yes' or 'no' score to indicate whether the STUDENT ANSWER meets the criteria. And a key, explanation, that contains an explanation of the score."""

    # # Test
    # question = the_question
    # # answer = "The Llama 3.2 models released today include two vision models: Llama 3.2 11B Vision Instruct and Llama 3.2 90B Vision Instruct, which are available on Azure AI Model Catalog via managed compute. These models are part of Meta's first foray into multimodal AI and rival closed models like Anthropic's Claude 3 Haiku and OpenAI's GPT-4o mini in visual reasoning. They replace the older text-only Llama 3.1 models."

    # # Test using question and generation from above
    # answer_grader_prompt_formatted = answer_grader_prompt.format(
    #     question=question, generation=the_question
    # )
    # result = llm_json_mode.invoke(
    #     [SystemMessage(content=answer_grader_instructions)]
    #     + [HumanMessage(content=answer_grader_prompt_formatted)]
    # )
    # st.write('\n\n')
    # st.write('### Answer Grader')
    # st.write(json.loads(result.content))

