import json
from pathlib import Path
import time
from pypdf import PdfReader
from dotenv import load_dotenv
import os
from langchain_community.graphs import Neo4jGraph
from langchain_core.documents import Document
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_core.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.prompts.prompt import PromptTemplate
from langchain_openai import ChatOpenAI
from openai import OpenAI
from tqdm import tqdm


os.environ["NEO4J_URI"] = "bolt://localhost:7687"
os.environ["NEO4J_USERNAME"] = "neo4j"
os.environ["NEO4J_PASSWORD"] = "H3110_W0r1d"

load_dotenv()
oai = OpenAI()

graph = Neo4jGraph() # apoc issues : https://github.com/langchain-ai/langchain/issues/12901

def extract_text_before_keywords(pdf_path, keywords=None):
    if not keywords:
        keywords = ["abstract", "introduction"]
    text = ""
    with open(pdf_path, "rb") as f:
        reader = PdfReader(f)
        page = reader.pages[0]
        content = page.extract_text()
        index = len(content)
        for keyword in keywords:
            keyword_index = content.lower().find(keyword.lower())
            if keyword_index != -1 and keyword_index < index:
                index = keyword_index
        text = content[:index]
    preface = "The following is information about a single academic paper:\n\n"
    return preface + text

def structure_text(text):
    prompt = """
    {text}
    
    Based on the above snippet produce a text with the following plain text structure:
    Paper title: <paper title>

    Author name: <name of the author>
    Author organisation: <organisation the author is affiliated with>

    Author name: <name of the author>
    Author organisation: <organisation the author is affiliated with>
    ...
    """
    prompt = prompt.format(text=text)
    response = oai.chat.completions.create(
        model="gpt-4-0125-preview",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    
    return response.choices[0].message.content

pdfs = Path("pdfs").glob("*.pdf")

texts = []
for f in tqdm(list(pdfs)):
    
    print(F"##### PROCESSING {f.name} #####")
    t = structure_text(extract_text_before_keywords(f))
    print(t)
    texts.append(t)
print(texts)


documents = [Document(page_content=text) for text in texts]

llm = ChatOpenAI(temperature=0, model_name="gpt-4-0125-preview")

llm_transformer = LLMGraphTransformer(
    llm=llm,
    allowed_nodes=["Person", "Paper", "Organisation"],
    allowed_relationships=["WORKS_FOR", "AUTHOR_OF"],
    prompt= ChatPromptTemplate(
        input_variables=['input'], 
        messages=[
            SystemMessagePromptTemplate(
                prompt=PromptTemplate(
                    input_variables=[], 
                    template="""
#Knowledge Graph Instructions for Academic Papers
##1. Overview
You are a top-tier algorithm designed for extracting information in structured formats to build a knowledge graph.
This knowledge graph aims to represent structured information from academic papers to provide clear and accessible insights. 
** Nodes**  represent entities and concepts found in the papers. 
** Relationships ** denote connections between them.
A user input will feature a paper title and all the authors with their affiliated organisation. All of the provided information will be represented in the knowledge graph.

##2. Labeling Nodes

    Consistency: Use basic types for node labels. For example, label authors as 'author' instead of more specific terms like 'researcher'.
    Node IDs: Utilize names or human-readable identifiers found in the text as node IDs. Avoid using integers.

##3. Coreference Resolution

    Maintain Entity Consistency: Ensure consistency in entity references. If an entity is mentioned by different names or pronouns, always use the most complete identifier throughout the knowledge graph.

##4. Strict Compliance

Adhere strictly to the rules provided. Non-compliance may lead to termination."""
                    )
                ), 
            HumanMessagePromptTemplate(
                prompt=PromptTemplate(
                    input_variables=['input'], 
                    template='Tip: Make sure to answer in the correct format and do not include any explanations. Use the given format to extract information from the following input: {input}'
                )
            )
        ]
    )
)

print("##### ADDING DOCUMENTS TO GRAPH #####")
start_time = time.time()
graph_documents = llm_transformer.convert_to_graph_documents(documents)
end_time = time.time()
execution_time = end_time - start_time


print(f"Nodes:{graph_documents[0].nodes}")
print(f"Relationships:{graph_documents[0].relationships}")
print("Execution time:", execution_time, "seconds")

graph.add_graph_documents(graph_documents)

graph.refresh_schema()
