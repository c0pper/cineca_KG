import json
from pathlib import Path
import time
from pypdf import PdfReader
from dotenv import load_dotenv
import os
from langchain_community.graphs import Neo4jGraph
from langchain_core.documents import Document
# from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_core.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.prompts.prompt import PromptTemplate
from langchain_openai import ChatOpenAI
from openai import OpenAI
from tqdm import tqdm
from custom_classes import LLMGraphTransformer


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
