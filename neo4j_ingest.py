from pathlib import Path
from pypdf import PdfReader
from dotenv import load_dotenv
import os
from langchain_community.graphs import Neo4jGraph
from langchain_core.documents import Document
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_openai import ChatOpenAI
from langchain.chains import GraphCypherQAChain


os.environ["NEO4J_URI"] = "bolt://localhost:7687"
os.environ["NEO4J_USERNAME"] = "neo4j"
os.environ["NEO4J_PASSWORD"] = "H3110_W0r1d"

load_dotenv()

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

pdfs = Path(".").glob("*.pdf")

texts = []
for f in pdfs:
    t = extract_text_before_keywords(f)
    texts.append(t)
print(texts)


documents = [Document(page_content=text) for text in texts]

llm = ChatOpenAI(temperature=0, model_name="gpt-4-0125-preview")

llm_transformer = LLMGraphTransformer(
    llm=llm,
    allowed_nodes=["Person", "Paper Title", "Organization"],
    allowed_relationships=["WORKS_FOR", "AUTHOR_OF_PAPER"],
)

graph_documents = llm_transformer.convert_to_graph_documents(documents)
print(f"Nodes:{graph_documents[0].nodes}")
print(f"Relationships:{graph_documents[0].relationships}")

graph.add_graph_documents(graph_documents)

graph.refresh_schema()
