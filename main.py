from rdflib import Graph, Namespace, Literal
from rdflib.namespace import RDF, FOAF
import os

# Create an RDF Graph
g = Graph()

relations = [
    ("Ashish Vaswani", "person", "works_for", "Google Brain", "organization"),
    ("Noam Shazeer", "person", "works_for", "Google Brain", "organization"),
    ("Niki Parmar", "person", "works_for", "Google Research", "organization"),
    ("Jakob Uszkoreit", "person", "works_for", "Google Research", "organization"),
    ("Llion Jones", "person", "works_for", "Google Research", "organization"),
    ("Aidan N. Gomez", "person", "is_employed_by", "University of Toronto", "organization"),
    ("Łukasz Kaiser", "person", "works_for", "Google Brain", "organization"),
    ("Illia Polosukhin", "person", "None", "None", "None")
]

# Define namespaces
ns = Namespace("http://example.org/")
g.bind("ex", ns)

# Add relations to the RDF Graph
for relation in relations:
    entity1_uri = ns[relation[0].replace(" ", "_")]
    entity2_uri = ns[relation[3].replace(" ", "_")]
    g.add((entity1_uri, RDF.type, ns[relation[1]]))
    g.add((entity2_uri, RDF.type, ns[relation[4]]))
    g.add((entity1_uri, ns[relation[2]], entity2_uri))

# Serialize the RDF Graph to a file
g.serialize(destination='knowledge_base.rdf', format='xml')