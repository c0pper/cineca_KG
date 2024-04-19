import rdflib
import rdfextras

# Register the RDF extensions
rdfextras.registerplugins()

# Load the RDF data
filename = "knowledge_base.rdf"
g = rdflib.Graph()
g.parse(filename)

# Define the SPARQL query
sparql_query = """
PREFIX ex: <http://example.org/>

SELECT ?person
WHERE {
  ?person rdf:type ex:person .
  ?person ex:works_for ex:Google_Brain .
}
"""

# Execute the SPARQL-like query
results = g.query(sparql_query)

# Print the results
for row in results:
    print(row.person)
