import json
from neo4j import GraphDatabase

class Neo4jConnection:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def run_query(self, query, parameters=None):
        with self.driver.session() as session:
            result = session.run(query, parameters)
            return [record for record in result]

def create_graph_from_json(conn, json_data):
    # Clear existing data
    conn.run_query("MATCH (n) DETACH DELETE n")

    # Create Chapter node
    chapter_title = json_data['title']
    chapter_main_idea = json_data['main_idea']
    conn.run_query(
        "CREATE (c:Chapter {title: $title, main_idea: $main_idea})",
        {'title': chapter_title, 'main_idea': chapter_main_idea}
    )

    for sub_chapter in json_data['sub_chapters']:
        sub_title = sub_chapter['sub_title']
        sub_content = sub_chapter['sub_content']
        sub_main_idea = sub_chapter['main_idea']

        # Create SubChapter node
        conn.run_query(
            """
            CREATE (sc:SubChapter {
                title: $sub_title,
                content: $sub_content,
                main_idea: $sub_main_idea
            })
            """,
            {
                'sub_title': sub_title,
                'sub_content': sub_content,
                'sub_main_idea': sub_main_idea
            }
        )

        # Link SubChapter to Chapter
        conn.run_query(
            """
            MATCH (c:Chapter {title: $chapter_title})
            MATCH (sc:SubChapter {title: $sub_title})
            MERGE (c)-[:HAS_SUB_CHAPTER]->(sc)
            """,
            {'chapter_title': chapter_title, 'sub_title': sub_title}
        )

        # Create and link entities
        for entity_name in sub_chapter['entities']:
            conn.run_query(
                "MERGE (e:Entity {name: $name})",
                {'name': entity_name}
            )
            conn.run_query(
                """
                MATCH (sc:SubChapter {title: $sub_title})
                MATCH (e:Entity {name: $entity_name})
                MERGE (sc)-[:MENTIONS_ENTITY]->(e)
                """,
                {'sub_title': sub_title, 'entity_name': entity_name}
            )

        # Create and link keywords
        for keyword_name in sub_chapter['keywords']:
            conn.run_query(
                "MERGE (k:Keyword {name: $name})",
                {'name': keyword_name}
            )
            conn.run_query(
                """
                MATCH (sc:SubChapter {title: $sub_title})
                MATCH (k:Keyword {name: $keyword_name})
                MERGE (sc)-[:HAS_KEYWORD]->(k)
                """,
                {'sub_title': sub_title, 'keyword_name': keyword_name}
            )

if __name__ == "__main__":
    # --- નિયો4j કનેક્શન માહિતી ---
    # કૃપા કરીને ખાતરી કરો કે તમે docker-compose.yml માં સેટ કરેલ વપરાશકર્તા નામ અને પાસવર્ડ સાથે મેળ ખાય છે
    uri = "bolt://localhost:7687"
    user = "neo4j"
    password = "password123"

    conn = Neo4jConnection(uri, user, password)

    with open('data.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    create_graph_from_json(conn, data)

    print("Knowledge graph created successfully in Neo4j.")

    conn.close()
