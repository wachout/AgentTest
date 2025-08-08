import os
import json
from neo4j import GraphDatabase

class Neo4jUploader:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def upload_data(self, data):
        with self.driver.session() as session:
            session.write_transaction(self._create_and_link_data, data)

    @staticmethod
    def _create_and_link_data(tx, data):
        # 1. Create Chapter and its MainIdea
        tx.run("MERGE (c:Chapter {title: $title}) "
               "MERGE (m:MainIdea {text: $main_idea}) "
               "MERGE (c)-[:HAS_MAIN_IDEA]->(m)",
               title=data['title'], main_idea=data['main_idea'])

        # 2. Process SubChapters
        for sub_chapter in data['sub_chapters']:
            # Create SubChapter and its MainIdea
            tx.run("MATCH (c:Chapter {title: $chapter_title}) "
                   "MERGE (sc:SubChapter {sub_title: $sub_title, sub_content: $sub_content}) "
                   "MERGE (m:MainIdea {text: $main_idea}) "
                   "MERGE (c)-[:HAS_SUBCHAPTER]->(sc) "
                   "MERGE (sc)-[:HAS_MAIN_IDEA]->(m)",
                   chapter_title=data['title'], sub_title=sub_chapter['sub_title'],
                   sub_content=sub_chapter['sub_content'], main_idea=sub_chapter['main_idea'])

            # Link Entities
            for entity in sub_chapter['entities']:
                tx.run("MATCH (sc:SubChapter {sub_title: $sub_title}) "
                       "MERGE (e:Entity {name: $entity_name}) "
                       "MERGE (sc)-[:MENTIONS_ENTITY]->(e)",
                       sub_title=sub_chapter['sub_title'], entity_name=entity)

            # Link Keywords
            for keyword in sub_chapter['keywords']:
                tx.run("MATCH (sc:SubChapter {sub_title: $sub_title}) "
                       "MERGE (k:Keyword {name: $keyword_name}) "
                       "MERGE (sc)-[:HAS_KEYWORD]->(k)",
                       sub_title=sub_chapter['sub_title'], keyword_name=keyword)

def main():
    # Load data from the provided JSON file
    json_data = {"title":"第一章 总 则",
      "main_idea":"该文段主要规定了社会保险费的征缴范围、征收管理及相关部门职责，旨在保障社会保险金的发放并规范征缴工作。",
      "sub_chapters":[
        {"sub_content":"第一条 为了加强和规范社会保险费征缴工作，保障社会保险金的发放，制定本条例。","sub_title":"第一条","main_idea":"为了加强和规范社会保险费征缴工作，保障社会保险金的发放，制定了本条例。","entities":["社会保险费","社会保险金","条例"],"keywords":["社会保险费","征缴工作","社会保险金","条例"]},
        {"sub_content":"第二条 基本养老保险费、基本医疗保险费、失业保险费（以下统称社会保险费）的征收、缴纳，适用本条例。\n本条例所称缴费单位、缴费个人，是指依照有关法律、行政法规和国务院的规定，应当缴纳社会保险费的单位和个人。","sub_title":"第二条","main_idea":"本条例规定了社会保险费的征收和缴纳适用本条例，并明确了缴费单位和个人的定义。","entities":["基本养老保险费","基本医疗保险费","失业保险费","社会保险费","条例"],"keywords":["基本养老保险费","基本医疗保险费","失业保险费","社会保险费","征收","缴纳","条例","缴费单位","缴费个人"]}
      ]
    }

    # Neo4j connection details (replace with your actual credentials)
    uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    user = os.environ.get("NEO4J_USER", "neo4j")
    password = os.environ.get("NEO4J_PASSWORD", "password")

    uploader = Neo4jUploader(uri, user, password)
    uploader.upload_data(json_data)
    uploader.close()
    print("Data uploaded to Neo4j successfully!")

if __name__ == "__main__":
    main()
