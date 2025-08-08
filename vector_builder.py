import os
import json
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
from transformers import BertTokenizer, BertModel
import torch

class MilvusManager:
    def __init__(self, host='localhost', port='19530'):
        self.host = host
        self.port = port
        connections.connect("default", host=self.host, port=self.port)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        self.model = BertModel.from_pretrained('bert-base-chinese')
        self.collection = None

    def create_collection(self, collection_name, dim=768):
        if utility.has_collection(collection_name):
            self.collection = Collection(collection_name)
            return

        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="sub_title", dtype=DataType.VARCHAR, max_length=200),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim)
        ]
        schema = CollectionSchema(fields=fields, description="Text embeddings")
        self.collection = Collection(name=collection_name, schema=schema)

        index_params = {
            "metric_type": "L2",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128}
        }
        self.collection.create_index(field_name="embedding", index_params=index_params)
        self.collection.load()

    def insert_data(self, sub_chapters):
        if not self.collection:
            raise Exception("Collection not created or loaded.")

        sub_titles = [item['sub_title'] for item in sub_chapters]
        contents = [item['sub_content'] for item in sub_chapters]

        embeddings = self._get_embeddings(contents)

        data_to_insert = [sub_titles, embeddings]
        self.collection.insert(data_to_insert)
        self.collection.flush()
        print(f"Inserted {len(sub_chapters)} vectors into Milvus.")

    def _get_embeddings(self, texts):
        inputs = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state[:, 0, :].cpu().numpy()

def main():
    # Load data from the provided JSON file
    json_data = {"title":"第一章 总 则",
      "main_idea":"该文段主要规定了社会保险费的征缴范围、征收管理及相关部门职责，旨在保障社会保险金的发放并规范征缴工作。",
      "sub_chapters":[
        {"sub_content":"第一条 为了加强和规范社会保险费征缴工作，保障社会保险金的发放，制定本条例。","sub_title":"第一条","main_idea":"为了加强和规范社会保险费征缴工作，保障社会保险金的发放，制定了本条例。","entities":["社会保险费","社会保险金","条例"],"keywords":["社会保险费","征缴工作","社会保险金","条例"]},
        {"sub_content":"第二条 基本养老保险费、基本医疗保险费、失业保险费（以下统称社会保险费）的征收、缴纳，适用本条例。\n本条例所称缴费单位、缴费个人，是指依照有关法律、行政法规和国务院的规定，应当缴纳社会保险费的单位和个人。","sub_title":"第二条","main_idea":"本条例规定了社会保险费的征收和缴纳适用本条例，并明确了缴费单位和个人的定义。","entities":["基本养老保险费","基本医疗保险费","失业保险费","社会保险费","条例"],"keywords":["基本养老保险费","基本医疗保险费","失业保险费","社会保险费","征收","缴纳","条例","缴费单位","缴费个人"]}
      ]
    }

    # Milvus connection details
    milvus_host = os.environ.get("MILVUS_HOST", "localhost")
    milvus_port = os.environ.get("MILVUS_PORT", "19530")

    manager = MilvusManager(host=milvus_host, port=milvus_port)

    collection_name = "law_embeddings"
    manager.create_collection(collection_name)
    manager.insert_data(json_data['sub_chapters'])

    connections.disconnect("default")
    print("Data vectorized and uploaded to Milvus successfully!")

if __name__ == "__main__":
    main()
