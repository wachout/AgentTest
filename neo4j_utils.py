from py2neo import Graph, Node, Relationship

class Neo4jCRUD:
    def __init__(self, uri, user, password):
        self.graph = Graph(uri, auth=(user, password))

    def create_node(self, label, properties):
        node = Node(label, **properties)
        self.graph.create(node)
        return node

    def create_relationship(self, node1, rel_type, node2, properties=None):
        if properties is None:
            properties = {}
        relationship = Relationship(node1, rel_type, node2, **properties)
        self.graph.create(relationship)
        return relationship

    def find_node(self, label, property_key, property_value):
        return self.graph.nodes.match(label, **{property_key: property_value}).first()

    def find_nodes(self, label):
        return list(self.graph.nodes.match(label))

    def update_node(self, node, new_properties):
        for key, value in new_properties.items():
            node[key] = value
        self.graph.push(node)
        return node

    def delete_node(self, node):
        self.graph.delete(node)

    def delete_all(self):
        self.graph.delete_all()
