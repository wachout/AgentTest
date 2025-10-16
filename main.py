import time
import concurrent.futures
from rag_components.state import State
from rag_components.query_enhancer import QueryEnhancer
from rag_components.topic_validator import TopicValidator
from rag_components.content_retriever import ContentRetriever
from rag_components.relevance_assessor import RelevanceAssessor
from rag_components.response_generator import ResponseGenerator
from rag_components.query_optimizer import QueryOptimizer
from rag_components.thread_monitor import thread_monitor

class Agent:
    def __init__(self, provider="deepseek"):
        self.state = State()
        self.query_enhancer = QueryEnhancer(provider)
        self.topic_validator = TopicValidator(provider)
        self.content_retriever = ContentRetriever()
        self.relevance_assessor = RelevanceAssessor(provider)
        self.response_generator = ResponseGenerator(provider)
        self.query_optimizer = QueryOptimizer(provider)

    def run(self, user_query: str):
        self.state.add_message("user", user_query)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            # 1. Enhance query
            thread_monitor.start("query_enhancer")
            enhanced_query = self.query_enhancer.enhance_query(self.state)
            thread_monitor.stop("query_enhancer")
            print(f"Enhanced Query: {enhanced_query}")

            # 2. Validate topic
            thread_monitor.start("topic_validator")
            topic_validation = self.topic_validator.validate_topic(enhanced_query)
            thread_monitor.stop("topic_validator")
            print(f"Topic Validation: {topic_validation}")
            if topic_validation["classification"] == "out-of-domain":
                return "I can only answer questions about technical support for software."

            # 3. Retrieve content
            thread_monitor.start("content_retriever")
            retrieved_docs = self.content_retriever.retrieve(enhanced_query)
            thread_monitor.stop("content_retriever")
            self.state.retrieved_docs = retrieved_docs
            print(f"Retrieved {len(retrieved_docs)} documents.")

            # 4. Assess relevance
            if retrieved_docs:
                thread_monitor.start("relevance_assessor")
                relevance = self.relevance_assessor.assess_relevance(enhanced_query, retrieved_docs)
                thread_monitor.stop("relevance_assessor")
                print(f"Relevance Assessment: {relevance}")
                self.state.document_relevance = relevance["is_relevant"]
            else:
                self.state.document_relevance = False

            # 5. Generate response or optimize
            if self.state.document_relevance:
                thread_monitor.start("response_generator")
                response = self.response_generator.generate_response(enhanced_query, self.state.retrieved_docs)
                thread_monitor.stop("response_generator")
                self.state.add_message("assistant", response)
                return response
            else:
                # 6. Optimize query and retry
                if self.state.refinement_attempts < 2: # Loop protection
                    self.state.refinement_attempts += 1
                    print("Optimizing query and retrying...")
                    thread_monitor.start("query_optimizer")
                    optimized_query = self.query_optimizer.optimize_query(enhanced_query)
                    thread_monitor.stop("query_optimizer")
                    return self.run(optimized_query) # Recursive call with optimized query
                else:
                    return "I could not find a relevant answer to your query."

if __name__ == "__main__":
    try:
        agent = Agent(provider="deepseek")
        response = agent.run("how do I reset my password?")
        print(f"\nFinal Response:\n{response}")

        # Follow-up question
        response = agent.run("what about billing?")
        print(f"\nFinal Response:\n{response}")

        # Print thread monitoring report
        print("\n--- Thread Monitoring Report ---")
        report = thread_monitor.get_report()
        for thread_name, data in report.items():
            print(f"Thread '{thread_name}':")
            print(f"  Start Time: {data['start_time']:.4f}")
            print(f"  End Time: {data['end_time']:.4f}")
            print(f"  Duration: {data['duration']:.4f}s")
        print("---------------------------------")

    except Exception as e:
        print(f"An error occurred: {e}")
        print("Please ensure you have set your API keys in a .env file.")