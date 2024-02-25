import torch
from llama_index.llms.huggingface import HuggingFaceLLM
from transformers import BitsAndBytesConfig
from llama_index.core import PromptTemplate
from langchain.embeddings.huggingface import HuggingFaceBgeEmbeddings
from llama_index.embeddings.langchain import LangchainEmbedding
from llama_index.core import Settings
from sqlalchemy import make_url
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.core import VectorStoreIndex
from llama_index.core.query_engine.retriever_query_engine import RetrieverQueryEngine
from llama_index.core.indices.vector_store.retrievers.retriever import VectorIndexRetriever
from llama_index.core import get_response_synthesizer


class AIChatBot:
    def __init__(self, args):
        print("Initializing AI Chatbot")
        # Initialize model configuration and database connection here
        self.model_name = args.model_name
        self.embedding_model_name = args.embedding_model_name
        self.database_connection_string = args.database_connection_string
        self.database_name = args.database_name
        self.table_name = args.table_name
        self.system_prompt = args.system_prompt
        self.context_window = args.context_window
        self.max_new_tokens = args.max_new_tokens
        self.device = args.device
        self.chunk_size = args.chunk_size
        self.chunk_overlap = args.chunk_overlap
        self.top_k_index_to_return = args.top_k_index_to_return

    def setup_model(self):
        print("Setting up model")
        # This will wrap the default prompts that are internal to llama-index
        query_wrapper_prompt = PromptTemplate("<|USER|>{query_str}<|ASSISTANT|>")

        # Create the LLM using the HuggingFaceLLM class
        llm = HuggingFaceLLM(
            context_window=self.context_window,
            max_new_tokens=self.max_new_tokens,
            system_prompt=self.system_prompt,
            query_wrapper_prompt=query_wrapper_prompt,
            tokenizer_name=self.model_name,
            model_name=self.model_name,
            device_map=self.device,
            generate_kwargs={"temperature": 0.2, "top_k": 5, "top_p": 0.95, "do_sample": True},
            # uncomment this if using CUDA to reduce memory usage
            # model_kwargs={
            #     # "torch_dtype": torch.float16
            #     'quantization_config':quantization_config
            # }
        )

        # Create the embedding model using the HuggingFaceBgeEmbeddings class
        embed_model = LangchainEmbedding(
        HuggingFaceBgeEmbeddings(model_name=self.embedding_model_name)
        )

        # Get the embedding dimension of the model by doing a forward pass with a dummy input
        embed_dim = len(embed_model.get_text_embedding("Hello world")) # 1024

        self.embed_dim = embed_dim
        self.llm = llm
        self.embed_model = embed_model

    def apply_settings(self):
        print("Applying settings")
        Settings.llm = self.llm
        Settings.embed_model = self.embed_model


        Settings.chunk_size = self.chunk_size
        Settings.chunk_overlap = self.chunk_overlap

    def get_index_from_database(self):
        print("Getting index from database")
        # Creates a URL object from the connection string
        url = make_url(self.database_connection_string)

        # Create the vector store
        vector_store = PGVectorStore.from_params(
            database=self.database_name,
            host=url.host,
            password=url.password,
            port=url.port,
            user=url.username,
            table_name=self.table_name,
            embed_dim=self.embed_dim,
        )

        # Load the index from the vector store of the database
        index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
        self.index = index

    def setup_engine(self):
        print("Setting up engine")
        # Create the retriever that manages the index and the number of results to return
        retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=self.top_k_index_to_return,
        )

        # Create the response synthesizer that will be used to synthesize the response
        response_synthesizer = get_response_synthesizer(
            response_mode='simple_summarize',
        )

        # Create the query engine that will be used to query the retriever and synthesize the response
        engine = RetrieverQueryEngine(
            retriever=retriever,
            response_synthesizer=response_synthesizer,
        )
        self.engine = engine

    def build_bot(self):
        self.setup_model()
        self.apply_settings()
        self.get_index_from_database()
        self.setup_engine()
        print("Bot built successfully")


    def process_query(self, query_text):
        response = self.engine.query(query_text)
        return response