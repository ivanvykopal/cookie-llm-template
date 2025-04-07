import logging
import faiss
import numpy as np
from typing import Any, List

from src.retrievers.retriever import Retriever
from src.retrievers.vectorizers.vectorizer import Vectorizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Faiss(Retriever):
    def __init__(
        self, 
        name: str = 'faiss', 
        top_k: int = 5,
        vectorizer_document: Vectorizer = None,
        vectorizer_query: Vectorizer = None,
        device: str = 'cpu',
        save_if_missing: bool = False,
        knowledge_base: Any = None, 
        **kwargs: Any
    ):    
        super().__init__(name, top_k, knowledge_base=knowledge_base)
        self.vectorizer_document = vectorizer_document
        self.vectorizer_query = vectorizer_query
        self.device = device
        self.save_if_missing = save_if_missing
        self.document_embeddings = None
        self.index = None
        self.create_index()

    def create_index(self):        
        self.calculate_embeddings()
        vectors = self.document_embeddings.cpu().numpy()
        ids = np.array(self.dataset.get_document_ids())
        dim = vectors.shape[1]

        index_flat = faiss.IndexIDMap2(faiss.IndexFlatIP(dim))

        if self.device == 'cuda':
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, index_flat)
        else:
            self.index = index_flat
        
        self.index.add_with_ids(vectors, ids)
    
    def calculate_embeddings(self):
        self.document_embeddings = self.vectorizer_document.vectorize(
            self.dataset.get_documents_texts(),
            save_if_missing=self.save_if_missing,
            normalize=True
        )

    def set_knowledge_base(self, knowledge_base: Any):
        self.knowledge_base = knowledge_base
        self.calculate_embeddings()

    def retrieve(self, query: List) -> Any:
        query_embeddings = self.vectorizer_query.vectorize(
            query,
            save_if_missing=self.save_if_missing,
            normalize=True
        )
        query_vectors = query_embeddings.cpu().numpy()

        sims, retrieved_ids = self.index.search(query_vectors, self.top_k)
        return [], retrieved_ids, sims
