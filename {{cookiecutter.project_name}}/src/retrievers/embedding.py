import logging
from typing import Any
import torch
import nltk

from src.retrievers.retriever import Retriever
from src.retrievers.vectorizers.vectorizer import Vectorizer

nltk.download('punkt')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Embedding(Retriever):
    def __init__(
            self,
            name: str = 'embedding',
            top_k: int = 5,
            vectorizer_document: Vectorizer = None,
            vectorizer_query: Vectorizer = None,
            sliding_window: bool = False,
            sliding_window_pooling: str = 'max',
            sliding_window_size: int = None,
            sliding_window_stride: int = None,
            sliding_window_type: str = None,
            query_split_size: int = 1,
            dtype: torch.dtype = torch.float32,
            device: str = 'cpu',
            save_if_missing: bool = False,
            knowledge_base: Any = None,
            **kwargs: Any
    ):
        super().__init__(name, top_k, knowledge_base=knowledge_base)
        self.vectorizer_document = vectorizer_document
        self.vectorizer_query = vectorizer_query
        self.sliding_window = sliding_window
        self.sliding_window_pooling = sliding_window_pooling
        self.sliding_window_size = sliding_window_size
        self.sliding_window_stride = sliding_window_stride
        self.sliding_window_type = sliding_window_type
        self.query_split_size = query_split_size
        self.dtype = dtype
        self.device = device
        self.save_if_missing = save_if_missing
        self.document_embeddings = None

    def calculate_embeddings(self):
        document_embeddings = self.vectorizer_document.vectorize(
            self.knowledge_base.get_documents_texts(),
            save_if_missing=self.save_if_missing,
            normalize=True
        )

        document_embeddings = document_embeddings.transpose(
            0, 1)

        self.document_embeddings = document_embeddings.to(
            device=self.device, dtype=self.dtype)

    def set_knowledge_base(self, knowledge_base: Any):
        self.knowledge_base = knowledge_base
        self.calculate_embeddings()
        
    def retrieve(self, query: str) -> Any:
        if self.document_embeddings is None:
            self.calculate_embeddings()
        
        query_embeddings = self.vectorizer_query.vectorize(
            [query],
            save_if_missing=self.save_if_missing,
            normalize=True
        )
        delimiters = [(0, 1)]
        results = []

        for start_id, end_id in delimiters:
            sims = torch.mm(
                query_embeddings[start_id:end_id].to(
                    device=self.device, dtype=self.dtype),
                self.document_embeddings
            )

            sorted_ids = torch.argsort(sims, descending=True, dim=1)
            if self.top_k is None:
                top_k = sorted_ids[0, :].tolist()
            else:
                top_k = sorted_ids[0, :self.top_k].tolist()
            top_k = self.knowledge_base.map_topK(top_k)

            results.append([
                self.knowledge_base.get_document(int(i))
                for i in top_k
            ])

        results = [item for sublist in results for item in sublist]
        return results, top_k, []
