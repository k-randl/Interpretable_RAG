import torch
import numpy as np
from resources.retrieval_offline import ExplainableAutoModelForRetrieval as ExplainableAutoModelForOfflineRetrieval
from resources.retrieval_online  import ExplainableAutoModelForRetrieval as ExplainableAutoModelForOnlineRetrieval
from resources.generation import ExplainableAutoModelForGeneration

from typing import Optional, Dict, List, Any
from numpy.typing import NDArray

class ExplainableAutoModelForRAG:
    def __init__(self,
        generator_name_or_path:str,
        query_encoder_name_or_path:str,
        context_encoder_name_or_path:Optional[str]=None,
        *,
        offline:bool=False,
        retriever_query_format:str='{query}',
        retriever_kwargs:Dict[str, Any]={},
        generator_kwargs:Dict[str, Any]={}
    ):
        # save general info:
        self.__is_offline = offline
        self.__retriever_query_format = retriever_query_format
        
        # instantiate generator:
        self.generator = ExplainableAutoModelForGeneration.from_pretrained(
            pretrained_model_name_or_path=generator_name_or_path,
            **generator_kwargs
        )

        # instantiate  correct retriever:
        if offline:
            if context_encoder_name_or_path is not None:
                print('WARNING: Parameter `context_encoder_name_or_path` is ignored if `offline == True`.')

            self.retriever = ExplainableAutoModelForOfflineRetrieval.from_pretrained(
                query_encoder_name_or_path=query_encoder_name_or_path,
                **retriever_kwargs
            ).to('cuda' if torch.cuda.is_available() else 'cpu')

        else:
            self.retriever = ExplainableAutoModelForOnlineRetrieval.from_pretrained(
                query_encoder_name_or_path=query_encoder_name_or_path,
                context_encoder_name_or_path=context_encoder_name_or_path,
                **retriever_kwargs
            ).to('cuda' if torch.cuda.is_available() else 'cpu')

    def __call__(self, query:str, *,
        k:Optional[int]=None,
        contexts:Optional[List[str]]=None,
        dir:Optional[str]=None,
        index:Optional[torch.FloatTensor]=None, 
        generator_kwargs:Optional[Dict[str, Any]]=None,
        retriever_kwargs:Optional[Dict[str, Any]]=None
    ):
        # init kwargs if unspecified:
        if generator_kwargs is None: generator_kwargs = {}
        if retriever_kwargs is None: retriever_kwargs = {}

        # check parameters:
        if self.__is_offline:
            if index is not None: retriever_kwargs['index'] = index
            elif dir is not None: retriever_kwargs['dir'] = dir
            else: raise ValueError('Either `index` or `dir` must be specified in offline retrieval mode.')

            if k is None: raise ValueError('`k` must be specified in offline retrieval mode.')

        else:
            if contexts is not None: retriever_kwargs['contexts'] = contexts
            else: raise ValueError('`contexts` must be specified in online retrieval mode.')

            if k is None: k = len(contexts)

        # calculate similarity:
        self.retrieved_ids, self.retrieved_sim = self.retriever(
            self.__retriever_query_format.format(query=query),
            k=k,
            reorder=True,
            output_attentions=True,
            output_hidden_states=True,
            **retriever_kwargs
        )

        # get documents:
        self.retrieved_docs = [contexts[i] for i in self.retrieved_ids]

        # generate response:
        chat = self.generator.explain_generate(query, self.retrieved_docs, **generator_kwargs)

        # generate response:
        return chat
    
    @property
    def retriever_document_importance(self) -> NDArray[np.float_]:
        '''Normalized document importance estimated by the retriever.'''

        if not hasattr(self, 'retrieved_sim'):
            raise AttributeError('`__call__(...)` needs to be called at least once before accessing `document_importance`!')

        doc_importance_retriever = self.retrieved_sim.numpy()
        doc_importance_retriever /= np.abs(doc_importance_retriever).sum()

        return doc_importance_retriever
    
    @property
    def generator_document_importance(self) -> NDArray[np.float_]:
        '''Normalized document importance of the generator.'''
        #doc_importance_generator = self.generator.get_shapley_values('token').sum(axis=1)
        doc_importance_generator = self.generator.get_shapley_values('context', 'sequence')[:,0]
        doc_importance_generator /= np.abs(doc_importance_generator).sum()

        return doc_importance_generator
    
    @property
    def mean_document_importance(self) -> NDArray[np.float_]:
        '''Mean normalized document importance of the rag pipeline.'''
        return (self.retriever_document_importance + self.generator_document_importance) / 2.