import os
import json
import torch
import numpy as np
from scipy.stats import spearmanr
from .retrieval import Methods_t
from .retrieval_offline import ExplainableAutoModelForRetrieval as ExplainableAutoModelForOfflineRetrieval
from .retrieval_online  import ExplainableAutoModelForRetrieval as ExplainableAutoModelForOnlineRetrieval
from .generation import ExplainableAutoModelForGeneration, Aggregations_t
from .utils import tokens2words

from typing import Optional, Union, Callable, Literal, Dict, List, Tuple, Any
from numpy.typing import NDArray

class ExplainableAutoModelForRAG:
    def __init__(self,
            generator_name_or_path:str,
            query_encoder_name_or_path:str,
            context_encoder_name_or_path:Optional[str]=None,
            *,
            offline:bool=False,
            dir:Optional[str]=None,
            index:Union[Callable[[str, int],List[str]],str,torch.FloatTensor,None]=None,
            retriever_query_format:str='{query}',
            retriever_token_processor:Optional[Callable[[str],str]]=None,
            retriever_kwargs:Dict[str, Any]={},
            generator_token_processor:Optional[Callable[[str],str]]=None,
            generator_kwargs:Dict[str, Any]={}
        ) -> None:
        '''Initialize the Interpretable RAG (Retrieval-Augmented Generation) model.
        This constructor sets up the generator and retriever components for the RAG system.
        It supports both online and offline retrieval modes, loading pre-trained models
        and configuring them with the provided parameters.

        Args:
            generator_name_or_path (str):       Path or name of the pre-trained generator model
                                                (e.g., a language model for generation).
            query_encoder_name_or_path (str):   Path or name of the pre-trained query encoder model
                                                used in the retriever.
            context_encoder_name_or_path (str): Path or name of the pre-trained context encoder model.
                                                Ignored in offline mode. Defaults to None.
            offline (bool, optional):           If True, use offline retrieval mode (requires `index` or `dir`).
                                                Defaults to False.
            dir (str, optional):                Directory path for offline retrieval data. Ignored in online mode.
                                                Defaults to None.
            index ((str, int) -> List[str] | str | FloatTensor, optional): Pre-built index for retrieval.
                                                Can be a callable, file path, or tensor. Defaults to None.
            retriever_query_format (str, optional): Format string for processing retriever queries (e.g., `'{query}'`).
                                                Defaults to `'{query}'`.
            retriever_token_processor ((str) -> str, optional): Function to process tokens for the retriever.
                                                Defaults to None.
            retriever_kwargs (Dict[str, Any], optional): Additional keyword arguments for the retriever model.
            generator_token_processor ((str) -> str, optional): Function to process tokens for the generator.
                                                Defaults to None.
            generator_kwargs (Dict[str, Any], optional): Additional keyword arguments for the generator model.

        Raises:
            ValueError: If in offline mode, neither `index` nor `dir` is provided.

        Notes:
            - In offline mode, `context_encoder_name_or_path` is ignored.
            - In online mode, `dir` is ignored.
            - Models are automatically moved to GPU if available, otherwise CPU.
        '''
        # save general info:
        self.__is_offline = offline
        self.__retriever_query_format = retriever_query_format
        self.__retriever_token_processor = retriever_token_processor
        self.__generator_token_processor = generator_token_processor
        
        # instantiate generator:
        self.generator = ExplainableAutoModelForGeneration.from_pretrained(
            pretrained_model_name_or_path=generator_name_or_path,
            **generator_kwargs
        )

        # instantiate correct retriever:
        if offline:
            if index is not None: retriever_kwargs['index'] = index
            elif dir is not None: retriever_kwargs['dir'] = dir
            else: raise ValueError('Either `index` or `dir` must be specified in offline retrieval mode.')

            if context_encoder_name_or_path is not None:
                print('WARNING: Parameter `context_encoder_name_or_path` is ignored if `offline == True`.')

            self.retriever = ExplainableAutoModelForOfflineRetrieval.from_pretrained(
                query_encoder_name_or_path=query_encoder_name_or_path,
                **retriever_kwargs
            ).to('cuda' if torch.cuda.is_available() else 'cpu')

        else:
            if index is not None: retriever_kwargs['index'] = index

            if dir is not None:
                print('WARNING: Parameter `dir` is ignored if `offline == False`.')

            self.retriever = ExplainableAutoModelForOnlineRetrieval.from_pretrained(
                query_encoder_name_or_path=query_encoder_name_or_path,
                context_encoder_name_or_path=context_encoder_name_or_path,
                **retriever_kwargs
            ).to('cuda' if torch.cuda.is_available() else 'cpu')

    def __call__(self, query:str, k:Optional[int]=None, *,
            contexts:Optional[List[str]]=None,
            fast_retrieval:bool=True,
            generator_kwargs:Dict[str, Any] = {},
            retriever_kwargs:Dict[str, Any] = {}
        ) -> List[Dict[Literal['role','content'],str]]:
        '''Callable method to perform retrieval-augmented generation (RAG) for a given query. This method
        retrieves relevant documents based on the query, computes similarities, and generates an explained
        response using the retrieved documents.

        Args:
            query (str):                     The input query string for which to retrieve and generate
                                             a response.
            k (int, optional):               The number of documents to retrieve. If not specified, it
                                             is inferred from the mode (offline/online) or the length
                                             of provided contexts. In offline mode, `k` must be specified.
                                             In online mode, either `k` or `contexts` must be provided.
            contexts (List[str], optional):  A list of context strings to use for retrieval.
                                             If provided, overrides any contexts in retriever_kwargs.
                                             In online mode, either `k` or `contexts` must be provided. 
            fast_retrieval (bool, optional): If True, skips computation of gradients, attentions,
                                             and hidden states for faster retrieval. Defaults to True.
            generator_kwargs (Dict[str, Any], optional): Additional keyword arguments to pass to the
                                             generator's explain_generate method.
            retriever_kwargs (Dict[str, Any], optional): Additional keyword arguments to pass to the
                                             retriever.

        Returns:
            The generated response from the generator, typically an explained output based on the query
            and retrieved documents.

        Raises:
            ValueError: If `k` is None in offline mode, or if neither `contexts` nor `k` is specified in
            online mode.

        '''
        # check parameters:
        contexts = retriever_kwargs.pop('contexts', contexts)
        k        = retriever_kwargs.pop('k', k)

        if k is None:
            if self.__is_offline:
                raise ValueError('`k` must be specified in offline retrieval mode.')

            else:
                if contexts is None:
                    raise ValueError('`contexts` or `k` must be specified in online retrieval mode.')

                k = len(contexts)

        # compute similarity:
        self.retrieved_docs, self.retrieved_sim = self.retriever(
            self.__retriever_query_format.format(query=query),
            contexts=contexts,
            k=k,
            reorder=True,
            output_texts=True,
            compute_grad=not fast_retrieval,
            output_attentions=not fast_retrieval,
            output_hidden_states=not fast_retrieval,
            **retriever_kwargs
        )

        # generate response:
        return self.generator.explain_generate(query, self.retrieved_docs, **generator_kwargs)

    @property
    def retriever_query_format(self) -> str:
        '''Format string applied to queries.'''
        return self.__retriever_query_format

    @property
    def retriever_document_importance(self) -> NDArray[np.float64]:
        '''Normalized document importance estimated by the retriever.'''

        if not hasattr(self, 'retrieved_sim'):
            raise AttributeError('`__call__(...)` needs to be called at least once before accessing `document_importance`!')

        doc_importance_retriever = self.retrieved_sim.numpy()
        doc_importance_retriever /= np.abs(doc_importance_retriever).sum()

        return doc_importance_retriever

    @property
    def generator_document_importance(self) -> NDArray[np.float64]:
        '''Normalized document importance of the generator.'''
        doc_importance_generator = self.generator.get_shapley_values('context', 'token').sum(axis=1)
        #doc_importance_generator = self.generator.get_shapley_values('context', 'sequence')
        doc_importance_generator /= np.abs(doc_importance_generator).sum()

        return doc_importance_generator

    @property
    def mean_document_importance(self) -> NDArray[np.float64]:
        '''Mean normalized document importance of the rag pipeline.'''
        return (self.retriever_document_importance + self.generator_document_importance) / 2.

    @property
    def document_agreement(self) -> NDArray[np.float64]:
        '''Document importance disagreement between retriever and generator.'''

        # get normalized importance rankings:
        ret = np.argsort(self.retriever_document_importance)
        gen = np.argsort(self.generator_document_importance)

        # compute spearman correlation:
        r, p = spearmanr(ret, gen)

        return r

    @property
    def retriever_query_importance(self) -> NDArray[np.float64]:
        '''Normalized word importance of the query for retrieving.'''

        # get special tokens:
        special_tokens = set(self.retriever.tokenizer.special_tokens_map.values())

        # get token to word mapping:
        indices = tokens2words(
            self.retriever.in_tokens['query'][0],
            token_processor=self.__retriever_token_processor,
            filter_tokens=special_tokens
        )

        # delete formating tokens:
        if self.__retriever_query_format is not None:
            pattern = self.__retriever_query_format.format(query='[§§§]').split()

            prefix_size = pattern.index('[§§§]')
            if prefix_size > 0: indices = indices[prefix_size:]

            suffix_size = len(pattern) - prefix_size - 1
            if suffix_size > 0: indices = indices[:-suffix_size]

        # get token importance:
        qry_importance_retriever = np.abs(self.retriever.IntGrad()['query'].numpy()[0])

        # aggregate to word importance:
        qry_importance_retriever = [np.mean([qry_importance_retriever[i] for i in w]) for w in indices]

        # normalize:
        qry_importance_retriever /= np.abs(qry_importance_retriever).sum()

        return qry_importance_retriever

    @property
    def generator_query_importance(self) -> NDArray[np.float64]:
        '''Normalized word importance of the query during generation.'''
        qry_importance_generator = self.generator.get_shapley_values('query', 'token').sum(axis=1)
        #qry_importance_generator = self.generator.get_shapley_values('query', 'sequence')
        qry_importance_generator /= np.abs(qry_importance_generator).sum()

        return qry_importance_generator

    @property
    def mean_query_importance(self) -> NDArray[np.float64]:
        '''Mean word importance of the query for the rag pipeline.'''
        return (self.retriever_query_importance + self.generator_query_importance) / 2.

    @property
    def query_agreement(self) -> NDArray[np.float64]:
        '''Word importance agreement of the query between retriever and generator.'''

        # get normalized importance rankings:
        ret = np.argsort(self.retriever_query_importance)
        gen = np.argsort(self.generator_query_importance)

        # compute spearman correlation:
        r, p = spearmanr(ret, gen)

        return r
    
    def save_values(self, path:str, file_name:str, *,
            ret_methods:Optional[List[Methods_t]]=None,
            gen_aggregations:Optional[List[Aggregations_t]]=None,
            filter_special_tokens:bool=True,
            num_steps:int=100,
            batch_size:int=64
        ) -> Union[str, None]:
        """Saves the explanation data to a file.

        Args:
            path (str):                             The path where the values should be saved.
            ret_methods (List[str]):                List of retriever explanations to save.
            gen_aggregations (List[str]):           List of generation aggregations to save.
                                                    If unspecified, saves all aggregations.
            filter_special_tokens (bool, optional): If `True` (default), set the importance of special tokens to 0.
            num_steps (int, optional):              Number of approximation steps for the Rieman approximation
                                                    of the integral in `intGrad` (default 100).
            batch_size (int, optional):             Batch size used for calculating the gradients in `intGrad` (default 64).

        Returns:
            If `path` is not specified, returns the saved data instead.
        """
        ret_dir = os.path.join(path, 'retrieval')
        os.makedirs(ret_dir, exist_ok=True)
        self.retriever.save_values(os.path.join(ret_dir, file_name), methods=ret_methods, filter_special_tokens=filter_special_tokens, num_steps=num_steps, batch_size=batch_size)

        gen_dir = os.path.join(path, 'generation')
        os.makedirs(gen_dir, exist_ok=True)
        self.generator.save_values(os.path.join(gen_dir, file_name), aggregations=gen_aggregations)


class WARGScorer(ExplainableAutoModelForRAG):
    def __init__(self, index_path:str, generator_name_or_path:str, query_encoder_name_or_path:str, context_encoder_name_or_path:Optional[str]=None, *,
            retriever_kwargs:Dict[str, Any]={},
            generator_kwargs:Dict[str, Any]={},
            **kwargs
        ) -> None:
        '''Initialize a WARG (Weighted Attribution-Relevance Gap) scorer.

        This class specializes ExplainableAutoModelForRAG for computing WARG scores,
        which quantify the disagreement between retriever-based and generator-based
        document importance rankings.

        Args:
            index_path (str):
                Path to the retrieval index used by the retriever.
            generator_name_or_path (str):
                Name or path of the generator model.
            query_encoder_name_or_path (str):
                Name or path of the query encoder model.
            context_encoder_name_or_path (str, optional):
                Name or path of the context encoder model.
            retriever_kwargs (Dict[str, Any], optional):
                Additional keyword arguments passed to the retriever.
            generator_kwargs (Dict[str, Any], optional):
                Additional keyword arguments passed to the generator.
            **kwargs:
                Additional keyword arguments forwarded to the parent RAG initializer.

        Notes:
            - The scorer is fixed to online retrieval mode and uses a pre-built index.
            - Retrieved query results are cached internally to avoid recomputation.
        '''

        # fix parameters to online retrieval:
        kwargs['offline'] = False
        kwargs['dir'] = None

        # init parent class:
        super().__init__(generator_name_or_path, query_encoder_name_or_path, context_encoder_name_or_path, index=index_path,
                         retriever_kwargs=retriever_kwargs, generator_kwargs=generator_kwargs, **kwargs)
        
        # init query buffer:
        self._index_path = index_path
        self._queries = {}

    def __generate(self, query:str, k:int, n:int, *,
            generator_kwargs:Dict[str, Any] = {},
            retriever_kwargs:Dict[str, Any] = {}
        ) -> Tuple[NDArray[np.float64], NDArray[np.float64], List[str], List[str]]:
        '''Run retrieval-augmented generation for a single query and cache results.

        This method retrieves `2k` documents, computes retriever and generator
        document importances, and stores the results in the internal query cache.

        Args:
            query (str):
                Input query string.
            k (int):
                Number of top documents used for scoring.
            generator_kwargs (Dict[str, Any], optional):
                Keyword arguments passed to the generator.
            retriever_kwargs (Dict[str, Any], optional):
                Keyword arguments passed to the retriever.

        Returns:
            Tuple containing:
                - retriever document importance scores
                - generator document importance scores
                - top-k retrieved documents
                - remaining retrieved documents
        '''

        # Run generation
        super().__call__(query, n, fast_retrieval=True,
                        generator_kwargs=generator_kwargs,
                        retriever_kwargs=retriever_kwargs)
        
        # Get importance scores
        document_importance = (
            self.retriever_document_importance,
            self.generator_document_importance,
            self.retrieved_docs[:k],
            self.retrieved_docs[k:]
        )

        self._queries[query] = document_importance
        return document_importance

    def __call__(self, queries:Union[List[str], str], k:int, p:float=0.9, *,
            batch_size:int=64,
            max_gen_len:int=200,
            checkpoint_path:Optional[str]=None,
            generator_kwargs:Dict[str, Any]={},
            retriever_kwargs:Dict[str, Any]={}
        ) -> NDArray[np.float64]:
        '''Compute WARG scores for one or more queries.

        For each query, this method compares retriever and generator document
        rankings using Rank-Biased Overlap (RBO) and returns a disagreement score.

        Args:
            queries (Union[List[str], str]):
                One or more query strings.
            k (int):
                Cutoff for top-k document ranking comparison.
            p (float, optional):
                Persistence parameter for RBO. Defaults to 0.9.
            batch_size (int, optional):
                Batch size used in generation and retrieval. Defaults to 64.
            max_gen_len (int, optional):
                Maximum number of generated tokens. Defaults to 200.
            checkpoint_path (str, optional):
                Path to a JSON checkpoint file for resuming computation.
            generator_kwargs (Dict[str, Any], optional):
                Generator-specific keyword arguments.
            retriever_kwargs (Dict[str, Any], optional):
                Retriever-specific keyword arguments.

        Returns:
            numpy.ndarray:
                Array of WARG disagreement scores (one per query).

        Notes:
            - Results are cached per query.
            - Higher WARG values indicate greater disagreement between retriever
              and generator importance rankings.
        '''

        # Recover from checkpoint:
        if checkpoint_path is not None:
            if os.path.exists(checkpoint_path):
                with open(checkpoint_path, 'r') as fp:
                    load_qrys = json.load(fp)['queries']

                for _q, _r, _g, _p, _n in load_qrys:
                    self._queries[_q] = (np.array(_r), np.array(_g), _p, _n)

        # Convert queries to list:
        if isinstance(queries, str):
            queries = [queries]

        # Get total retrieved samples:
        n = int(np.ceil(k*1.5))

        # Init generation arguments:
        generator_kwargs['do_sample']           = generator_kwargs.get('do_sample', False)
        generator_kwargs['conditional']         = generator_kwargs.get('conditional', True)
        generator_kwargs['max_samples_query']   = generator_kwargs.get('max_samples_query', 0)
        generator_kwargs['max_samples_context'] = generator_kwargs.get('max_samples_context', batch_size)
        generator_kwargs['max_new_tokens']      = max_gen_len
        generator_kwargs['batch_size']          = batch_size
        retriever_kwargs['batch_size']          = batch_size
        gen_args = {
            'k':k, 'n':n,
            'generator_kwargs':generator_kwargs,
            'retriever_kwargs':retriever_kwargs
        }

        scores = []
        for i, query in enumerate(queries):
            print(f'Scoring query {i+1:d} of {len(queries):d} ...')

            # Get rankings:
            try:
                ret_scores, gen_scores, ret_docs, cnt_docs = self._queries[query]

            except KeyError:
                ret_scores, gen_scores, ret_docs, cnt_docs = self.__generate(query, **gen_args)
            
            if n > min(len(ret_scores), len(gen_scores)):
                ret_scores, gen_scores, ret_docs, cnt_docs = self.__generate(query, **gen_args)

            # Get indices:
            ret_rank = np.argsort(ret_scores[:n])
            gen_rank = np.argsort(gen_scores[:n])

            # Compute rbo:
            rbo = 0.0
            for i in range(2*k):
                d = i + 1
                a = set(ret_rank[:d])
                b = set(gen_rank[:d])
                overlap = len(a.intersection(b))
                
                term = (1 - p) * (p ** i) * (overlap / d)
                rbo += term

            # Calculate gap:
            scores.append(1.0 - rbo)
            print(f'WARG = {scores[-1]:.2f}')

            # Save scorer checkpoint:
            if checkpoint_path is not None:
                self.dump(checkpoint_path)

            print('Done.\n')

        return np.array(scores)
    
    def dump(self, path:str) -> None:
        '''Save the scorer state and cached query results to disk.

        The dump includes model identifiers, index path, retriever configuration,
        and cached query importance scores.

        Args:
            path (str):
                Path to the output JSON file.
        '''

        # Initialize dictionary that will be serialized to disk
        dump_dict = {}

        # Store retriever metadata needed for reconstruction
        dump_dict['retriever']  = {'query_encoder': self.retriever.query_encoder_name_or_path,
                                   'query_format': self.retriever_query_format}

        # Store generator model identifier
        dump_dict['generator']  = self.retriever.query_encoder_name_or_path

        # Store index path used by the retriever
        dump_dict['index_path'] = self._index_path
        
        # Store cached query results:
        #   k : query string
        #   r : retriever importance scores
        #   g : generator importance scores
        #   p : top-k retrieved documents
        #   n : remaining retrieved documents
        #
        # NumPy arrays are converted to lists to ensure JSON serializability
        dump_dict['queries']    = [(k, r.tolist(), g.tolist(), p, n)
                                   for k, (r, g, p, n)
                                   in self._queries.items()]

        # If the retriever uses different encoders for queries and contexts,
        # store the context encoder separately so it can be restored correctly
        if self.retriever.query_encoder != self.retriever.context_encoder:
            dump_dict['retriever']['context_encoder'] = self.retriever.context_encoder_name_or_path

        # Write the full configuration and cached results to a JSON file
        with open(path, 'w') as fp:
            json.dump(dump_dict, fp, indent=2)

    @classmethod
    def load(cls, path:str, **kwargs) -> 'WARGScorer':
        '''Load a WARGScorer from a previously saved checkpoint.

        Args:
            path (str):
                Path to the checkpoint JSON file.
            **kwargs:
                Optional overrides for model paths or configuration.

        Returns:
            WARGScorer:
                A fully initialized scorer with restored cached queries.
        '''
        # Load checkpoint JSON from disk
        with open(path, 'r') as fp:
           load_dict = json.load(fp)

        # Recover model and index configuration from the checkpoint,
        # allowing keyword arguments to override saved values
        index_path                   = kwargs.pop('index_path', load_dict['index_path'])
        generator_name_or_path       = kwargs.pop('generator_name_or_path', load_dict['generator'])
        query_encoder_name_or_path   = kwargs.pop('query_encoder_name_or_path', load_dict['retriever']['query_encoder'])
        context_encoder_name_or_path = kwargs.pop('context_encoder_name_or_path', load_dict['retriever'].get('context_encoder', None))
        retriever_query_format       = kwargs.pop('query_format', load_dict['retriever']['query_format'])

        # Reconstruct the WARGScorer instance with the recovered configuration
        scorer =  cls(index_path                   = index_path,
                      generator_name_or_path       = generator_name_or_path,
                      query_encoder_name_or_path   = query_encoder_name_or_path,
                      context_encoder_name_or_path = context_encoder_name_or_path,
                      retriever_query_format       = retriever_query_format
                      **kwargs)

        # Restore cached query results:
        # Convert list representations back into NumPy arrays
        for k, r, g, p, n in load_dict['queries']:
            scorer._queries[k] = (np.array(r), np.array(g), p, n)

        # Return fully reconstructed scorer
        return scorer