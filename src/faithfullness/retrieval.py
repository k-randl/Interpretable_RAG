import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm.autonotebook import tqdm
from typing import Optional, Dict, List

from src.Interpretable_RAG.retrieval_online import ExplainableAutoModelForRetrieval

#=======================================================================#
# Retriever Explanation Faithfullness:                                  #
#=======================================================================#

class AIPCForRetrieval:
    def __init__(self, retriever:ExplainableAutoModelForRetrieval, *, query_format:str='{query}') -> None:
        '''Initialize the AIPCForRetrieval evaluator. This class computes
        area-inside-perturbation-curves (AIPC) for retrieval explanations by iteratively
        masking tokens in retrieved contexts according to relevancy scores and measuring
        the change in similarity. The object stores the retriever and a fixed perturbation
        grid `xs` created after init.

        Args:
            retriever (ExplainableAutoModelForRetrieval):
                Explainable retrieval model that provides .forward(...), explanation
                methods (e.g. 'IntGrad') and exposes tokenizers/encoders required for
                perturbation experiments.
            query_format (str, optional):
                Format string used to build the actual query passed to the retriever.
                Default is '{query}'.
        '''
        self.retriever    = retriever
        self.query_format = query_format
        self.xs           = np.arange(0., 1.01, .01)

    def __call__(self, data:Dict[str, List], k:int=5, method:str='intGrad', *, step:int=1, normalize:bool=True, **kwargs) -> float:
        '''Compute a faithfulness score by comparing area-under-curve (AUC) values for two
        perturbation strategies: perturbing most-relevant features first (MoRF) and
        perturbing least-relevant features first (LeRF).
        This method:
        - Generates the MoRF perturbation curve and stores the result in self.morf.
        - Generates the LeRF perturbation curve and stores the result in self.lerf.
        - Computes the mean curve across examples for each strategy and computes the
            area under each mean curve.
        - Returns the absolute difference between the two AUC values.

        Args:
            data (Dict[str, List]):     Input data structure expected by self.perturbe (typically a mapping of
                                        input fields to lists/arrays of instances and any required metadata).
            k (int, optional):          Number of elements/features to consider during perturbation (e.g. top-k).
                                        Defaults to 5.
            method (str, optional):     Name of the attribution/importance method to use (forwarded to
                                        self.perturbe). Defaults to 'intGrad'.
            step (int, optional):       Perturbation granularity / number of features to remove per perturbation
                                        step (forwarded to self.perturbe). Defaults to 1.
            normalize (bool, optional): If `True`, y axis is normalized.
            **kwargs:                   Additional keyword arguments forwarded to self.retriever.forward.

        Returns:
            aipc (float):
                Absolute difference between the area under the mean MoRF curve and the
                area under the mean LeRF curve. A larger value indicates a greater
                separation between the two perturbation strategies.
        '''
        
        # perturbation curve most relevant first:
        self.morf = self.perturbe(data, True, k, method, step=step, desc='Computing MoRF', **kwargs)

        # perturbation curve least relevant first:
        self.lerf = self.perturbe(data, False, k, method, step=step, desc='Computing LeRF', **kwargs)

        # set last value to mean max_error:
        max_err = 0.5 * (self.morf[:,-1] + self.lerf[:,-1])
        self.morf[:,-1] = max_err
        self.lerf[:,-1] = max_err

        # normalize:
        if normalize:
            self.morf /= max_err[:,None]
            self.lerf /= max_err[:,None]

        # return area inside curves:
        return self.get_aipc()

    def perturbe(self, data:Dict[str, List], descending:bool, k:int=5, method:str='intGrad', *, step:int=1, desc:str='Computing perturbations', **kwargs):
        '''Compute perturbation-based fidelity curves by progressively masking context tokens
        and measuring the change in retrieval similarity.
        For each (query, context) pair in `data` this method:
        1. Obtains the top-`k` retrieved contexts and their baseline similarities via
            self.retriever.forward(..., reorder=True).
        2. Obtains a token-level relevancy/ranking for each retrieved context from
            getattr(self.retriever, method)()['context'].
        3. For each retrieved context, repeatedly masks an increasing number of the most
            relevant tokens (according to `descending` and the relevancy ranking) in
            increments of `step`, re-encodes the perturbed context, and measures the
            change in similarity between the query and the perturbed context (dot
            product of the first token vectors from the encoders' last_hidden_state).
        4. Records pairs of (fraction_of_tokens_masked, similarity_change) and
            interpolates the resulting curve onto self.xs. Returns all interpolated
            curves stacked into a NumPy array.

        Args:
             data (Dict[str, List]):    A dict with keys 'query' and 'context', each mapping
                                        to a list of tokenized inputs (or inputs compatible with the retriever).
                                        The lengths of these lists should match.
             descending (bool):         If True, treat larger relevancy scores as more important
                                        (i.e., mask tokens with larger relevancy first). If False, mask the
                                        least-relevant tokens first.
             k (int, optional):         Number of top retrieved contexts to consider per query.
                                        Defaults to 5.
             method (str, optional):    Name of the explanation method to call on the
                                        retriever. Defaults to 'IntGrad'.
             step (int, optional):      Number of additional tokens to mask at
                                        each perturbation step. Must be >= 1. Defaults to 1.
             desc (str, optional):      Optional description for the tqdm progress bar.
             **kwargs:                  Additional keyword arguments forwarded to self.retriever.forward.

        Returns:
             numpy.ndarray:
                Array of shape (M, len(self.xs)) where M is the total number
                of perturbation curves produced (num_queries * k). Each row is
                the interpolated perturbation curve mapping fraction of tokens masked ->
                change_in_similarity (sim_ptb - sim_baseline).
        '''

        ys = []

        # calculate explanations:
        for qry, ctx in tqdm(zip(data['query'], data['context']), total=len(data['query']), desc=desc):

            # calculate similarity online:
            retrieved_ids, similarity = self.retriever.forward(
                self.query_format.format(query=qry), ctx, k,
                reorder=True, **kwargs
            )

            # calculate relevancy scores:
            relevancy:torch.Tensor = getattr(self.retriever, method)()['context']

            with torch.no_grad():
                # get original input:
                qry_in:torch.Tensor = torch.tensor(self.retriever._x['query']).to(self.retriever.query_encoder.device)
                ctx_in:torch.Tensor = torch.tensor(self.retriever._x['context']).to(self.retriever.context_encoder.device)

                # get attention masks:
                qry_msk:torch.Tensor = qry_in != self.retriever.tokenizer.pad_token_id
                ctx_msk:torch.Tensor = ctx_in != self.retriever.tokenizer.pad_token_id

                # apply embedding to query:
                qry_out = self.retriever.query_encoder(input_ids=qry_in, attention_mask=qry_msk)

                for id, (sim, rel) in enumerate(zip(similarity, relevancy)):
                    rel = rel.argsort(descending=descending)

                    pc = [(0., 0.)]
                    for i in range(0, len(rel), step):
                        # copy contexts:
                        ctx_ptb = ctx_in.clone()

                        # mask tokens:
                        ctx_ptb[id, rel[:i+step]] = self.retriever.tokenizer.mask_token_id

                        # apply embedding to context:
                        ctx_out = self.retriever.context_encoder(input_ids=ctx_ptb, attention_mask=ctx_msk)

                        # compute similarity:
                        sim_ptb = qry_out.last_hidden_state[0, 0, :] @ ctx_out.last_hidden_state[id, 0, :]

                        pc.append(((i+step)/float(len(rel)), (sim_ptb-sim).cpu().numpy()))

                    # interpolate curve:
                    ys.append(np.interp(self.xs, *np.array(pc).T))

        # convert to numpy array:
        return np.stack(ys)
    
    def get_aipc(self, *, k:Optional[int]=None) -> float:
        '''Compute the area inside the pertubation curves (AIPC) between the mean MORF
        and LERF curves using the trapezoidal rule over self.xs.

        Args:
            k (int, optional):  Number of top documents (rows) to consider from self.morf and self.lerf.
                                If None (default), all available documents (len(self.morf)) are used.
        
        Returns:
            aipc (float):
                Aarea inside the pertubation curves (|∫ mean(self.morf[:k], axis=0) dx - ∫ mean(self.lerf[:k], axis=0) dx|)
        '''
        
        # set k to max available docs if unset:
        if k is None: k = len(self.morf)

        # return aipc:
        return np.abs(
            np.trapezoid(self.morf[:k].mean(axis=0), self.xs) -
            np.trapezoid(self.lerf[:k].mean(axis=0), self.xs)
        )
 
    def plot(self, ax:plt.Axes, *, k:Optional[int]=None) -> None:
        '''Render averaged MoRF and LeRF fidelity curves on a Matplotlib axis.
        This method computes the mean of the first `k` rows of `self.morf` and
        `self.lerf` along the first axis and plots the resulting MoRF and LeRF
        curves against `self.xs`. The area between the two curves is filled to
        visualize the gap, an equal aspect ratio is enforced, and axis labels
        and a legend are added.

        Args:
            ax (matplotlib.axes.Axes):  The axes on which to draw the plot.
            k (int, optional):          Number of top documents (rows) to consider from self.morf and self.lerf.
                                        If None (default), all available documents (len(self.morf)) are used.
        '''

        # set k to max available docs if unset:
        if k is None: k = len(self.morf)

        # calculate means:
        ys_morf = self.morf[:k].mean(axis=0)
        ys_lerf = self.lerf[:k].mean(axis=0)

        # plot to axis:
        ax.plot(self.xs, ys_morf, label='MoRF')
        ax.plot(self.xs, ys_lerf, label='LeRF')
        ax.fill_between(self.xs, ys_morf, ys_lerf, color='lightgray')
        ax.set_aspect(1)
        ax.legend()
        ax.set_xlabel('Masked Tokens [%]')
        ax.set_ylabel('Normalized Similarity $\Delta$ [%]')