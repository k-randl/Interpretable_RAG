import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm.autonotebook import tqdm

from typing import Dict, List, Tuple, Literal
from numpy.typing import NDArray

from src.Interpretable_RAG.utils import bootstrap_ci
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

    def __call__(self, data:Dict[str, List], k:int=5, target:Literal['query', 'context']='context', method:str='intGrad', *, step:int=1, normalize:bool=True, method_args:Dict[str, any]={}, **kwargs) -> float:
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
            data (Dict[str, List]):       Input data structure expected by self.perturbe (typically a mapping of
                                          input fields to lists/arrays of instances and any required metadata).
            k (int, optional):            Number of elements/features to consider during perturbation (e.g. top-k).
                                          Defaults to 5.
            target (str, optional):       Target to compute saliency for (default: 'context').
            method (str, optional):       Name of the attribution/importance method to use (forwarded to
                                          self.perturbe). Defaults to 'intGrad'.
            step (int, optional):         Perturbation granularity / number of features to remove per perturbation
                                          step (forwarded to self.perturbe). Defaults to 1.
            normalize (bool, optional):   If `True`, y axis is normalized.
            method_args (dict, optional): Keyword arguments forwarded to `method`.
            **kwargs:                     Additional keyword arguments forwarded to self.retriever.forward.

        Returns:
            aipc (float):
                Absolute difference between the area under the mean MoRF curve and the
                area under the mean LeRF curve. A larger value indicates a greater
                separation between the two perturbation strategies.
        '''

        # perturbation curve most relevant first:
        self.morf = self.perturbe(data, True, k, target, method, step=step, method_args=method_args, desc='Computing MoRF', **kwargs)

        # perturbation curve least relevant first:
        self.lerf = self.perturbe(data, False, k, target, method, step=step, method_args=method_args, desc='Computing LeRF', **kwargs)

        # set last value to mean max_error:
        max_err = 0.5 * (self.morf[:,-1] + self.lerf[:,-1])
        self.morf[:,-1] = max_err
        self.lerf[:,-1] = max_err

        # normalize:
        if normalize:
            max_err   = np.abs(max_err[:,None])
            self.morf /= max_err
            self.lerf /= max_err

        # return area inside curves:
        return self.get_aipc()

    def perturbe(self, data:Dict[str, List], descending:bool, k:int=5, target:Literal['query', 'context']='context', method:str='intGrad', *,
                 step:int=1, method_args:Dict[str, any]={}, desc:str='Computing perturbations', **kwargs):
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
            data (Dict[str, List]):       A dict with keys 'query' and 'context', each mapping
                                          to a list of tokenized inputs (or inputs compatible with the retriever).
                                          The lengths of these lists should match.
            descending (bool):            If True, treat larger relevancy scores as more important
                                          (i.e., mask tokens with larger relevancy first). If False, mask the
                                          least-relevant tokens first.
            k (int, optional):            Number of top retrieved contexts to consider per query.
                                          Defaults to 5.
            target (str, optional):       Target to compute saliency for (default: 'context').
            method (str, optional):       Name of the explanation method to call on the
                                          retriever. Defaults to 'IntGrad'.
            step (int, optional):         Number of additional tokens to mask at
                                          each perturbation step. Must be >= 1. Defaults to 1.
            method_args (dict, optional): Keyword arguments forwarded to `method`.
            desc (str, optional):         Optional description for the tqdm progress bar.
             **kwargs:                    Additional keyword arguments forwarded to self.retriever.forward.

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
            if   method == 'random': relevancy = torch.rand(self.retriever._x[target].shape)
            elif method == 'grad':   relevancy = [doc.mean(axis=-1) for doc in self.retriever.grad(**method_args)[target]]
            elif method == 'aGrad':  relevancy = [doc.mean(axis=0) for doc in self.retriever.aGrad(**method_args)[target]]
            else:                    relevancy = getattr(self.retriever, method)(**method_args)[target]

            with torch.no_grad():
                # get original input:
                qry_in:torch.Tensor = torch.tensor(self.retriever._x['query']).to(self.retriever.query_encoder.device)
                ctx_in:torch.Tensor = torch.tensor(self.retriever._x['context']).to(self.retriever.context_encoder.device)

                # get attention masks:
                qry_msk:torch.Tensor = qry_in != self.retriever.tokenizer.pad_token_id
                ctx_msk:torch.Tensor = ctx_in != self.retriever.tokenizer.pad_token_id

                if target == 'query':
                    ys.append(self._perturbe_qry(qry_in, ctx_in, qry_msk, ctx_msk, similarity, relevancy, descending, step))

                elif target == 'context':
                    ys.append(self._perturbe_ctx(qry_in, ctx_in, qry_msk, ctx_msk, similarity, relevancy, descending, step))
                
                else: raise ValueError('target')
            
        # convert to numpy array:
        return np.concat(ys)

    def _perturbe_qry(self, qry_in, ctx_in, qry_msk, ctx_msk, similarity, relevancy, descending, step):
        # apply embedding to context once (we will perturb the query):
        ctx_out = self.retriever.context_encoder(input_ids=ctx_in, attention_mask=ctx_msk)

        rel = relevancy[0].argsort(descending=descending)

        pcs = [[(0., 0.)] for _ in similarity]
        for i in range(0, len(rel), step):
            # copy query:
            qry_ptb = qry_in.clone()

            # mask tokens:
            qry_ptb[0, rel[:i+step]] = self.retriever.tokenizer.mask_token_id

            # apply embedding to query:
            qry_out = self.retriever.query_encoder(input_ids=qry_ptb, attention_mask=qry_msk)

            # compute similarity:
            sim_ptb = qry_out.last_hidden_state[0, 0, :] @ ctx_out.last_hidden_state[:, 0, :].T

            for i, sim in enumerate((sim_ptb.cpu() - similarity.cpu()).numpy()):
                pcs[i].append(((i+step)/float(len(rel)), sim))

        # interpolate curve:
        ys = [np.interp(self.xs, *np.array(pc).T) for pc in pcs]
            
        # convert to numpy array:
        return np.stack(ys)

    def _perturbe_ctx(self, qry_in, ctx_in, qry_msk, ctx_msk, similarity, relevancy, descending, step):
        # apply embedding to query once (perturbing the context):
        qry_out = self.retriever.query_encoder(input_ids=qry_in, attention_mask=qry_msk)

        ys = []
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

    def get_aipc(self, *, num_samples=1000, confidence_level=0.95) -> Tuple[float, float, float]:
        '''Compute the area inside the pertubation curves (AIPC) between the mean MORF
        and LERF curves using the trapezoidal rule over self.xs.

        Args:
            num_samples (int, optional):        Number of bootstrap resamples to draw. Default is 1000.
            confidence_level (float, optional): Confidence level for the interval, between 0 and 1. Default is 0.95.

        Returns:
            tuple: 
                - **aipc** Mean area inside the pertubation curves (|∫ mean(self.morf[:k], axis=0) dx| - |∫ mean(self.lerf[:k], axis=0) dx|)
                - **lower_bound** of the percentile-based bootstrap confidence interval.
                - **upper_bound** of the percentile-based bootstrap confidence interval.
        '''
        
        # compute aipc per sample:
        aupc_morf = np.stack([np.abs(np.trapezoid(ys, self.xs)) for ys in self.morf])
        aupc_lerf = np.stack([np.abs(np.trapezoid(ys, self.xs)) for ys in self.lerf])
        aipc = aupc_morf - aupc_lerf

        # return aipc:
        return (float(aipc.mean(axis=0)),) + bootstrap_ci(aipc, num_samples=num_samples, confidence_level=confidence_level)

    def plot_lerf(self, ax:plt.Axes, *, label:str='LeRF', **kwargs) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        '''Plot the LeRF curve on a matplotlib axis.

        Compute the mean of the stored LeRF values across the first dimension and
        plot the resulting curve against self.xs on the provided Axes.

        Args:
            ax (matplotlib.axes.Axes):  The axes on which to draw the plot.
            label (str, optional):      Label to apply to the plotted line (default: 'LeRF').
            **kwargs:                   Additional keyword arguments forwarded to matplotlib.axes.Axes.plot
                                        (e.g., color, linestyle, linewidth).
        
        Returns:
            `Tuple[numpy.ndarray, numpy.ndarray]` of x and y yalues of the plotted curve.
        '''

        # calculate means and convert to percent:
        xs = self.xs * 100.
        ys = self.lerf.mean(axis=0) * 100.

        # plot to axis:
        ax.plot(xs, ys, label=label, **kwargs)

        return xs, ys

    def plot_morf(self, ax:plt.Axes, *, label:str='MoRF', **kwargs) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        '''Plot the MoRF curve on a matplotlib axis.

        Compute the mean of the stored MoRF values across the first dimension and
        plot the resulting curve against self.xs on the provided Axes.

        Args:
            ax (matplotlib.axes.Axes):  The axes on which to draw the plot.
            label (str, optional):      Label to apply to the plotted line (default: 'MoRF').
            **kwargs:                   Additional keyword arguments forwarded to matplotlib.axes.Axes.plot
                                        (e.g., color, linestyle, linewidth).
        
        Returns:
            `Tuple[numpy.ndarray, numpy.ndarray]` of x and y yalues of the plotted curve.
        '''

        # calculate means and convert to percent:
        xs = self.xs * 100.
        ys = self.morf.mean(axis=0) * 100.

        # plot to axis:
        ax.plot(xs, ys, label=label, **kwargs)

        return xs, ys

    def plot(self, ax:plt.Axes) -> None:
        '''Render averaged MoRF and LeRF fidelity curves on a Matplotlib axis.
        This method computes the mean of the documents and plots the
        resulting MoRF and LeRF curves. The area between the two curves is filled to
        visualize the gap, an equal aspect ratio is enforced, and axis labels
        and a legend are added.

        Args:
            ax (matplotlib.axes.Axes):  The axes on which to draw the plot.
        '''

        # plot to axis:
        _, ys_morf = self.plot_morf(ax)
        _, ys_lerf = self.plot_lerf(ax)
        ax.fill_between(self.xs*100., ys_morf, ys_lerf, color='lightgray')
        ax.set_aspect(1)
        ax.legend()
        ax.set_xlabel('Masked Tokens [%]')
        ax.set_ylabel('Normalized Similarity $\Delta$ [%]')