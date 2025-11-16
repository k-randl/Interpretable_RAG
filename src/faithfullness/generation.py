import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm.autonotebook import tqdm
from typing import Optional, Dict, List

from typing import Optional, Dict, List, Tuple
from numpy.typing import NDArray

from src.Interpretable_RAG.generation import ExplainableAutoModelForGeneration, generate_permutations, create_rag_prompt

#=======================================================================#
# Generator Explanation Faithfullness:                                  #
#=======================================================================#

class AIPCForGeneration:
    def __init__(self, generator:ExplainableAutoModelForGeneration) -> None:
        '''Initialize the AIPCForGeneration evaluator. This class computes
        area-inside-perturbation-curves (AIPC) for generation explanations by iteratively
        masking tokens in generated contexts according to relevancy scores and measuring
        the change in similarity. The object stores the generator and a fixed perturbation
        grid `xs` created after init.

        Args:
            generator (GeneratorExplanation):
                Explainable generation model that provides .explain_generate(...) and exposes tokenizers/encoders required for
                perturbation experiments.
        '''
        self.generator      = generator
        self.xs             = np.arange(0., 1.01, .01)

    def __call__(self, data:Dict[str, List], batch_size:int=64, *,
            system        :Optional[str]=None,
            num_mc_samples:int=100,
            mc_sample_size:int=10,
            step          :int=1,
            normalize     :bool=True,
            **kwargs
        ) -> float:
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
            data (Dict[str, List]):         Input data structure expected by self.perturbe (typically a mapping of
                                            input fields to lists/arrays of instances and any required metadata).
            batch_size (int, optional):     Batch size for LLM calls (default: 64).
            system (str, optional):         An optional system prompt.
            num_mc_samples (int, optional): Number of samples for Monte-Carlo approximation.
                                            Ignored in case of precise calculation (default: `100`).
            mc_sample_size (int, optional): Size of samples for Monte-Carlo approximation.
                                            Ignored in case of precise calculation (default: `10`).
            step (int, optional):           Perturbation granularity / number of features to remove per perturbation
                                            step (forwarded to self.perturbe). Defaults to 1.
            normalize (bool, optional):     If `True`, y axis is normalized.
            **kwargs:                       Additional keyword arguments forwarded to self.generator.forward.

        Returns:
            aipc (float):
                Absolute difference between the area under the mean MoRF curve and the
                area under the mean LeRF curve. A larger value indicates a greater
                separation between the two perturbation strategies.
        '''

        # get data size:
        num_queries = len(data['query'])
        num_points  = len(self.xs)

        # calculate explanations:
        self.morf, self.lerf = {}, {}
        for i, (qry, ctx) in enumerate(tqdm(zip(data['query'], data['context']), total=num_queries, desc='Computing perturbations')):

            # calculate relevancy scores:
            relevancy = {}

            # Generate precise explanations:
            self.generator.explain_generate(
                query=qry,
                contexts=ctx[:5],
                max_samples=64,
                max_samples_query='auto',
                max_samples_context='inf',
                conditional=True,
                **kwargs
            )

            # Get shapley values for the generated tokens:
            relevancy['Precise'] = self.generator.get_shapley_values('context', 'token')

            # Get random baseline:
            relevancy['Random'] = np.random.random(relevancy['Precise'].shape)

            # Generate approximated explanations:
            self.generator.explain_generate(
                query=qry,
                contexts=ctx[:5],
                max_samples_query='auto',
                max_samples_context=30,
                batch_size=batch_size,
                conditional=True,
                **kwargs
            )

            # Get the shap parameters:
            indices = self.generator._shap_cache['context']['indices'].copy()
            sets    = self.generator._shap_cache['context']['sets'].copy()

            # Get shapley values for the generated tokens:
            for max_samples in range(mc_sample_size, 31, 5):
                # Set the number of samples:
                sample_indices = np.random.choice(len(indices)-2, size=max_samples-2, replace=False) + 1 # skip the first and last indices
                self.generator._shap_cache['context']['indices'] = np.concatenate([indices[:1], indices[sample_indices], indices[-1:]])
                self.generator._shap_cache['context']['sets']    = np.concatenate([sets[:1], sets[sample_indices], sets[-1:]])

                # Get kernel shapley values:
                relevancy[f'Kernel (n = {max_samples:d})'] = self.generator.get_shapley_values('context', 'token', num_samples=1, sample_size=max_samples)

                # Get Monte Carlo approximated shapley values:
                relevancy[f'Monte Carlo (n = {max_samples:d})'] = self.generator.get_shapley_values('context', 'token', num_samples=num_mc_samples, sample_size=mc_sample_size)

            for key in relevancy:
                # generate prompts for perturbed inputs:
                permutations, new_items, perturbed_prompts = generate_permutations(
                    ctx,                                                       # permute the contexts
                    lambda items:create_rag_prompt(qry, items, system=system), # build a prompt for each permutation
                )

                # generate comparison output:
                num_batches = int(np.ceil(len(perturbed_prompts) / batch_size))
                for j in range(num_batches):
                    # print batch number:
                    if num_batches > 1: print(f'Batch {j+1:d} of {num_batches:d}:')

                    # get prompts of this batch:
                    prompts_batch = perturbed_prompts[j * batch_size:(j+1) * batch_size]

                    # generate probabilities:
                    self.generator.compare(
                        [self.generator.tokenizer.apply_chat_template(prmpt, tokenize=False) for prmpt in prompts_batch],
                        'last'
                    )

                # Flatten each token probability array from compared documents:
                probs = np.stack([p.flatten() for p in self.generator.cmp_token_probs])[-permutations.max()-1:]

                # perturbation curve most relevant first:
                if key not in self.morf: self.morf[key] = np.full((num_queries, num_points), np.nan, dtype=float)
                self.morf[key][i] = self._make_pc(relevancy[key], True, permutations, new_items, probs, step=step).mean(axis=0)

                # perturbation curve least relevant first:
                if key not in self.lerf: self.lerf[key] = np.full((num_queries, num_points), np.nan, dtype=float)
                self.lerf[key][i] = self._make_pc(relevancy[key], False, permutations, new_items, probs, step=step).mean(axis=0)

        for key in self.morf:
            # set first value to mean:
            y_min = 0.5 * (self.morf[key][:,0] + self.lerf[key][:,0])
            self.morf[key][:,0] = y_min
            self.lerf[key][:,0] = y_min

            # set last value to mean:
            y_max = 0.5 * (self.morf[key][:,-1] + self.lerf[key][:,-1])
            self.morf[key][:,-1] = y_max
            self.lerf[key][:,-1] = y_max

            # normalize:
            if normalize:
                self.morf[key] -= y_min[:,None]
                self.morf[key] /= (y_max-y_min)[:,None]

                self.lerf[key] -= y_min[:,None]
                self.lerf[key] /= (y_max-y_min)[:,None]

        # return area inside curves:
        return {key:self.get_aipc(key) for key in self.morf}

    def _make_pc(self, relevancy:NDArray[np.float64], descending:bool, permutations:NDArray[np.int_], new_items:NDArray[np.int_], probs:NDArray[np.float64], *, step:int=1):
        
        # calculate relevancy:
        rel = relevancy.argsort(axis=0)
        if descending: rel = rel[::-1]

        # get perturbation paths:
        idx = np.concatenate([np.nonzero((new_items==token).all(axis=1))[0] for token in rel.T])
        idx = permutations[idx]

        # select curve values:
        xs = np.linspace(0., 1., idx.shape[1])
        ys = np.stack([np.interp(self.xs, xs, [probs[j,i] for j in row]) for i, row in enumerate(idx)])

        return ys

    def get_aipc(self, key:str) -> float:
        '''Compute the area inside the pertubation curves (AIPC) between the mean MORF
        and LERF curves using the trapezoidal rule over self.xs.

        Args:
            key (str):
                Identifier used to select which set of perturbation curves to use from the instance.
        
        Returns:
            aipc (float):
                Aarea inside the pertubation curves (|∫ mean(self.morf[key], axis=0) dx - ∫ mean(self.lerf[key], axis=0) dx|)
        '''
        
        # return aipc:
        return np.abs(
            np.trapezoid(self.morf[key].mean(axis=0), self.xs) -
            np.trapezoid(self.lerf[key].mean(axis=0), self.xs)
        )
 
    def plot_lerf(self, ax:plt.Axes, key:str, *, label:str='LeRF', **kwargs) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        '''Plot the LeRF curve on a matplotlib axis.

        Compute the mean of the stored LeRF values across the first dimension for the first documents
        and plotd the resulting curve against self.xs on the provided Axes.

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
        ys = self.lerf[key].mean(axis=0) * 100.

        # plot to axis:
        ax.plot(xs, ys, label=label, **kwargs)

        return xs, ys

    def plot_morf(self, ax:plt.Axes, key:str, *, label:str='MoRF', **kwargs) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        '''Plot the MoRF curve on a matplotlib axis.

        Compute the mean of the stored MoRF values across the first dimension for the documents
        and plots the resulting curve against self.xs on the provided Axes.

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
        ys = self.morf[key].mean(axis=0) * 100.

        # plot to axis:
        ax.plot(xs, ys, label=label, **kwargs)

        return xs, ys

    def plot(self, ax:plt.Axes, key:str) -> None:
        '''Render averaged MoRF and LeRF fidelity curves on a Matplotlib axis.
        This method computes the mean of the documents and plots the resulting
        MoRF and LeRF curves. The area between the two curves is filled to
        visualize the gap, an equal aspect ratio is enforced, and axis labels
        and a legend are added.

        Args:
            ax (matplotlib.axes.Axes):  The axes on which to draw the plot.
        '''

        # plot to axis:
        _, ys_morf = self.plot_morf(ax, key)
        _, ys_lerf = self.plot_lerf(ax, key)
        ax.fill_between(self.xs*100., ys_morf, ys_lerf, color='lightgray')
        ax.set_aspect(1)
        ax.legend()
        ax.set_xlabel('Masked Documents [%]')
        ax.set_ylabel('Normalized $\Delta$ Token Probability [%]')