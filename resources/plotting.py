import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from html import escape
from IPython.display import display, HTML

from resources.utils import decode_chat_template, nucleus_sample_tokens

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray
from typing import List, Literal, Tuple, Callable, Optional, Union

#====================================================================#
# General plotting functions:                                        #
#====================================================================#

def plot_token_vbars(ax:Axes, scores:NDArray[np.float_], tokens:List[str], document_names:Optional[List[str]]=None, *,
        normalize:bool=False,
        skip_tokens:List[str]=[],
        cmap:str='tab10',
        **kwargs
    ) -> None:
    """Creates a vertical bar plot for attribution scores of tokens.

    Args:
        ax (matplotlib.axes.Axes):   `matplotlib.axes.Axes` object.
        scores (NDArray[np.float_]): A 2D array of attribution scores with shape `(len(documents), len(tokens))`,
                                     where each row represents a document and each column corresponds to a token's
                                     attribution score.
        tokens (List[str]):          A list of token strings corresponding to the scores. Length must match the number
                                     of columns in `scores`.
        document_names (List[str]):  An optional list of names of the documents.
        normalize (bool):            If `True`, normalizes the attribution scores across tokens
                                     so that the sum of absolute values equals 1 (default=`False`).
        skip_tokens (List[str]):     An optional list of tokens that will not be printed.
        cmap (str):                  The name of a matplotlib colormap used for highlighting.
    """

    # filter tokens:
    scores = scores[:, [t not in skip_tokens for t in tokens]]
    tokens = [t for t in tokens if t not in skip_tokens]

    # normalize so that values sum to 1 (if enabled):
    if normalize: scores = scores / np.abs(scores).sum(axis=0)

    # get colormap:
    colors = cm.get_cmap(cmap)

    # fallback for document names:
    if document_names is None:
        document_names = [f'Document {i+1:d}' for i in range(len(scores))]
    elif len(document_names) != len(scores): raise ValueError('`len(document_names)` does not match the number of documents!')

    s_pos = 0.
    s_neg = 0.
    for i, s in enumerate(scores):
        s_doc_pos = np.maximum(s, 0)
        ax.bar(range(len(s)), s_doc_pos, bottom=s_pos, color=colors(i), label=document_names[i] if len(scores) > 1 else None, **kwargs)
        s_pos += s_doc_pos

        s_doc_neg = np.minimum(s, 0)
        ax.bar(range(len(s)), s_doc_neg, bottom=s_neg, color=colors(i))
        s_neg += s_doc_neg

    # tick and axis settings:
    ax.set_xticks(range(len(tokens)), tokens, rotation=90)
    if normalize:
        if np.any(s_neg < 0.): ax.set_ylim(-1, 1)
        else: ax.set_ylim(0, 1)

    # clean frame (keep only left spine):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    if ax.get_ylim()[0] < 0.:
        ax.spines['bottom'].set_visible(False)

        # add horizontal grid line at zero:
        ax.axhline(y=0, color='black', linewidth=0.5)

    # add vertical grid lines at each tick:
    for tick_loc in range(len(tokens)):
        ax.axvline(x=tick_loc, color='lightgray', linewidth=0.5, zorder=0)

def highlight_dominant_passages(scores:NDArray[np.float_], tokens:List[str], title:str, document_names:Optional[List[str]]=None, *,
        threshold:float=0.0,
        total:Optional[float]=None,
        skip_tokens:List[str]=[],
        token_processor:Optional[Callable[[str],str]]=None,
        legend:bool=True,
        cmap:str='tab10'
    ) -> Tuple[str, str]:
    """Highlights tokens in a text sequence where a single document contributes
    at least `threshold` of the total positive attribution scores at that token.

    Args:
        scores (NDArray[np.float_]):    A 2D array of attribution scores with shape `(len(documents), len(tokens))`,
                                        where each row represents a document and each column corresponds to a token's attribution score.
        tokens (List[str]):             A list of token strings corresponding to the scores.
                                        Length must match the number of columns in `scores`.
        document_names (List[str]):     An optional list of names of the documents.
        title (str):                    The title of the produced html table row.
        threshold (float):              Minimum attribution for highlighting in the intervall `[0., 1.]` (default: `0.0`).
        total (float):                  Optional Maximum attribution for highlighting (default: `None`).
        skip_tokens (List[str]):        An optional list of tokens that will not be printed.
        token_processor ((str) -> str): An otional function applied to each token before printing.
        legend (bool):                  Whether to create a legend for the plot (default: `True`).
        cmap (str):                     The name of a matplotlib colormap used for highlighting.

    Returns:
        A tuple `(text, legend)`, where both `text` and `legend` are html `<tr>` elements.
    """

    # check threshold value:
    if (threshold >= 1.) or (threshold < 0.):
        raise ValueError(f'Parameter `threshold` must be in intervall `[0,1))` but is `{threshold:.2f}`.')

    # extract only on positive contributions:
    scores_pos = np.maximum(scores, 0)
    num_docs, num_tokens = scores_pos.shape

    if total == None: total = scores_pos.sum(axis=0) if num_docs > 1 else scores_pos.max()
    total = np.maximum(total, 1e-9) # make sure that total > 0

    token_docs = scores_pos.argmax(axis=0)
    token_vals = scores_pos[token_docs, np.arange(num_tokens)] / total
    token_vals = np.maximum((token_vals - threshold) / (1. - threshold), 0.)
    token_docs[token_vals == 0.] = -1

    # prepare color map
    cmap = cm.get_cmap(cmap)
    rgb_colors = [mcolors.to_hex(cmap(i)) for i in range(num_docs)]

    # build highlighed HTML:
    html_tokens = []
    for tok, doc, val in zip(tokens, token_docs, token_vals):
        if tok in skip_tokens: continue

        if token_processor is not None:
            tok = token_processor(tok)

        html_tokens.append(
            f'<span style="background-color:{rgb_colors[doc]}{int((val)*255):02x}; padding:0px; border-radius:3px;">' +
            escape(tok) +
            '</span>'
        )

    html_text = (
        '<tr style="border-top: 1px solid">\n' +
        '   <td style="text-align:right; vertical-align:top">\n' +
        '       <b style="line-height:2">' + title + ':</b>\n' +
        '   </td>\n' +
        '   <td style="text-align:left; vertical-align:top">\n' +
        '       <div style="line-height:2">' + ''.join(html_tokens) + '</div>\n' +
        '   </td>\n' +
        '</tr>\n'
    )

    if not legend: return html_text

    # fallback for document names:
    if document_names is None:
        document_names = [f'Document {i+1:d}' for i in range(num_docs)]
    elif len(document_names) != num_docs: raise ValueError('`len(document_names)` does not match the number of documents!')

    # extract mean absolute document contributions:
    document_vals = scores.mean(axis=1)
    document_vals /= np.abs(document_vals).sum()

    # build legend:
    html_legend = (
        '<tr style="border-top: 1px solid">\n' +
        '   <td style="text-align:right; vertical-align:top">\n' +
        '       <i style="line-height:2">Legend:</i>\n' +
        '   </td>\n' +
        '   <td style="text-align:left; vertical-align:top">\n' +
        '       <div style="line-height:1">' +
                    ''.join([f'<div style="background-color:{c}; text-align: center; padding:3px; margin:3px; border-radius:3px; float:left;"><i>{document_names[i]}</i><br><small>({document_vals[i] * 100.:.0f}%)</small></div>' for i, c in enumerate(rgb_colors)]) +
                '</div>\n' +
        '   </td>\n' +
        '</tr>\n'
    )
    
    return html_text, html_legend

def plot_document_vbars(ax:Axes, scores:NDArray[np.float_], document_names:Optional[List[str]]=None, *,
        normalize:bool=False,
        cmap:str='tab10',
        **kwargs
    ) -> None:
    """Creates a vertical bar plot for attribution scores of documents.

    Args:
        ax (matplotlib.axes.Axes):   `matplotlib.axes.Axes` object.
        scores (NDArray[np.float_]): A 1D array of attribution scores with shape `(len(documents),)`,
                                     where each item represents a document's attribution score.
        document_names (List[str]):  An optional list of names of the documents.
        normalize (bool):            If `True`, normalizes the attribution scores across documents so
                                     that the sum of absolute values equals 1 (default=`False`).
        cmap (str):                  The name of a matplotlib colormap used for highlighting.
    """
    # normalize so that values sum to 1 (if enabled):
    if normalize: scores = scores / np.abs(scores).sum()

    # fallback for document names:
    if document_names is None:
        document_names = [f'Document {i+1:d}' for i in range(len(scores))]
    elif len(document_names) != len(scores): raise ValueError('`len(document_names)` does not match the number of documents!')

    # get color:
    color = kwargs.get('color', cm.get_cmap(cmap)(0))

    # create bar plot:
    bars = ax.bar(range(len(document_names)), scores, color=color, **kwargs)

    # add value labels on bars:
    for bar, value in zip(bars, scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
            f'{value:.4f}', ha='center', va='bottom')

    # tick and axis settings:
    ax.set_xticks(range(len(document_names)), document_names, rotation=45, ha='right')
    if normalize: ax.set_ylim(-1, 1)

    # clean frame (keep only left spine):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    if ax.get_ylim()[0] < 0.:
        ax.spines['bottom'].set_visible(False)

        # add horizontal grid line at zero:
        ax.axhline(y=0, color='black', linewidth=0.5)

    # add horizontal grid lines at each tick:
    for tick_loc in ax.get_yticks(minor=False):
        ax.axhline(y=tick_loc, color='lightgray', linewidth=0.5, zorder=0)

def plot_waterfall(ax:Axes, scores:NDArray[np.float_], x_labels:Optional[List[str]], *,
        base_value:float=0.0,
        normalize:bool=False,
        cmap:str='tab10',
        **kwargs
    ) -> None:
    """Create a waterfall plot showing cumulative scores.
    
    Args:
        ax (matplotlib.axes.Axes):   `matplotlib.axes.Axes` object.
        scores (NDArray[np.float_]): A 1D array of attribution scores with shape `(len(documents),)`,
                                     where each item represents a document's attribution score.
        x_labels (List[str]):        The list of x-axis labels.
        base_value (float):          Base value to start the waterfall from (default: `0.0`).
        normalize (bool):            If `True`, normalizes the attribution scores across documents so
                                     that the sum of absolute values equals 1 (default=`False`).
        cmap (str):                  The name of a matplotlib colormap used for highlighting.
    """
    # normalize so that values sum to 1 (if enabled):
    if normalize: scores = scores / np.abs(scores).sum()

    # calculate cumulative values:
    cumulative = np.cumsum(np.concatenate([[base_value], scores]))

    # get colormap:
    colors = cm.get_cmap(cmap)

    # plot each contribution:
    for i, value in enumerate(scores):
        color = colors(0) if value > 0 else colors(1)
        ax.bar(i, value, bottom=cumulative[i], color=color, **kwargs)
        
        # add value label:
        ax.text(i, cumulative[i] + value/2, f'{value:.2f}', 
               ha='center', va='center', fontweight='bold', color='white')

    # plot final value:
    ax.bar(len(x_labels), cumulative[-1], color=colors(2), **kwargs)
        
    # add value label:
    ax.text(len(x_labels), cumulative[-1]/2, f'{cumulative[-1]:.2f}', 
            ha='center', va='center', fontweight='bold', color='white')

    # connect bars with lines:
    for i in range(len(x_labels)):
        ax.plot([i + 0.4, i + 1.6], [cumulative[i+1], cumulative[i+1]], 'k--', alpha=0.5)

    # tick and axis settings:
    ax.set_xticks(range(len(x_labels) + 1), x_labels + ['Total'], rotation=45, ha='right')
    if normalize: 
        if ax.get_ylim()[0] < 0: ax.set_ylim(-1, 1)
        else: ax.set_ylim(0, 1)

    # clean frame (keep only left spine):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    if ax.get_ylim()[0] < 0.:
        ax.spines['bottom'].set_visible(False)

        # add horizontal grid line at zero:
        ax.axhline(y=0, color='black', linewidth=0.5)

    # add horizontal grid lines at each tick:
    for tick_loc in ax.get_yticks(minor=False):
        ax.axhline(y=tick_loc, color='lightgray', linewidth=0.5, zorder=0)

#====================================================================#
# Plots for retrieval:                                               #
#====================================================================#

from resources.retrieval import RetrieverExplanationBase

def plot_importance_retriever(explanation:RetrieverExplanationBase, document_names:Optional[List[str]]=None, *,
        method:Literal['grad', 'gradIn', 'aGrad']='gradIn',
        absolute:bool=True,
        figsize: Tuple[int, int] = (12, 6),
        cmap:str='tab10',
        show:bool=True,
        **kwargs
    ) -> Union[Figure, None]:
    """Plot tokens in a text sequence that are important for retrieving the document.

    Args:
        explanation (RetrieverExplanationBase): An object containing the necessary information for plotting.
        document_names (List[str]):             An optional list of names of the documents.
        method (str):                           The method for calculating token importance.
        aggregation (str):                      Aggregation method for probabilities (default: `'token'`).
        normalize (bool):                       If `True`, normalizes the SHAP values across tokens so that the sum of absolute values equals 1 (default=`True`).
        figsize ((int, int)):                   The size of the figure.
        cmap (str):                             The name of a matplotlib colormap used for highlighting.
        show (bool):                            If `True` shows the plot directly, if `False` the plot is returned instead (default: `True`).

    Returns:
        `matplotlib.figure.Figure` object if `show == False`.
    """

    # get scores:
    if   method == 'grad':   scores = {key:[doc.mean(axis=-1) for doc in docs] for key, docs in explanation.grad(**kwargs).items()}
    elif method == 'gradIn': scores = explanation.gradIn(**kwargs)
    elif method == 'aGrad':  scores = {key:[doc.mean(axis=0) for doc in docs] for key, docs in explanation.aGrad(**kwargs).items()}
    else: raise ValueError()

    # compute absolute:
    if absolute: scores = {key:[np.abs(doc) for doc in docs] for key, docs in scores.items()}

    # fallback for document names:
    if document_names is None:
        document_names = [f'Doc. {i+1:d}' for i in range(len(scores['context']))]
    elif len(document_names) != len(scores['context']): raise ValueError('`len(document_names)` does not match the number of documents!')

    # get special tokens:
    special_tokens = set(explanation.tokenizer.special_tokens_map.values())

    fig, axs = plt.subplots(len(scores['query']) + len(scores['context']), 1, figsize=figsize)

    # plot query:
    plot_token_vbars(axs[0], 
        scores      = scores['query'][0].numpy()[None,:], 
        tokens      = explanation.in_tokens['query'][0],
        skip_tokens = special_tokens,
        cmap        = cmap
    )
    axs[0].set_ylabel('Query:')

    # plot contexts:
    for i, s in enumerate(scores['context']):
        plot_token_vbars(axs[i+1],
            scores      = s.numpy()[None,:], 
            tokens      = explanation.in_tokens['context'][i],
            skip_tokens = special_tokens,
            cmap        = cmap
        )
        axs[i+1].set_ylabel(document_names[i] + ':')

    fig.suptitle(f'Token Importance - {method}', fontsize=14, fontweight='bold')
    fig.tight_layout()

    if show: fig.show()
    else: return fig

def higlight_importance_retriever(explanation:RetrieverExplanationBase, document_names:Optional[List[str]]=None, *,
        method:Literal['grad', 'gradIn', 'aGrad']='gradIn',
        threshold:float=0.0,
        token_processor:Optional[Callable[[str],str]]=None,
        cmap:str='tab10',
        show:bool=True,
        **kwargs
    ) -> Union[str,None]:
    """Highlights tokens in a text sequence that are important for retrieving the document.

    Args:
        explanation (RetrieverExplanationBase): An object containing the necessary information for plotting.
        document_names (List[str]):             An optional list of names of the documents.
        method (str):                           The method for calculating token importance.
        threshold (float):                      Minimum importance for highlighting in the intervall `[0., 1.)` (default: `0.`).
        token_processor ((str) -> str):         An optional function applied to each token before printing.
        cmap (str):                             The name of a matplotlib colormap used for highlighting.
        show (bool):                            If `True` shows the plot directly, if `False` the plot is returned instead (default: `True`).

    Returns:
        A HTML-formatted string with spans highlighting important tokens if `show == False`.
    """
    # get scores:
    if   method == 'grad':   scores = {key:[doc.mean(axis=-1) for doc in docs] for key, docs in explanation.grad(**kwargs).items()}
    elif method == 'gradIn': scores = explanation.gradIn(**kwargs)
    elif method == 'aGrad':  scores = {key:[doc.mean(axis=0) for doc in docs] for key, docs in explanation.aGrad(**kwargs).items()}
    else: raise ValueError()

    # compute absolute:
    scores = {key:[np.abs(doc) for doc in docs] for key, docs in scores.items()}

    # fallback for document names:
    if document_names is None:
        document_names = [f'Document {i+1:d}' for i in range(len(scores['context']))]
    elif len(document_names) != len(scores['context']): raise ValueError('`len(document_names)` does not match the number of documents!')

    # get special tokens:
    special_tokens = set(explanation.tokenizer.special_tokens_map.values())

    # plot query:
    html = highlight_dominant_passages(
        scores          = scores['query'][0].numpy()[None,:], 
        tokens          = explanation.in_tokens['query'][0],
        title           = 'Query',
        threshold       = threshold,
        skip_tokens     = special_tokens,
        token_processor = token_processor,
        legend          = False,
        cmap            = cmap
    )

    # plot contexts:
    for i, s in enumerate(scores['context']):
        html += highlight_dominant_passages(
            scores          = s.numpy()[None,:], 
            tokens          = explanation.in_tokens['context'][i],
            title           = document_names[i],
            threshold       = threshold,
            total           = np.concatenate(scores['context']).max(),
            skip_tokens     = special_tokens,
            token_processor = token_processor,
            legend          = False,
            cmap            = cmap
        )

    # Build, display, and return final HTML:
    html_str = (
        '<!DOCTYPE html>\n' +
        '<html>\n' +
        '<head>\n' +
        '<title>' + method + '</title>\n' +
        '</head>\n' +

        '<body>\n' +
        '<table>\n' +
        html +
        '</table>\n' +
        '</body>\n' +

        '</html>'
    )

    if show: display(HTML(html_str))
    else: return html_str

def plot_importance_summary_retriever(explanation:RetrieverExplanationBase, document_names:Optional[List[str]]=None, *, 
        method:Literal['grad', 'gradIn', 'aGrad']='gradIn',
        normalize:bool=True,
        threshold:float=.9,
        figsize: Tuple[int, int] = (12, 6),
        cmap:str='tab10',
        show:bool=True,
        **kwargs
    ) -> Union[Figure, None]:
    """Create a summary plot showing the most important tokens per document.

    Args:
        explanation (GeneratorExplanationBase): An object containing the necessary information for plotting.
        document_names (List[str]):             An optional list of names of the documents.
        method (str):                           The method for calculating token importance.
        normalize (bool):                       If `True`, normalizes the importance values across tokens so that the sum of absolute values equals 1 (default=`True`).
        threshold (float):                      The accumulated perctentual impact of shown tokens per document in the interval `[0.,1.]` (default: `.9`).
        figsize ((int, int)):                   The size of the figure.
        cmap (str):                             The name of a matplotlib colormap used for highlighting.
        show (bool):                            If `True` shows the plot directly, if `False` the plot is returned instead (default: `True`).

    Returns:
        `matplotlib.figure.Figure` object if `show == False`.
    """

    # get scores:
    if   method == 'grad':   scores = {key:[doc.mean(axis=-1) for doc in docs] for key, docs in explanation.grad(**kwargs).items()}
    elif method == 'gradIn': scores = explanation.gradIn(**kwargs)
    elif method == 'aGrad':  scores = {key:[doc.mean(axis=0) for doc in docs] for key, docs in explanation.aGrad(**kwargs).items()}
    else: raise ValueError()

    # compute absolute:
    scores = {key:[np.abs(doc) for doc in docs] for key, docs in scores.items()}
    
    # fallback for document names:
    if document_names is None:
        document_names = [f'Doc. {i+1:d}' for i in range(len(scores['context']))]
    elif len(document_names) != len(scores['context']): raise ValueError('`len(document_names)` does not match the number of documents!')

    # create figure:
    fig, axs = plt.subplots(len(scores['query']) + len(scores['context']), 1, figsize=figsize)

    # plot query:
    s, t = nucleus_sample_tokens(scores['query'][0], explanation.in_tokens['query'][0], threshold)
    plot_waterfall(axs[0],
        scores    = s,
        x_labels  = t,
        normalize = normalize,
        cmap      = cmap 
    )
    axs[0].set_ylabel('Query:')

    # plot contexts:
    for i, s in enumerate(scores['context']):
        s, t = nucleus_sample_tokens(s, explanation.in_tokens['context'][i], threshold)
        plot_waterfall(axs[i+1],
            scores    = s,
            x_labels  = t,
            normalize = normalize,
            cmap      = cmap 
        )
        axs[i+1].set_ylabel(document_names[i] + ':')

    # title and x-label:
    fig.suptitle(f'Importance Summary - {method}', fontsize=14, fontweight='bold')
    fig.tight_layout()

    if show: fig.show()
    else: return fig

def visualize_importance_retriever(explanation:RetrieverExplanationBase, document_names:Optional[List[str]]=None, *,
        method:Literal['grad', 'gradIn', 'aGrad']='gradIn',
        cmap:str='tab10',
        show:bool=True,
        **kwargs
    ) -> Union[Figure, str, None]:
    """Visualizes token importance while automatically choosing a fitting plot for the chosen `method`.

    Args:
        explanation (RetrieverExplanationBase): An object containing the necessary information for plotting.
        document_names (List[str]):             An optional list of names of the documents.
        method (str):                           The method for calculating token importance.
        cmap (str):                             The name of a matplotlib colormap used for highlighting.
        show (bool):                            If `True` shows the plot directly, if `False` the plot is returned instead (default: `True`).

    Returns:
        The illustration object if `show == False`.
    """

    return higlight_importance_retriever(
        explanation    = explanation,
        document_names = document_names, 
        method         = method,
        cmap           = cmap,
        show           = show,
        **kwargs
    )

#====================================================================#
# Plots for generation:                                              #
#====================================================================#

from resources.generation import GeneratorExplanationBase

def plot_attribution_generator(explanation:GeneratorExplanationBase, document_names:Optional[List[str]]=None, *,
        aggregation:Literal['token', 'bow', 'nucleus']='token',
        normalize:bool=True,
        figsize: Tuple[int, int] = (12, 6),
        cmap:str='tab10',
        show:bool=True
    ) -> Union[Figure, None]:
    """Plot stacked SHAP attributions for multiple documents as positive and negative bar segments.

    Args:
        explanation (GeneratorExplanationBase): An object containing the necessary information for plotting.
        document_names (List[str]):             An optional list of names of the documents.
        aggregation (str):                      Aggregation method for probabilities (default: `'token'`).
        normalize (bool):                       If `True`, normalizes the SHAP values across tokens so that the sum of absolute values equals 1 (default=`True`).
        figsize ((int, int)):                   The size of the figure.
        cmap (str):                             The name of a matplotlib colormap used for highlighting.
        show (bool):                            If `True` shows the plot directly, if `False` the plot is returned instead (default: `True`).

    Returns:
        `matplotlib.figure.Figure` object if `show == False`.
    """

    # get attribution scores:
    shap_values = explanation.get_shapley_values('context', aggregation)

    # get tokens:
    if aggregation == 'token':
        tokens = explanation.gen_tokens

    elif aggregation == 'bow':
        # get the top-K varying tokens per document:
        K = 50
        indices = np.unique(shap_values.var(axis=0).argsort()[:-K:-1])

        shap_values = shap_values[:,indices]
        tokens = explanation.tokenizer.convert_ids_to_tokens(indices[:,None])

    elif aggregation == 'nucleus':
        # get the top-K probable tokens per document: 
        K = 10
        indices = np.unique(shap_values.argsort()[:,:-K:-1].flatten())

        shap_values = shap_values[:,indices]
        tokens = explanation.tokenizer.convert_ids_to_tokens(indices[:,None])

    else: raise ValueError(f'Unknown value for parameter `aggregation`: "{aggregation}"')

    # get special tokens:
    special_tokens = set(explanation.tokenizer.special_tokens_map.values())

    # plot:
    fig, ax = plt.subplots(figsize=figsize)
    plot_token_vbars(ax,
        scores         = shap_values,
        tokens         = tokens,
        document_names = document_names,
        normalize      = normalize,
        skip_tokens    = special_tokens,
        cmap           = cmap
    )
    ax.set_title(f'SHAP Attribution - {aggregation.capitalize()} Aggregation', 
                fontsize=14, fontweight='bold')
    ax.set_ylabel('Normalized Attribution' if  normalize else 'Attribution')
    ax.legend()
    fig.tight_layout()

    if show: fig.show()
    else: return fig

def higlight_attribution_generator(explanation:GeneratorExplanationBase, document_names:Optional[List[str]]=None, *,
        threshold:float=.5,
        token_processor:Optional[Callable[[str],str]]=None,
        show:bool=True,
        cmap:str='tab10'
    ) -> str:
    """Highlights tokens in a text sequence where a single document contributes
    at least `threshold` of the total positive SHAP value at that token.

    Args:
        explanation (GeneratorExplanationBase): An object containing the necessary information for plotting.
        document_names (List[str]):             An optional list of names of the documents.
        threshold (float):                      Minimum attribution for highlighting in the intervall `[0., 1.)` (default: `0.5`).
        token_processor ((str) -> str):         An optional function applied to each token before printing.
        cmap (str):                             The name of a matplotlib colormap used for highlighting.
        show (bool):                            If `True` shows the plot directly, if `False` the plot is returned instead (default: `True`).

    Returns:
        A HTML-formatted string with spans highlighting dominant SHAP regions if `show == False`.
    """

    # get attribution scores:
    shap_values = explanation.get_shapley_values(None, 'token')

    # get indices of the actual content:
    # (excluding chat template specific tokens) 
    response = decode_chat_template(
        explanation.gen_tokens,
        explanation.model_name_or_path,
        return_indices=True
    )[0]

    # create role:
    role = explanation.gen_tokens[response['role']]
    if token_processor is not None:
        role = [token_processor(t) for t in role]
    role = ''.join(role).strip().capitalize()

    # get special tokens:
    special_tokens = set(explanation.tokenizer.special_tokens_map.values())

    # create query html:
    html_query = highlight_dominant_passages(
        scores          = shap_values['query'].mean(axis=1, keepdims=True).T,
        tokens          = explanation.qry_tokens,
        title           = 'Query',
        document_names  = document_names,
        threshold       = threshold,
        skip_tokens     = special_tokens,
        legend          = False,
        cmap            = cmap
    )

    # create context html:
    html_response, html_legend = highlight_dominant_passages(
        scores          = shap_values['context'][:,response['content']],
        tokens          = explanation.gen_tokens[response['content']],
        title           = role,
        document_names  = document_names,
        threshold       = threshold,
        skip_tokens     = special_tokens,
        token_processor = token_processor,
        cmap            = cmap
    )

    # Build, display, and return final HTML:
    html_str = (
        '<!DOCTYPE html>\n' +
        '<html>\n' +
        '<head>\n' +
        '<title>Generated Text</title>\n' +
        '</head>\n' +

        '<body>\n' +
        '<table>\n' +
        html_query +
        html_response +
        html_legend +
        '</table>\n' +
        '</body>\n' +

        '</html>'
    )

    if show: display(HTML(html_str))
    else: return html_str

def plot_attribution_summary_generator(explanation:GeneratorExplanationBase, document_names:Optional[List[str]]=None, *,
        aggregation:Literal['token', 'sequence', 'bow', 'nucleus']='token',
        absolute:bool=False,
        normalize:bool=True,
        figsize: Tuple[int, int] = (12, 6),
        cmap:str='tab10',
        show:bool=True
    ) -> Union[Figure, None]:
    """Create a summary plot showing mean absolute SHAP values per document.

    Args:
        explanation (GeneratorExplanationBase): An object containing the necessary information for plotting.
        document_names (List[str]):             An optional list of names of the documents.
        aggregation (str):                      Aggregation method for probabilities (default: `'token'`).
        absolute (bool):                        Bar plot tf `True`, else waterfall plot (default=`False`).
        normalize (bool):                       If `True`, normalizes the SHAP values across tokens so that the sum of absolute values equals 1 (default=`True`).
        figsize ((int, int)):                   The size of the figure.
        cmap (str):                             The name of a matplotlib colormap used for highlighting.
        show (bool):                            If `True` shows the plot directly, if `False` the plot is returned instead (default: `True`).

    Returns:
        `matplotlib.figure.Figure` object if `show == False`.
    """

    # get attribution scores:
    shap_values = explanation.get_shapley_values('context', aggregation)

    # fallback for document names:
    if document_names is None:
        document_names = [f'Document {i+1:d}' for i in range(len(shap_values))]
    elif len(document_names) != len(shap_values): raise ValueError('`len(document_names)` does not match the number of documents!')
    
    # create figure:
    fig, ax = plt.subplots(figsize=figsize)

    if absolute:
        # calculate mean absolute attributions per document:
        shap_values = np.mean(np.abs(shap_values), axis=1)
        plot_document_vbars(ax,
            scores         = shap_values,
            document_names = document_names,
            normalize      = normalize,
            cmap           = cmap 
        )
        ax.set_ylabel('Mean Absolute SHAP Attribution')
        
    else:
        # calculate mean attributions per document
        shap_values = np.mean(shap_values, axis=1)
        plot_waterfall(ax,
            scores    = shap_values,
            x_labels  = document_names,
            normalize = normalize,
            cmap      = cmap 
        )
        ax.set_ylabel('Cumulative Attribution')

    # title and x-label:
    ax.set_xlabel('Documents')
    ax.set_title(f'SHAP Attribution Summary - {aggregation.capitalize()} Aggregation', 
                fontsize=14, fontweight='bold')

    fig.tight_layout()
    if show: fig.show()
    else: return fig

def visualize_attribution_generator(explanation:GeneratorExplanationBase, document_names:Optional[List[str]]=None, *,
        aggregation:Literal['token', 'sequence', 'bow', 'nucleus']='token',
        cmap:str='tab10',
        show:bool=True,
        **kwargs
    ) -> Union[Figure, str, None]:
    """Visualizes Shapley attribution values while automatically choosing a fitting plot for the chosen `aggregation`.

    Args:
        explanation (RetrieverExplanationBase): An object containing the necessary information for plotting.
        document_names (List[str]):             An optional list of names of the documents.
        aggregation (str):                      Aggregation method for probabilities (default: `'token'`).
        cmap (str):                             The name of a matplotlib colormap used for highlighting.
        show (bool):                            If `True` shows the plot directly, if `False` the plot is returned instead (default: `True`).

    Returns:
        The illustration object if `show == False`.
    """

    if aggregation == 'token':
        return higlight_attribution_generator(
            explanation    = explanation,
            document_names = document_names,
            cmap           = cmap,
            show           = show,
            **kwargs
        )

    elif aggregation == 'sequence':
        return plot_attribution_summary_generator(
            explanation    = explanation,
            document_names = document_names,
            aggregation    = aggregation,
            normalize      = kwargs.get('normalize', True),
            cmap           = cmap,
            show           = show,
            **kwargs
        )

    else:
        return plot_attribution_generator(
            explanation    = explanation,
            document_names = document_names,
            aggregation    = aggregation,
            cmap           = cmap,
            show           = show,
            **kwargs
        )

#====================================================================#
# Plots for retrieval + generation:                                  #
#====================================================================#

from resources.rag import ExplainableAutoModelForRAG

def plot_document_importance_rag(explanation:ExplainableAutoModelForRAG, document_names:Optional[List[str]]=None, *,
    figsize: Tuple[int, int] = (12, 6),
    mean_color:str='gray',
    cmap:str='tab10',
    show:bool=True
) -> Union[Figure, None]:
    """Plot document importance.

    Args:
        explanation (ExplainableAutoModelForRA): An object containing the necessary information for plotting.
        document_names (List[str]):              An optional list of names of the documents.
        figsize ((int, int)):                    The size of the figure.
        mean_color (str):                        The color used for the mean values.
        cmap (str):                              The name of a matplotlib colormap used for highlighting.
        show (bool):                             If `True` shows the plot directly, if `False` the plot is returned instead (default: `True`).

    Returns:
        `matplotlib.figure.Figure` object if `show == False`.
    """

    # get y-values:
    y_ret = explanation.retriever_document_importance
    y_gen = explanation.generator_document_importance
    y_avg = (y_ret + y_gen) / 2.

    # get x-values:
    num_documents = y_avg.shape[0]
    x = np.arange(num_documents)

    # get colormap:
    colors = cm.get_cmap(cmap)

    # fallback for document names:
    if document_names is None:
        document_names = [f'Document {i+1:d}' for i in range(num_documents)]
    elif len(document_names) != num_documents: raise ValueError('`len(document_names)` does not match the number of documents!')

    fig, ax = plt.subplots(figsize=figsize)

    # draw average shade:
    ax.bar(x,    y_avg, .8, color=mean_color, alpha=.5)

    # draw bars:
    ax.bar(x-.2, y_ret, .4, label='Retriever', color=colors(0))
    ax.bar(x+.2, y_gen, .4, label='Generator', color=colors(1))

    # draw average:
    ax.hlines(y_avg, x-.4, x+.4, color=mean_color, label='average')

    # draw texts:
    y_top = np.maximum(y_ret, y_gen)
    for i, s in enumerate(document_names):
        ax.text(x[i], max(y_top[i], 0)+.01, s, ha='center', va='bottom')

    # clean frame (keep only left spine):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    if ax.get_ylim()[0] < 0.:
        ax.spines['bottom'].set_visible(False)

        # add horizontal grid line at zero:
        ax.axhline(y=0, color='black', linewidth=0.5)

    # add horizontal grid lines at each tick:
    for tick_loc in ax.get_yticks(minor=False):
        ax.axhline(y=tick_loc, color='lightgray', linewidth=0.5, zorder=0)

    ax.set_title(f'Overall Document Importance', 
                fontsize=14, fontweight='bold')
    ax.set_xticks([])
    ax.set_ylabel('Normalized Importance')
    ax.legend()

    fig.tight_layout()

    if show: fig.show()
    else: return fig