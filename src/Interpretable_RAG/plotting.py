import numpy as np
import pandas as pd
import spacy
import pickle
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from html import escape
from IPython.display import display, HTML
from src.Interpretable_RAG.utils import match_token_attributions

from .utils import decode_chat_template, nucleus_sample_tokens

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.colors import Colormap
from numpy.typing import NDArray
from typing import List, Literal, Tuple, Callable, Optional, Union

#====================================================================#
# Custom colormao specifications:                                    #
#====================================================================#

cmap = mcolors.ListedColormap(["#00dddd", "#dd00dd"], name='ragbin')
mpl.colormaps.register(cmap=cmap, force=True)

#====================================================================#
# General plotting functions:                                        #
#====================================================================#

def plot_token_vbars(ax:Axes, scores:NDArray[np.float64], tokens:List[str], document_names:Optional[List[str]]=None, *,
        normalize:bool=False,
        skip_tokens:List[str]=[],
        cmap:str='tab10',
        **kwargs
    ) -> None:
    """Creates a vertical bar plot for attribution scores of tokens.

    Args:
        ax (matplotlib.axes.Axes):   `matplotlib.axes.Axes` object.
        scores (NDArray[np.float64]): A 2D array of attribution scores with shape `(len(documents), len(tokens))`,
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

def html_legend_discrete(names:List[str], cmap:Colormap, vals:Optional[List[float]]=None) -> str:
    n = len(names)
    
    # prepare color map:
    rgb_colors = np.array([mcolors.to_rgb(cmap(i)) for i in range(n)])*255.
    
    # format texts:
    if vals is None:
        texts = [f'<i>{n}</i>' for n in names]
    else:
        texts = [f'<i>{n}</i><br><small>({v * 100.:.0f}%)</small>' for n, v in zip(names, vals, strict=True)]

    # return legend:
    return (
        '<small style="float:left;">\n' +
        '   <div style="line-height:1">' +
                '<div style="line-height: 2; text-align: center; padding:3px; margin:3px; float:left;"><i>Legend:</i></div>' +
                ''.join([f'<div style="background-color:#{int(r):02x}{int(g):02x}{int(b):02x}; text-align: center; padding:3px; margin:3px; border-radius:3px; float:left;">{texts[i]}</div>' for i, (r, g, b) in enumerate(rgb_colors)]) +
            '</div>\n' +
        '</small>\n'
    )

def html_legend_continuous(names:List[str], cmap:Colormap, vals:Optional[List[float]]=None) -> str:
    # prepare color map:
    hex_left = mcolors.to_hex(cmap(0.0))
    hex_right = mcolors.to_hex(cmap(1.0))

    # format texts:
    if vals is None:
        texts = [f'<i>{n}</i>' for n in names]
    else:
        texts = [f'<div><i>{n}</i><br><small>({v * 100.:.0f}%)</small></div>' for n, v in zip(names, vals, strict=True)]

    # return a simple horizontal colorbar:
    return (
        '<small style="float:left;">\n' +
        '   <div style="line-height: 1;">' +
                '<div style="text-align: center; padding:3px; margin:3px; float:left;"><i>Legend:</i></div>' +
                f'<div style="background: linear-gradient(90deg, {hex_left}, {hex_right}); text-align: center; padding:3px; margin:3px; border-radius:3px; float:left;">' +
                    f'{"&nbsp;"*5}&#8594;{"&nbsp;"*5}'.join(texts) +
                '</div>' +
            '</div>\n' +
        '</small>\n'
    )

def highlight_dominant_passages(scores:NDArray[np.float64], tokens:List[str], title:str, labels:List[str]=[], *,
        threshold:float=0.0,
        max_score:Optional[float]=None,
        skip_tokens:List[str]=[],
        token_processor:Optional[Callable[[str],str]]=None,
        color_mode:Literal['winner_takes_it_all', 'average']='winner_takes_it_all',
        legend:bool=True,
        cmap:str='tab10'
    ) -> Tuple[str, str]:
    """Highlights tokens in a text sequence where a single document contributes
    at least `threshold` of the total positive attribution scores at that token.

    Args:
        scores (NDArray[np.float64]):   A 1D or 2D array of attribution scores with shape `([len(documents),] len(tokens))`,
                                        where each row represents a document and each column corresponds to a token's attribution score.
        tokens (List[str]):             A list of token strings corresponding to the scores.
                                        Length must match the number of columns in `scores`.
        labels (List[str]):             An optional list of names of the documents.
        title (str):                    The title of the produced html table row.
        threshold (float):              Minimum attribution for highlighting in the intervall `[0., 1.]` (default: `0.0`).
        max_score (float):              Optional Maximum attribution for highlighting (default: `None`).
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

    # check dimenisons of `scores` input:
    num_dims = scores.ndim
    if   num_dims == 1: scores = np.stack([scores, -scores])
    elif num_dims != 2: raise ValueError(f"Parameter `scores` must have 2 dimensions abut has {num_dims:d}.")

    # extract only on positive contributions:
    scores_pos = np.maximum(scores, 0)
    num_docs, num_tokens = scores_pos.shape

    # check color mode parameter:
    if color_mode == 'winner_takes_it_all':
        # set non maximal scores to nan:
        token_docs = scores_pos == scores_pos.max(axis=0, keepdims=True)
        scores_pos[~token_docs] = np.nan

    elif color_mode != 'average':
        raise ValueError(f'Unknown `color_mode` "{color_mode}"')

    # compute document influence per token:
    token_totals = np.nansum(scores_pos, initial=0., axis=0, keepdims=True)
    token_totals = np.maximum(token_totals, 1e-9) # make sure that token_sum > 0
    token_vals = scores_pos / token_totals
    token_vals = np.maximum((token_vals - threshold) / (1. - threshold), 0.)
    
    # compute alpha values:
    alphas = np.nanmean(scores_pos, axis=0)
    alphas /= alphas.max() if max_score == None else max_score
    alphas *= 255.

    # prepare color map
    cmap = cm.get_cmap(cmap)
    rgb_colors = np.array([mcolors.to_rgb(cmap(i)) for i in range(num_docs)])*255.

    # build highlighed HTML:
    html_tokens = []
    for tok, val, alpha in zip(tokens, token_vals.T, alphas):
        if tok in skip_tokens: continue

        if token_processor is not None:
            tok = token_processor(tok)

        if not np.isnan(val).all():
            r,g,b = np.nanmean(np.stack([c*v for c,v in zip(rgb_colors, val)]), axis=0).astype(int)
            html_tokens.append(
                f'<span style="background-color:#{r:02x}{g:02x}{b:02x}{int(alpha):02x}; padding:0px; border-radius:3px;">' +
                escape(tok) +
                '</span>'
            )

        else: html_tokens.append(escape(tok))

    html_legend = ''
    if legend:
        if (color_mode == 'average') or (num_dims == 1):
            # add a simple horizontal colorbar:
            html_legend = html_legend_continuous(labels, cmap)

        elif color_mode == 'winner_takes_it_all':
            # extract mean absolute document contributions:
            vals = scores.mean(axis=1)
            vals /= np.abs(vals).sum()

            # build legend:
            html_legend = html_legend_discrete(labels, cmap, vals)

    html_text = (
        '<tr style="border-top: 1px solid">\n' +
        '   <td style="text-align:right; vertical-align:top">\n' +
        '       <b style="line-height:2">' + title + ':</b>\n' +
        '   </td>\n' +
        '   <td style="text-align:left; vertical-align:top">\n' +
        '       <div style="line-height:2">' + ''.join(html_tokens) + '</div>\n' + html_legend + 
        '   </td>\n' +
        '</tr>\n'
    )
    
    return html_text

def plot_document_vbars(ax:Axes, scores:NDArray[np.float64], document_names:Optional[List[str]]=None, *,
        normalize:bool=False,
        cmap:str='tab10',
        **kwargs
    ) -> None:
    """Creates a vertical bar plot for attribution scores of documents.

    Args:
        ax (matplotlib.axes.Axes):   `matplotlib.axes.Axes` object.
        scores (NDArray[np.float64]): A 1D array of attribution scores with shape `(len(documents),)`,
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

def plot_waterfall(ax:Axes, scores:NDArray[np.float64], x_labels:Optional[List[str]], *,
        base_value:float=0.0,
        normalize:bool=False,
        cmap:str='tab10',
        **kwargs
    ) -> None:
    """Create a waterfall plot showing cumulative scores.
    
    Args:
        ax (matplotlib.axes.Axes):   `matplotlib.axes.Axes` object.
        scores (NDArray[np.float64]): A 1D array of attribution scores with shape `(len(documents),)`,
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

from .retrieval import RetrieverExplanationBase

def plot_importance_retriever(explanation:RetrieverExplanationBase, document_names:Optional[List[str]]=None, *,
        method:Literal['grad', 'gradIn', 'aGrad', 'intGrad']='intGrad',
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
    if   method == 'grad':    scores = {key:[doc.mean(axis=-1) for doc in docs] for key, docs in explanation.grad(**kwargs).items()}
    elif method == 'aGrad':   scores = {key:[doc.mean(axis=0) for doc in docs] for key, docs in explanation.aGrad(**kwargs).items()}
    else:
        # try to call a method on the explanation object by name
        method_fn = getattr(explanation, method, None)
        if callable(method_fn): scores = method_fn(**kwargs)
        else: raise ValueError(f"Explanation has no callable method named '{method}'")

    # compute absolute:
    if absolute: scores = {key:[np.abs(doc) for doc in docs] for key, docs in scores.items()}

    # fallback for document names:
    if document_names is None:
        document_names = [f'Document {i+1:d}' for i in range(len(scores['context']))]
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
        method:Literal['grad', 'gradIn', 'aGrad', 'intGrad']='intGrad',
        threshold:float=0.0,
        token_processor:Optional[Callable[[str],str]]=None,
        cmap:str='ragbin',
        show:bool=True,
        **kwargs
    ) -> Union[str,None]:
    """Highlights tokens in a text sequence that are important for retrieving the document.

    Args:
        explanation (RetrieverExplanationBase):   An object containing the necessary information for plotting.
        document_names (List[str], optional):     An optional list of names of the documents.
        method (str, optional):                   The method for calculating token importance.
        threshold (float, optional):              Minimum importance for highlighting in the intervall `[0., 1.)` (default: `0.`).
        token_processor ((str) -> str, optional): An optional function applied to each token before printing.
        cmap (str, optional):                     The name of a matplotlib colormap used for highlighting.
        show (bool, optional):                    If `True` shows the plot directly, if `False` the plot is returned instead (default: `True`).

    Returns:
        A HTML-formatted string with spans highlighting important tokens if `show == False`.
    """
    # get scores:
    if   method == 'grad':    scores = {key:[doc.mean(axis=-1) for doc in docs] for key, docs in explanation.grad(**kwargs).items()}
    elif method == 'aGrad':   scores = {key:[doc.mean(axis=0) for doc in docs] for key, docs in explanation.aGrad(**kwargs).items()}
    else:
        # try to call a method on the explanation object by name
        method_fn = getattr(explanation, method, None)
        if callable(method_fn): scores = method_fn(**kwargs)
        else: raise ValueError(f"Explanation has no callable method named '{method}'")

    # compute absolute:
    scores = {key:[np.abs(doc) for doc in docs] for key, docs in scores.items()}

    # fallback for document names:
    if document_names is None:
        document_names = [f'Document&nbsp;{i+1:d}' for i in range(len(scores['context']))]
    elif len(document_names) != len(scores['context']): raise ValueError('`len(document_names)` does not match the number of documents!')

    # get special tokens:
    special_tokens = set(explanation.tokenizer.special_tokens_map.values())

    # plot query:
    html = highlight_dominant_passages(
        scores          = scores['query'][0].numpy(), 
        tokens          = explanation.in_tokens['query'][0],
        title           = 'Query',
        labels          = ['+', '-'],
        threshold       = threshold,
        skip_tokens     = special_tokens,
        token_processor = token_processor,
        legend          = True,
        cmap            = cmap
    )

    # plot contexts:
    for i, s in enumerate(scores['context']):
        html += highlight_dominant_passages(
            scores          = s.numpy(), 
            tokens          = explanation.in_tokens['context'][i],
            title           = document_names[i],
            labels          = ['+', '-'],
            threshold       = threshold,
            max_score       = np.concatenate(scores['context']).max(),
            skip_tokens     = special_tokens,
            token_processor = token_processor,
            legend          = True,
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
        method:Literal['grad', 'gradIn', 'aGrad', 'intGrad']='intGrad',
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
    if   method == 'grad':    scores = {key:[doc.mean(axis=-1) for doc in docs] for key, docs in explanation.grad(**kwargs).items()}
    elif method == 'aGrad':   scores = {key:[doc.mean(axis=0) for doc in docs] for key, docs in explanation.aGrad(**kwargs).items()}
    else:
        # try to call a method on the explanation object by name
        method_fn = getattr(explanation, method, None)
        if callable(method_fn): scores = method_fn(**kwargs)
        else: raise ValueError(f"Explanation has no callable method named '{method}'")

    # compute absolute:
    scores = {key:[np.abs(doc) for doc in docs] for key, docs in scores.items()}
    
    # fallback for document names:
    if document_names is None:
        document_names = [f'Document {i+1:d}' for i in range(len(scores['context']))]
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
        method:Literal['grad', 'gradIn', 'aGrad', 'intGrad']='intGrad',
        cmap:str='ragbin',
        show:bool=True,
        **kwargs
    ) -> Union[Figure, str, None]:
    """Visualizes token importance while automatically choosing a fitting plot for the chosen `method`.

    Args:
        explanation (RetrieverExplanationBase): An object containing the necessary information for plotting.
        document_names (List[str]):             An optional list of names of the documents.
        method (str):                           The method for calculating token importance.
        cmap (str, optional):                   The name of a matplotlib colormap used for highlighting.
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

from .generation import GeneratorExplanationBase

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
        query_cmap:str='ragbin',
        document_cmap:str='tab10',
        show:bool=True
    ) -> str:
    """Highlights tokens in a text sequence where a single document contributes
    at least `threshold` of the total positive SHAP value at that token.

    Args:
        explanation (GeneratorExplanationBase):   An object containing the necessary information for plotting.
        document_names (List[str], optional):     An optional list of names of the documents.
        threshold (float, optional):              Minimum attribution for highlighting in the intervall `[0., 1.)` (default: `0.5`).
        token_processor ((str) -> str, optional): An optional function applied to each token before printing.
        query_cmap (str, optional):               The name of a matplotlib colormap used for highlighting of the query attributions.
        document_cmap (str, optional):            The name of a matplotlib colormap used for highlighting of different documents.
        show (bool, optional):                    If `True` shows the plot directly, if `False` the plot is returned instead (default: `True`).

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
    role = response['role']
    if token_processor is not None:
        role = [token_processor(t) for t in role]
    role = ''.join(role).strip().capitalize()

    # fallback for document names:
    if document_names is None:
        document_names = [f'Document&nbsp;{i+1:d}' for i in range(len(shap_values['context']))]
    elif len(document_names) != len(shap_values['context']): raise ValueError('`len(document_names)` does not match the number of documents!')

    # get special tokens:
    special_tokens = set(explanation.tokenizer.special_tokens_map.values())

    # create query html:
    html_query = highlight_dominant_passages(
        scores          = shap_values['query'].mean(axis=1),
        tokens          = explanation.qry_tokens,
        title           = 'Query',
        labels          = ['+', '-'],
        threshold       = threshold,
        skip_tokens     = special_tokens,
        legend          = True,
        cmap            = query_cmap
    )

    # create response html:
    html_response = highlight_dominant_passages(
        scores          = shap_values['context'][:,response['content']],
        tokens          = explanation.gen_tokens[response['content']],
        title           = role,
        labels          = document_names,
        threshold       = threshold,
        skip_tokens     = special_tokens,
        token_processor = token_processor,
        legend          = True,
        cmap            = document_cmap
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
        primary_cmap:str='tab10',
        secondary_cmap:str='ragbin',
        show:bool=True,
        **kwargs
    ) -> Union[Figure, str, None]:
    """Visualizes Shapley attribution values while automatically choosing a fitting plot for the chosen `aggregation`.

    Args:
        explanation (RetrieverExplanationBase): An object containing the necessary information for plotting.
        document_names (List[str]):             An optional list of names of the documents.
        aggregation (str):                      Aggregation method for probabilities (default: `'token'`).
        primary_cmap (str, optional):           The name of a matplotlib colormap used for highlighting of different documents.
        secondary_cmap (str, optional):         The name of a matplotlib colormap used for highlighting of the query attributions.
        show (bool):                            If `True` shows the plot directly, if `False` the plot is returned instead (default: `True`).

    Returns:
        The illustration object if `show == False`.
    """

    if aggregation == 'token':
        return higlight_attribution_generator(
            explanation    = explanation,
            document_names = document_names,
            query_cmap     = secondary_cmap,
            document_cmap  = primary_cmap,
            show           = show,
            **kwargs
        )

    elif aggregation == 'sequence':
        return plot_attribution_summary_generator(
            explanation    = explanation,
            document_names = document_names,
            aggregation    = aggregation,
            normalize      = kwargs.get('normalize', True),
            cmap           = primary_cmap,
            show           = show,
            **kwargs
        )

    else:
        return plot_attribution_generator(
            explanation    = explanation,
            document_names = document_names,
            aggregation    = aggregation,
            cmap           = primary_cmap,
            show           = show,
            **kwargs
        )

#====================================================================#
# Plots for retrieval + generation:                                  #
#====================================================================#

from .rag import ExplainableAutoModelForRAG

def plot_document_importance_rag(explanation:ExplainableAutoModelForRAG, document_names:Optional[List[str]]=None, *,
    figsize: Tuple[int, int] = (12, 6),
    mean_color:str='gray',
    cmap:str='tab10',
    show:bool=True
) -> Union[Figure, None]:
    """Plot document importance.

    Args:
        explanation (ExplainableAutoModelForRAG): An object containing the necessary information for plotting.
        document_names (List[str]):               An optional list of names of the documents.
        figsize ((int, int)):                     The size of the figure.
        mean_color (str):                         The color used for the mean values.
        cmap (str):                               The name of a matplotlib colormap used for highlighting.
        show (bool):                              If `True` shows the plot directly, if `False` the plot is returned instead (default: `True`).

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

def higlight_importance_rag(explanation:ExplainableAutoModelForRAG, document_names:Optional[List[str]]=None, *,
        threshold:float=.5,
        retriever_token_processor:Optional[Callable[[str],str]]=None,
        generator_token_processor:Optional[Callable[[str],str]]=None,
        retriever_method:Literal['grad', 'gradIn', 'aGrad', 'intGrad']='intGrad',
        query_cmap:str='ragbin',
        document_cmap:str='tab10',
        show:bool=True,
        **kwargs
    ) -> str:
    """Highlights tokens in a text sequence where a single document contributes
    at least `threshold` of the total positive SHAP value at that token.

    Args:
        explanation (ExplainableAutoModelForRAG):
            An object containing the necessary information for plotting.
        document_names (List[str]):
            An optional list of names of the documents.
        threshold (float, optional):
            Minimum attribution for highlighting in the intervall `[0., 1.)` (default: `0.5`).
        retriever_token_processor ((str) -> str, optional):
            Optional function applied to each retriever token before printing.
        generator_token_processor ((str) -> str, optional):
            Optional function applied to each generator token before printing.
        retriever_method (str, optional):
            The method for calculating retriever token importance.
        query_cmap (str, optional):
            The name of a matplotlib colormap used for highlighting of the query attributions.
        document_cmap (str, optional):
            The name of a matplotlib colormap used for highlighting of different documents.
        show (bool, optional):
            If `True` shows the plot directly, if `False` the plot is returned instead (default: `True`).

    Returns:
        A HTML-formatted string with spans highlighting dominant SHAP regions if `show == False`.
    """

    # get retriever scores:
    if   retriever_method == 'grad':    retriever_attr = {key:[doc.mean(axis=-1) for doc in docs] for key, docs in explanation.retriever.grad(**kwargs).items()}
    elif retriever_method == 'aGrad':   retriever_attr = {key:[doc.mean(axis=0) for doc in docs] for key, docs in explanation.retriever.aGrad(**kwargs).items()}
    else:
        # try to call a method on the explanation object by name
        retriever_method_fn = getattr(explanation.retriever, retriever_method, None)
        if callable(retriever_method_fn): retriever_attr = retriever_method_fn(**kwargs)
        else: raise ValueError(f"Retriever has no callable method named '{retriever_method}'")

    # compute absolute:
    retriever_attr = {key:[np.abs(doc) for doc in docs] for key, docs in retriever_attr.items()}

    # get generator attribution scores:
    generator_attr = explanation.generator.get_shapley_values(None, 'token')

    # get special tokens:
    retriever_special_tokens = set(explanation.retriever.tokenizer.special_tokens_map.values())
    generator_special_tokens = set(explanation.generator.tokenizer.special_tokens_map.values())

    # fallback for document names:
    if document_names is None:
        document_names = [f'Document&nbsp;{i+1:d}' for i in range(len(retriever_attr['context']))]
    elif len(document_names) != len(retriever_attr['context']): raise ValueError('`len(document_names)` does not match the number of documents!')

    # get indices of the actual content:
    # (excluding chat template specific tokens) 
    response = decode_chat_template(
        explanation.generator.gen_tokens,
        explanation.generator.model_name_or_path,
        return_indices=True
    )[0]

    # create role:
    role = response['role']
    if generator_token_processor is not None:
        role = [generator_token_processor(t) for t in role]
    role = ''.join(role).strip().capitalize()

    # create query html:
    cmb_qry, ret_qry_attr, gen_qry_attr = match_token_attributions(
        retriever_attr['query'][0].numpy(),   explanation.retriever.in_tokens['query'][0],
        generator_attr['query'].mean(axis=1), explanation.generator.qry_tokens,
        ret_token_processor = retriever_token_processor,
        gen_token_processor = generator_token_processor
    )
    html = highlight_dominant_passages(
        scores          = np.stack([ret_qry_attr, gen_qry_attr]),
        tokens          = cmb_qry,
        title           = 'Query',
        labels          = ['Retriever', 'Generator'],
        color_mode      = 'average',
        legend          = True,
        cmap            = query_cmap
    )

    # plot contexts:
    for i, s in enumerate(retriever_attr['context']):
        html += highlight_dominant_passages(
            scores          = s.numpy(), 
            tokens          = explanation.retriever.in_tokens['context'][i],
            title           = document_names[i],
            labels          = ['+', '-'],
            threshold       = threshold,
            max_score       = np.concatenate(retriever_attr['context']).max(),
            skip_tokens     = retriever_special_tokens,
            token_processor = retriever_token_processor,
            legend          = True,
            cmap            = query_cmap
        )

    # create response html:
    html_response = highlight_dominant_passages(
        scores          = generator_attr['context'][:,response['content']],
        tokens          = explanation.generator.gen_tokens[response['content']],
        title           = role,
        labels          = document_names,
        threshold       = threshold,
        skip_tokens     = generator_special_tokens,
        token_processor = generator_token_processor,
        legend          = True,
        cmap            = document_cmap
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
        html +
        html_response +
        '</table>\n' +
        '</body>\n' +

        '</html>'
    )

    if show: display(HTML(html_str))
    else: return html_str

#====================================================================#
# (New) Plots for Part-of-Speech (POS) Analysis                      #
#====================================================================#

# Note: These functions require spacy and a model. 
# You may need to run:
# pip install spacy pandas
# python -m spacy download en_core_web_sm

# load small english model
nlp = spacy.load("en_core_web_sm")

def tokens_to_pos(tokens, detok="auto"):
    """
    Convert subword tokens into words, tag them with POS,
    and map back to original token indices.
    """
    # simple detok heuristic (works ok with GPT-style tokens like 'Ġhello')
    words = []
    mapping = []  # map each original token index -> word index
    current_word = ""
    word_idx = -1

    for i, tok in enumerate(tokens):
        # roberta/gpt2 style
        if tok.startswith("Ġ") or tok.startswith(" "):
            # start new word
            words.append(tok[1:])
            word_idx = len(words)-1
            mapping.append(word_idx)
        elif tok.startswith("##"):  # bert/wordpiece
            # continuation
            if words:
                words[-1] += tok[2:]
                mapping.append(word_idx)
            else:
                words.append(tok[2:])
                word_idx = len(words)-1
                mapping.append(word_idx)
        else:
            if words:
                words[-1] += tok
                mapping.append(word_idx)
            else:
                words.append(tok)
                word_idx = len(words)-1
                mapping.append(word_idx)

    text = " ".join(words)
    doc = nlp(text)

    # now assign POS back to each token
    pos_per_word = [token.pos_ for token in doc]  # e.g. "NOUN", "VERB", etc.
    pos_per_token = [pos_per_word[m] if m < len(pos_per_word) else "X" for m in mapping]

    return pos_per_token, pos_per_word, mapping, words

def shap_by_pos(shap_docs_tokens, pos_tags, doc_names=None, absolute=True):
    """
    Aggregate contributions per POS category across tokens.
    """
    S = np.asarray(shap_docs_tokens)   # (D, T)
    if absolute:
        S = np.abs(S)

    D, T = S.shape
    assert len(pos_tags) == T

    if doc_names is None:
        doc_names = [f"Doc {i+1}" for i in range(D)]

    # unique POS categories
    pos_set = sorted(set(pos_tags))
    agg = {pos: S[:, np.array(pos_tags) == pos].mean(axis=1) for pos in pos_set}

    df = pd.DataFrame(agg, index=doc_names).T
    return df

def plot_shap_by_pos(df, cmap="tab10", figsize=(10,6),dpi=300, normalize=True ,save_path=None,show=True):
    """
    df: POS x Documents DataFrame from shap_by_pos()
    """
    if normalize:
        df = df.div(df.sum(axis=1), axis=0)  # normalize per POS
    fig, ax = plt.subplots(figsize=figsize, dpi = dpi) # Create figure and axes

    # Generate colors consistently with other plot functions
    num_docs = df.shape[1]
    cmap_obj = cm.get_cmap(cmap)
    colors = [cmap_obj(i) for i in range(num_docs)]

    # Pass the explicit list of colors to the plot function
    df.plot(kind="bar", stacked=True, color=colors, ax=ax)

    ax.set_ylabel("Normalized contribution" if normalize else "Mean contribution")
    ax.set_xlabel("POS tag")
    ax.set_title("Document impact by Part-of-Speech")
    plt.xticks(rotation=45, ha="right")
    plt.legend(title="Documents", bbox_to_anchor=(1.05, 1), loc="upper left")
    fig.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)

    if show:
        plt.show()

    return fig

#====================================================================#
# (New) Plots for Global Importance Analysis                         #
#====================================================================#

def safe_load_pickle(file_path):
    """
    Safely loads a pickle file, handling potential exceptions.

    Args:
        file_path (str): The path to the .pkl file to load.

    Returns:
        dict: The object loaded from the pickle file, or None if an error occurs.
    """
    
    try:
        with open(file_path, 'rb') as f:
            obj = pickle.load(f)
        return obj
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return None

def process_global_importance(pickle_files: list, safe_load_pickle_func) -> pd.DataFrame:
    """
    Processes a list of pickle files to calculate the average importance of each document.

    Args:
        pickle_files (list): List of paths to the .pkl files.
        safe_load_pickle_func (function): The function to safely load the pickle files.

    Returns:
        pd.DataFrame: A DataFrame where each column is a document and each row
                      represents the average importance of that document for a single query.
    """
    all_doc_importances = []

    for file_path in pickle_files:
        try:
            obj = safe_load_pickle_func(file_path)
            if "shapley_values_token" in obj and "context" in obj["shapley_values_token"]:
                S = np.asarray(obj["shapley_values_token"]["context"])
                # Calculate the average (absolute) importance for each document in this file
                mean_importance_per_doc = np.abs(S).mean(axis=1)
                all_doc_importances.append(mean_importance_per_doc)
        except Exception:
            # Ignore files that cannot be loaded or do not have the necessary data
            continue
    
    if not all_doc_importances:
        return pd.DataFrame()

    # Determine the maximum number of documents found
    max_docs = max(len(row) for row in all_doc_importances)
    doc_names = [f"Doc {i+1}" for i in range(max_docs)]

    # Create the DataFrame
    df = pd.DataFrame(all_doc_importances, columns=doc_names[:len(all_doc_importances[0])])
    return df

def plot_global_importance_distribution(df: pd.DataFrame, cmap="tab10", figsize=(12, 7), dpi=300):
    """
    Creates a box plot to visualize the importance distribution of each document.
    """
    if df.empty:
        return None

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    num_docs = df.shape[1]
    cmap_obj = cm.get_cmap(cmap)
    colors = [cmap_obj(i) for i in range(num_docs)]

    # Box plot
    box = ax.boxplot(df.values, patch_artist=True, labels=df.columns)
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Calculate and plot the average
    means = df.mean().values
    ax.scatter(range(1, num_docs + 1), means, marker='o', color='red', s=50, zorder=3, label='Global Mean')

    ax.set_title(f"Distribution of Document Importance over {len(df)} Queries")
    ax.set_ylabel("Mean Absolute Importance per Query")
    ax.set_xlabel("Source Documents")
    ax.grid(True, axis='y', linestyle='--', alpha=0.6)
    ax.legend()
    fig.tight_layout()
    
    return fig