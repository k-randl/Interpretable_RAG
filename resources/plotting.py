import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from html import escape
from IPython.display import display, HTML

from resources.utils import decode_chat_template

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
    if normalize: ax.set_ylim(-1, 1)

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
        cmap (str):                     The name of a matplotlib colormap used for highlighting.

    Returns:
        A tuple `(text, legend)`, where both `text` and `legend` are html `<tr>` elements.
    """

    # extract only on positive contributions:
    scores_pos = np.maximum(scores, 0)
    num_docs, num_tokens = scores_pos.shape

    if total == None: total = scores_pos.sum(axis=0) if num_docs > 1 else scores_pos.max()
    if np.all(total > threshold):
        token_docs = scores_pos.argmax(axis=0)
        token_vals = scores_pos[token_docs, np.arange(num_tokens)] / total
        token_vals = np.maximum((token_vals - threshold) / (1. - threshold), 0.)
        token_docs[token_vals == 0.] = -1

    else:
        token_vals = np.zeros(num_tokens, dtype=float)
        token_docs = -np.ones(num_tokens, dtype=int)

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

    # fallback for document names:
    if document_names is None:
        document_names = [f'Document {i+1:d}' for i in range(num_docs)]
    elif len(document_names) != num_docs: raise ValueError('`len(document_names)` does not match the number of documents!')

    # build legend:
    html_legend = (
        '<tr style="border-top: 1px solid">\n' +
        '   <td style="text-align:right; vertical-align:top">\n' +
        '       <i style="line-height:2">Legend:</i>\n' +
        '   </td>\n' +
        '   <td style="text-align:left; vertical-align:top">\n' +
        '       <div style="line-height:2">' +
                    ' '.join([f'<i style="background-color:{c}; padding:0px; border-radius:3px;">{document_names[i]}</i>' for i, c in enumerate(rgb_colors)]) +
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

def plot_waterfall(ax:Axes, scores:NDArray[np.float_], document_names:Optional[List[str]]=None, *,
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
        document_names (List[str]):  An optional list of names of the documents.
        base_value (float):          Base value to start the waterfall from (default: `0.0`).
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

    # calculate cumulative values:
    cumulative = np.cumsum(np.concatenate([[base_value], scores]))

    # get colormap:
    colors = cm.get_cmap(cmap, 3)

    # plot each contribution:
    for i, value in enumerate(scores):
        color = colors(2) if value > 0 else colors(0)
        ax.bar(i, value, bottom=cumulative[i], color=color, **kwargs)
        
        # add value label:
        ax.text(i, cumulative[i] + value/2, f'{value:.4f}', 
               ha='center', va='center', fontweight='bold', color='white')

    # plot final value:
    ax.bar(len(document_names), cumulative[-1], color=colors(1), label='Total', **kwargs)

    # connect bars with lines:
    for i in range(len(document_names)):
        ax.plot([i + 0.4, i + 1.6], [cumulative[i+1], cumulative[i+1]], 'k--', alpha=0.5)

    # tick and axis settings:
    ax.set_xticks(range(len(document_names) + 1), document_names + ['Final'], rotation=45, ha='right')
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

#====================================================================#
# Plots for retrieval:                                               #
#====================================================================#

from resources.retrieval import RetrieverExplanationBase

def plot_importance_retriever(explanation:RetrieverExplanationBase, *,
        method:Literal['grad', 'gradIn', 'aGrad']='gradIn',
        absolute:bool=True,
        figsize: Tuple[int, int] = (12, 6),
        cmap:str='tab10',
        show:bool=True,
        **kwargs
    ) -> Union[Figure, None]:
    # get scores:
    if   method == 'grad':   scores = {key:[doc.mean(axis=-1) for doc in docs] for key, docs in explanation.grad(**kwargs).items()}
    elif method == 'gradIn': scores = explanation.gradIn(**kwargs)
    elif method == 'aGrad':  scores = {key:[doc.mean(axis=0) for doc in docs] for key, docs in explanation.aGrad(**kwargs).items()}
    else: raise ValueError()

    # compute absolute:
    if absolute: scores = {key:[np.abs(doc) for doc in docs] for key, docs in scores.items()}

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
    axs[0].set_title('Query:')

    # plot contexts:
    for i in range(len(scores['context'])):
        plot_token_vbars(axs[i+1],
            scores      = scores['context'][i].numpy()[None,:], 
            tokens      = explanation.in_tokens['context'][i],
            skip_tokens = special_tokens,
            cmap        = cmap
        )
        axs[i+1].set_title(f'Document {i+1:d}:')

    fig.suptitle(method)
    fig.tight_layout()

    if show: fig.show()
    else: return fig

def higlight_importance_retriever(explanation:RetrieverExplanationBase, *,
        method:Literal['grad', 'gradIn', 'aGrad']='gradIn',
        threshold:float=0.0,
        token_processor:Optional[Callable[[str],str]]=None,
        cmap:str='tab10',
        **kwargs
    ) -> str:
    # get scores:
    if   method == 'grad':   scores = {key:[doc.mean(axis=-1) for doc in docs] for key, docs in explanation.grad(**kwargs).items()}
    elif method == 'gradIn': scores = explanation.gradIn(**kwargs)
    elif method == 'aGrad':  scores = {key:[doc.mean(axis=0) for doc in docs] for key, docs in explanation.aGrad(**kwargs).items()}
    else: raise ValueError()

    # compute absolute:
    scores = {key:[np.abs(doc) for doc in docs] for key, docs in scores.items()}

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
        cmap            = cmap
    )[0]

    # plot contexts:
    for i in range(len(scores['context'])):
        html += highlight_dominant_passages(
            scores          = scores['context'][i].numpy()[None,:], 
            tokens          = explanation.in_tokens['context'][i],
            title           = f'Document {i+1:d}',
            threshold       = threshold,
            total           = np.concatenate(scores['context']).max(),
            skip_tokens     = special_tokens,
            token_processor = token_processor,
            cmap            = cmap
        )[0]

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
    display(HTML(html_str))
    return html_str

def visualize_importance_retriever(explanation:RetrieverExplanationBase, *,
        method:Literal['grad', 'gradIn', 'aGrad']='gradIn',
        cmap:str='tab10',
        **kwargs
    ) -> None:

    higlight_importance_retriever(
        explanation = explanation,
        method      = method,
        cmap        = cmap,
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
    shap_values = explanation.get_shapley_values(aggregation)

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
    fig.suptitle(aggregation)
    fig.legend()
    fig.tight_layout()

    if show: fig.show()
    else: return fig

def higlight_attribution_generator(explanation:GeneratorExplanationBase, document_names:Optional[List[str]]=None, *,
        threshold:float=.5,
        token_processor:Optional[Callable[[str],str]]=None,
        cmap:str='tab10'
    ) -> str:
    """Highlights tokens in a text sequence where a single document contributes
    at least `threshold` of the total positive SHAP value at that token.

    Args:
        explanation (GeneratorExplanationBase): An object containing the necessary information for plotting.
        document_names (List[str]):             An optional list of names of the documents.
        threshold (float):                      Minimum attribution for highlighting in the intervall `[0., 1.]` (default: `0.5`).
        token_processor ((str) -> str):         An otional function applied to each token before printing.
        cmap (str):                             The name of a matplotlib colormap used for highlighting.

    Returns:
        An HTML-formatted string with spans highlighting dominant SHAP regions.
    """

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

    # create html:
    html_response, html_legend = highlight_dominant_passages(
        scores          = explanation.get_shapley_values('token')[:,response['content']],
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
        html_response +
        html_legend +
        '</table>\n' +
        '</body>\n' +

        '</html>'
    )
    display(HTML(html_str))
    return html_str

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
        absolute (bool):                        Bar plot tf `True`, else waterfall plot.
        normalize (bool):                       If `True`, normalizes the SHAP values across tokens so that the sum of absolute values equals 1 (default=`True`).
        figsize ((int, int)):                   The size of the figure.
        cmap (str):                             The name of a matplotlib colormap used for highlighting.
        show (bool):                            If `True` shows the plot directly, if `False` the plot is returned instead (default: `True`).

    Returns:
        `matplotlib.figure.Figure` object if `show == False`.
    """

    # get attribution scores:
    shap_values = explanation.get_shapley_values(aggregation)
    
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
            scores         = shap_values,
            document_names = document_names,
            normalize      = normalize,
            cmap           = cmap 
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
        **kwargs
    ) -> None:

    if aggregation == 'token':
        higlight_attribution_generator(
            explanation    = explanation,
            document_names = document_names,
            cmap           = cmap,
            **kwargs
        )

    elif aggregation == 'sequence':
        plot_attribution_summary_generator(
            explanation    = explanation,
            document_names = document_names,
            aggregation    = aggregation,
            normalize      = kwargs.get('normalize', True),
            cmap           = cmap,
            **kwargs
        )

    else:
        plot_attribution_generator(
            explanation    = explanation,
            document_names = document_names,
            aggregation    = aggregation,
            cmap           = cmap,
            **kwargs
        )