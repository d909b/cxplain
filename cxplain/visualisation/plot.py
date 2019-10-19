"""
Copyright (C) 2019  Patrick Schwab, ETH Zurich

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions
 of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED
TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.
"""
import numpy as np
from cxplain.backend.validation import Validation


class Plot(object):
    @staticmethod
    def get_default_blue_color():
        N = 256
        vals = np.ones((N, 4))
        vals[:, 0] = np.linspace(1., 157 / 256., N)
        vals[:, 1] = np.linspace(1., 199 / 256., N)
        vals[:, 2] = np.linspace(1., 234 / 256., N)
        return vals

    @staticmethod
    def get_default_red_color():
        N = 256
        vals = np.ones((N, 4))
        vals[:, 0] = np.linspace(1., 245 / 256., N)
        vals[:, 1] = np.linspace(1., 151 / 256., N)
        vals[:, 2] = np.linspace(1., 153 / 256., N)
        return vals

    @staticmethod
    def get_custom_cmap(color_vals):
        import matplotlib
        custom_cmap = matplotlib.colors.ListedColormap(color_vals)
        return custom_cmap

    @staticmethod
    def check_plot_input(x, attribution, confidence=None):
        # Add sample dim - inputs to __check_plot_input__ are passed without sample dim,
        # but __get_attribution_shape__ expects a sample dim.
        x_with_sample_dim = np.expand_dims(x, axis=0)
        attribution_with_sample_dim = np.expand_dims(attribution, axis=0)
        expected_attribution_shape = Validation.get_attribution_shape(x_with_sample_dim)
        if not np.array_equal(attribution_with_sample_dim.shape, expected_attribution_shape):
            raise ValueError("__attribution__ was not of the expected shape. "
                             "__attribution__.shape = {}, "
                             "expected shape = {}.".format(attribution.shape, expected_attribution_shape))

        if confidence is not None:
            numel_a, numel_c = np.prod(attribution.shape), np.prod(confidence.shape)
            if 2*numel_a != numel_c:
                raise ValueError("__confidence__ must have exactly two times as many features as __attribution__. "
                                 "Found number of elements (__attribution__) = {},"
                                 "Found number of elements (__confidence__) = {}".format(numel_a, numel_c))

    @staticmethod
    def plot_attribution_1d(x, attribution, confidence=None, title="Feature Importance",
                            filepath=None, feature_names=None, run_without_gui=False,
                            file_format="pdf"):
        if x.ndim != 1:
            raise ValueError("__x__ must be an array of shape (num_features,).")

        if attribution.ndim != 1:
            raise ValueError("__x__ must be an array of shape (num_features,).")

        if confidence is not None and confidence.ndim != 2:
            raise ValueError("__x__ must be an array of shape (2, num_features).")

        if run_without_gui:
            import matplotlib
            matplotlib.use('Agg')

        import matplotlib.pyplot as plt

        Plot.check_plot_input(x, attribution, confidence)

        # Flatten input features.
        attribution = attribution.reshape((-1,))

        num_features = attribution.shape[-1]
        y_pos = np.arange(num_features)

        if feature_names is None:
            feature_names = ['Feature {}'.format(i) for i in range(num_features)]

        if confidence is not None:
            confidence = confidence.T

        plt.bar(y_pos, attribution,
                yerr=confidence,
                align='center', alpha=0.5, capsize=2)
        plt.xticks(y_pos, feature_names, rotation='vertical')
        plt.ylabel('Feature Importance [%]')
        plt.title(title)

        if filepath is None:
            plt.show()
        else:
            plt.savefig(filepath, format=file_format, bbox_inches="tight",
                        transparent=True, pad_inches=0)
        plt.clf()

    @staticmethod
    def plot_attribution_2d(x, attribution, confidence=None, title="Feature Importance",
                            filepath=None, run_without_gui=False, file_format="pdf"):
        if run_without_gui:
            import matplotlib
            matplotlib.use('Agg')

        import matplotlib.pyplot as plt

        Plot.check_plot_input(x, attribution, confidence)

        if x.shape[-1] == 1:
            x = np.squeeze(x, axis=-1)

        def squeeze_if_channel_dim_equals_1(original):
            if original.shape[-1] == 1:
                return np.squeeze(original, axis=-1)
            else:
                return original

        attribution = squeeze_if_channel_dim_equals_1(attribution)

        fig, ax = plt.subplots(nrows=1, ncols=3 if confidence is not None else 2)
        st = fig.suptitle(title, fontsize=16)

        ax[0].imshow(x)
        ax[0].set_title("Original image")
        ax[0].get_xaxis().set_visible(False)
        ax[0].get_yaxis().set_visible(False)

        ax[1].imshow(attribution, cmap=Plot.get_custom_cmap(Plot.get_default_red_color()))
        ax[1].set_title("Attribution")
        ax[1].get_xaxis().set_visible(False)
        ax[1].get_yaxis().set_visible(False)

        if confidence is not None:
            upper_bound = confidence[..., 1]
            lower_bound = confidence[..., 0]

            upper_bound = squeeze_if_channel_dim_equals_1(upper_bound)
            lower_bound = squeeze_if_channel_dim_equals_1(lower_bound)

            # Using the confidence interval width as uncertainty.
            ax[2].imshow(upper_bound - lower_bound, cmap=Plot.get_custom_cmap(Plot.get_default_blue_color()))
            ax[2].set_title("Uncertainty")
            ax[2].get_xaxis().set_visible(False)
            ax[2].get_yaxis().set_visible(False)

        if filepath is None:
            plt.show()
        else:
            plt.savefig(filepath, format=file_format, bbox_extra_artists=[st], bbox_inches="tight",
                        transparent=True, pad_inches=0)
        plt.clf()

    @staticmethod
    def plot_attribution_nlp(words, attributions,
                             output_format="raw",
                             highlight_color=list([92, 30, 26])):
        """
        Renders attributions for natural language processing (NLP) pipelines.

        :param words: List of strings that make up the input words to your natural language processing (NLP) pipeline.
        :param attributions: Attribution scores for each word in __words__.
        :param output_format: Defines the output format. (Optional, default: 'raw', one of: 'tex' or 'raw').
        :param highlight_color: Highlight color as a list of RGB values (0-255) used for the LaTeX output
                                if __output_format__ == 'tex'. No effect  if __output_format__ != 'tex'.
                                (Optional, default: [92, 30, 26]).
        :return: (i) If __output_format__ == 'tex': Returns a LaTeX string that can be used for rendering
                     text and attributions using a LaTeX renderer. The preamble including package imports
                     is prepended, or
                 (ii) If __output_format__ == 'raw': Returns a raw unicode string that represents words plus the
                      assigned attribution scores in curved brackets {} following the respective words.
        :exception ValueError Thrown if len(__words__) is not the same as len(__attributions__).
        """
        if len(words) != len(attributions):
            raise ValueError("__words__ must have the same length as __attribution__.")

        attributions = np.squeeze(attributions)
        max_a = np.max(attributions)

        if output_format == 'tex':
            highlight_color = (np.array(highlight_color).astype(float)/100.).tolist()
            preamble = r'\usepackage{xcolor}\definecolor{attentioncolor}{rgb}{' + \
                       '{:.3f}, {:.3f}, {:.3f}'.format(*highlight_color) + r'} \n'
            a_frac = lambda a_cur: int(np.rint(a_cur / max_a * 50))
            sentence = preamble + ''.join(['\colorbox{{attentioncolor!{:d}}}{{ {:s} }}'.format(a_frac(a_cur), word)
                                           for word, a_cur in zip(words, attributions)])
        else:
            sentence = ''.join(['{:s} {{{:}}} '.format(word, a_cur) for word, a_cur in zip(words, attributions)])

        return sentence
