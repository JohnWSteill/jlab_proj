import string

import gup
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap
from bokeh.models import Legend
from bokeh.palettes import Category20
from bokeh.plotting import figure, output_file, show
from scipy import stats
from sklearn.decomposition import PCA as sklearnPCA
from sklearn.preprocessing import StandardScaler


class Globb:
    """ Perhaps implement as Abstract Base Class:
    https://www.python-course.eu/python3_abstract_classes.php
    """

    def get_sample_df(self):
        # TODO: assert what I expect to be true.
        raise NotImplementedError

    def run_gup_submission(self):
        # TODO: assert what I expect to be true.
        raise NotImplementedError

    GUP_DB = gup.mongo_gup.MongoDb()

    def __init__(self, aligner=None):
        meta_df_filename = f"meta_df_{self.SUB_NUM}.csv"
        expr_df_filename = f"expr_df_{self.SUB_NUM}.csv"
        if not aligner:
            aligner = self.ALIGNER
        assert aligner in (gup.mi_rna_ss.MiRnaSs, gup.bt_rsem.BtRsem)

        try:
            self.expr_df = pd.read_csv(
                self.DATA_DIR / expr_df_filename, index_col=0
            )
        except FileNotFoundError:
            self.run_gup_submission()
            self.expr_df = self.get_expression_df()
            self.expr_df = self.add_sample_id_1_to_1(self.expr_df)
            self.expr_df = self.swap_names_to_id_in_index(self.expr_df)
            self.expr_df.drop("sample_name", axis=1, inplace=True)
            self.expr_df.to_csv(self.DATA_DIR / expr_df_filename)

        try:
            self.meta_df = pd.read_csv(
                self.DATA_DIR / meta_df_filename, index_col=0
            )
        except FileNotFoundError:
            self.get_sample_df()
            self.expr_df.index = self.expr_df.index.rename(
                self.meta_df.index.name
            )
            self.get_efficiency()
            self.get_alignment(aligner=aligner)
            self.meta_df.to_csv(self.DATA_DIR / meta_df_filename)
        assert set(self.meta_df.index) == set(self.expr_df.index)

    def get_sample_df_from_prep(self):
        """
        I expect this function to get cut in half, with the 1st half
        specific to a tightly defined self.PREP disctionary, and the
        second universal to other paths towards a meta_df.
        """
        self.meta_df = pd.read_excel(
            self.PREP["file"],
            usecols=self.PREP["cols"],
            nrows=self.PREP["nrows"],
            index_col=self.PREP["index_col"],
        ).rename_axis("SampleName")
        self.meta_df.columns = self.PREP["col_names"]
        self.meta_df = self.meta_df.iloc[
            [
                el
                for el, _ in enumerate(self.meta_df.index)
                if el not in self.PREP["blank_rows"]
            ],
            :,
        ]

        self.meta_df = self.add_sample_id_and_time(self.meta_df)
        self.meta_df = self.swap_names_to_id_in_index(self.meta_df)
        self.meta_df["lane"] = self.meta_df.index.map(
            lambda samp: self.GUP_DB.getSampInfoFromSamp(samp)["flowcell_lane"]
        )
        self.add_small_labels()

    def get_sub_data_dir(self):
        # will throw IndexError if not there
        return list(self.DATA_DIR.glob(f"Sub_{self.SUB_NUM}*"))[0]

    def get_expression_df(self):
        if hasattr(self, "EC_FILE"):
            expres_file = self.EC_FILE
        else:
            assert hasattr(self, "TPM_FILE")
            expres_file = self.TPM_FILE

        expr_df = pd.read_csv(expres_file, sep="\t", index_col=0).transpose()
        if "description" in expr_df.index:
            expr_df.drop("description", axis=0, inplace=True)
        return expr_df

    def get_efficiency(self):
        def get_eff(samp):
            return gup.run_gup.getBclCTForSamp(
                self.GUP_DB, samp
            ).getMetadata()["calcMetrics"]["efficiency"][samp]

        self.meta_df["efficiency"] = self.meta_df.index.map(get_eff)

    def get_alignment(
        self, aligner, df_met_name="aligned", gup_met_name="percAligned"
    ):
        """ get alignment values into meta_df
        The day we need kwargs, like ReadOneOnly, this will have to be changed.
        """
        self.meta_df[df_met_name] = np.NaN
        for samp in self.meta_df.index:
            samp_info = self.GUP_DB.getSampInfoFromSamp(samp)
            self.meta_df.loc[samp, "aligned"] = (
                aligner(db=self.GUP_DB, fcid=samp_info["fcid"], Sample=samp)
                .calcTuples[0]
                .getMetadata("calcMetrics")[gup_met_name]
            )

    def cpm_norm(self):
        return self.expr_df.div(self.expr_df.sum(axis=1), axis=0) * 1e6

    def quant_norm(self, q=0.5):
        pseudo_expr_df = self.expr_df + 0.1
        ref_sample = stats.gmean(pseudo_expr_df, axis=0)
        size_factor = pseudo_expr_df.div(ref_sample, axis=1).quantile(
            q=q, axis=1
        )
        return pseudo_expr_df.div(size_factor, axis=0)

    def add_sample_id_1_to_1(self, df):
        """ If sample_names are not unique, replace this function in subclass. """
        temp_dict = {
            self.GUP_DB.getSampInfoFromSamp(el)["sample_name"].replace(
                '"', ""
            ): el
            for el in self.GUP_DB.getAllSamples(subNum=self.SUB_NUM)
        }
        df["sample_id"] = df.index.map(temp_dict)
        return df

    def swap_names_to_id_in_index(self, df):
        df["sample_name"] = df.index
        return df.set_index("sample_id")

    def add_flowcell_lane(self):
        self.meta_df["flowcell_lane"] = self.meta_df.index.map(
            {
                s: self.GUP_DB.getSampInfoFromSamp(s)["flowcell_lane"]
                for s in self.meta_df.index
            }
        )

    def add_small_labels(self):
        """ 
        small labels are lowercase letters used to annotate plots
        """
        n = len(self.meta_df)
        if n <= 26:
            self.meta_df["small_labels"] = list(
                string.ascii_lowercase[: len(self.meta_df)]
            )
        else:
            self.meta_df["small_labels"] = [str(i + 1) for i in range(n)]

    def add_PC12_to_meta_df(self, df, col_names=["pc1", "pc2"]):
        # TODO: cast df to np.array.transpose()
        # move colnames out of signature into constant
        pca = sklearnPCA(n_components=2)
        pca.fit_transform(df.transpose())
        X = pca.transform(df.transpose())
        self.meta_df[col_names[0]] = np.NaN
        self.meta_df[col_names[1]] = np.NaN
        self.meta_df[col_names] = X.T.dot(df.transpose()).T

    def add_UMAP_to_meta_df(self, df):
        # https://umap-learn.readthedocs.io/en/latest/basic_usage.html
        umap_xy = umap.UMAP().fit_transform(
            StandardScaler().fit_transform(df.values)
        )
        self.meta_df["umap1"] = umap_xy[:, 0]
        self.meta_df["umap2"] = umap_xy[:, 1]

    def remove_outliers(self, drop_rows):
        self.meta_df.drop(self.meta_df.index[drop_rows], inplace=True)
        self.expr_df.drop(self.expr_df.index[drop_rows], inplace=True)
        assert set(self.meta_df.index) == set(self.expr_df.index)

    def make_quality_plots(self, annotations, filenames):
        for annot, fn in zip(annotations, filenames):
            Globb_Plot(
                self.meta_df,
                plot_annot=annot,
                plot_type="quality",
                filename=fn,
            )


class Globb_Plot:
    def __init__(
        self,
        df,
        cols=None,
        plot_annot=None,
        plot_type=None,
        plot_title=None,
        filename=None,
        scatter_sz = 80,
    ):
        if not plot_annot:
            plot_annot = {}
        plt.clf()
        plt.rcParams["figure.figsize"] = (10, 10)
        self.df = df
        self.plot_annot = plot_annot
        do_points_legend, do_shape_legend, do_cbar = False, False, False

        (
            x_col,
            y_col,
            xlabel,
            ylabel,
            title,
            filename,
        ) = self.get_xy_labels_title(plot_type, plot_title, filename)
        if "mode" in self.plot_annot and self.plot_annot["mode"] == "bokeh":

            self.get_bokeh_plot(x_col, y_col, xlabel, ylabel, title, filename)

        else:
            self.points_annot = [
                {"x": el[1][x_col], "y": el[1][y_col], "s": scatter_sz}
                for el in self.df.iterrows()
            ]
            if (
                "color" in self.plot_annot
                and self.plot_annot["color"]["type"] == "real"
            ):
                do_cbar = True
                cbar_ticks = self.real_colorbar_scale()
            if "shape" in self.plot_annot:
                do_shape_legend = True
                legend_handles, legend_title = self.shapes_and_legend()
            elif (
                "points_legend" in self.plot_annot
                and self.plot_annot["points_legend"]
            ):
                # cant do points and shape for now
                do_points_legend = True
                legend_handles, legend_title = self.points_legend_annot()

            for annot_dict in self.points_annot:
                plt.scatter(**annot_dict)

            if do_cbar:
                self.my_plotcolorbar(cbar_ticks)
            if do_shape_legend:
                plt.legend(handles=legend_handles, title=legend_title)
            if do_points_legend:
                self.sidebox_legend(legend_handles, legend_title)

            ax = plt.gca()
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            plt.savefig(filename)

    def get_xy_labels_title(self, plot_type, title, filename):
        if plot_type == "quality":
            if not filename:
                filename = "qplot.png"
            x, y = ("efficiency", "aligned")
            xlabel = "Efficiency: Relative number of reads"
            ylabel = "Alignment Percentage"
            if not title:
                title = "Quality Clustering"
        elif plot_type == "pca":
            if not filename:
                filename = "pca_plot.png"
            x, y, xlabel, ylabel = ("pc1", "pc2", "pc1", "pc2")
            if not title:
                title = "PCA Clustering"
        elif plot_type == "umap":
            if not filename:
                filename = "umap_plot.png"
            x, y, xlabel, ylabel = ("umap1", "umap2", "umap1", "umap2")
            if not title:
                title = "UMAP Clustering"
        else:
            raise ValueError(f"Plot type {plot_type} is not valid.")
        return (x, y, xlabel, ylabel, title, filename)

    def sidebox_legend(self, legend_handles, legend_title):
        ax = plt.gca()
        box = ax.get_position()
        # Put a legend to the right of the current axis
        ax.legend(
            handles=legend_handles,
            title=legend_title,
            loc="center left",
            bbox_to_anchor=(1, 0.5),
        )
        max_len = max(
            [len(el.get_text()) for el in plt.gca().get_legend().get_texts()]
        )
        if max_len >= 39:
            ax.set_position([box.x0, box.y0, box.width * 0.5, box.height])
            fontsize = 8
        else:
            ax.set_position([box.x0, box.y0, box.width * 0.7, box.height])
            fontsize = 12

        plt.setp(plt.gca().get_legend().get_texts(), fontsize=fontsize)
        plt.gca().get_legend().get_title().set_fontsize(fontsize + 2)

        for (d, row) in zip(self.points_annot, self.df.iterrows()):
            ax.annotate(
                row[1].small_labels,
                (d["x"], d["y"]),
                xytext=(5, -2),
                textcoords="offset points",
                size=fontsize + 2,
                color="darkslategrey",
            )

    def my_plotcolorbar(self, cbar_ticks):
        cbar = plt.colorbar(ticks=[0, 0.5, 1])
        cbar.ax.set_yticklabels(cbar_ticks)
        cbar.ax.set_ylabel(
            self.df.columns[self.plot_annot["color"]["col"]],
            rotation=270,
            labelpad=17,
            fontsize=12,
        )

    def shapes_and_legend(self):
        markers = "o+vspxd^1<>P*X"

        cat_data = pd.Categorical(
            self.df.iloc[:, self.plot_annot["shape"]["col"]]
        )
        cat_data_to_marker = {
            el: m for m, el in zip(markers, cat_data.categories)
        }
        for d, m in zip(self.points_annot, cat_data):
            d.update({"marker": cat_data_to_marker[m]})

        if (
            "color" in self.plot_annot
            and self.plot_annot["color"]["type"] != "real"
        ):
            cat_data = self.df.iloc[:, self.plot_annot["color"]["col"]]
            vals = sorted(list(set(cat_data)))
            qual_cmap = matplotlib.cm.get_cmap("tab20", len(vals))
            cat_data_to_color = {v: qual_cmap(i) for i, v in enumerate(vals)}
            for d, v in zip(self.points_annot, cat_data):
                d.update({"c": cat_data_to_color[v]})

        legend_handles = []
        legend_title = self.df.columns[self.plot_annot["shape"]["col"]]
        for el, m in cat_data_to_marker.items():
            legend_handles.append(
                matplotlib.lines.Line2D(
                    [],
                    [],
                    linestyle="None",
                    marker=m,
                    markerfacecolor="g",
                    label=el,
                )
            )
        return (legend_handles, legend_title)

    def real_colorbar_scale(self):
        # plt.set_cmap("gist_ncar")
        cm = plt.get_cmap()
        vals = list(self.df.iloc[:, self.plot_annot["color"]["col"]])
        vmin, vmax = min(vals), max(vals)
        cnorm = matplotlib.colors.Normalize(vmin, vmax)
        cvals = [cm(cnorm(v)) for v in vals]
        for d, c in zip(self.points_annot, cvals):
            d.update({"c": np.array([c])})
        cbar_ticks = [
            np.ceil(vmin),
            np.round((vmin + vmax) / 2),
            np.floor(vmax),
        ]
        return cbar_ticks

    def points_legend_annot(self):
        if "c" not in self.points_annot[0]:
            cm = plt.get_cmap("Spectral", len(self.df))
            for i, d in enumerate(self.points_annot):
                d["c"] = np.array([cm(i)])
        legend_title = "Samples"
        legend_handles = []
        for row, d in zip(self.df.iterrows(), self.points_annot):
            d.update({"label": f"{row[1].small_labels}) {row[0]}"})
            legend_handles.append(
                matplotlib.lines.Line2D(
                    [],
                    [],
                    linestyle="None",
                    marker="o",
                    markerfacecolor=d["c"].flatten(),
                    label=d["label"],
                )
            )
        return (legend_handles, legend_title)

    def get_bokeh_plot(self, x, y, xlabel, ylabel, title, filename):

        TOOLTIPS = [(ylabel, "$y"), (xlabel, "$x")]

        output_file(filename + ".html")

        tools = "pan,wheel_zoom,box_zoom,reset,save,hover"

        p = figure(
            title=title,
            plot_width=700,
            plot_height=500,
            toolbar_location="right",
            tooltips=TOOLTIPS,
            tools=tools,
        )
        legend_it = []

        (x_data, y_data, samps, palette) = (
            self.df[x],
            self.df[y],
            self.df.index,
            Category20[len(self.df.index)],
        )
        for x_d, y_d, name, color in zip(x_data, y_data, samps, palette):
            c = p.circle(x_d, y_d, size=10, color=color, alpha=0.8)
            legend_it.append((name, [c]))

        legend = Legend(items=legend_it, location=(0, 0))
        p.add_layout(legend, "right")
        p.legend.label_text_font_size = "8pt"
        p.xaxis.axis_label = xlabel
        p.yaxis.axis_label = ylabel

        show(p)


def dim_reduction_plot(
    sub, norm_df, norm_method, red_method, q=None, plot_annot=None
):
    assert norm_method in ["cpm", "qbr", "tpm"]
    assert red_method in ["pca", "umap"]
    if not plot_annot:
        plot_annot = {"color": {"col": 1, "type": "real"}}
    if red_method == "pca":
        sub.add_PC12_to_meta_df(norm_df)
    else:
        sub.add_UMAP_to_meta_df(norm_df)
    Globb_Plot(
        sub.meta_df,
        plot_annot=plot_annot,
        plot_type=red_method,
        filename=sub.OUT / get_plot_filename(sub, norm_method, red_method, q),
    )


def get_plot_filename(sub, norm_method, red_method, q):
    result = f"Sub_{sub.SUB_NUM}"
    if norm_method == "cpm":
        result += "_cpm"
    elif norm_method == "qbr":
        result += f"q{q*100:.0f}_qbr"
    else:
        result += "_tpm"
    if red_method == "pca":
        result += "_pca.png"
    else:
        result += "_umap.png"
    return result
