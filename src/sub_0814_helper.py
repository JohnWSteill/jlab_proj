"""
Sub_0814 DevClock Somite - 6.25m Time Course - miRNA
http://jira.morgridge.us:8080/browse/RNASEQ-778

runs with Globb v0.3
source /isitools/gup/turn_on_gup.sh
source ../turn_on_globb.sh

"""
from pathlib import Path

import gup
import pandas as pd
from globb import Globb, dim_reduction_plot

SUB_NUM = "814"
PROJ_HOME = Path(__file__).resolve().parent
DATA_DIR = PROJ_HOME / "Data"
EC_FILE = DATA_DIR / "Sub_814_hg19_bf36aa44efc3ad4e" / "miRNA.ec.tab"
ALIGNER = gup.mi_rna_ss.MiRnaSs
FCID = Globb.GUP_DB.getFcidFromFcno("209")
PREP = {
    "file": DATA_DIR / "LFC814_SS815.xlsx",
    "cols": "B, AA",
    "nrows": 98,
    "index_col": 0,  # count AFTER restricting to cols
    "col_names": ["cDNA"],
    "blank_rows": [33, 67],
}
QPLT_ANNOT = [
    {
        "points_legend": False,
        "color": {"col": 0, "type": "real"},
        "shape": {"col": 5, "type": "categ"},
    },
    {
        "points_legend": False,
        "color": {"col": 1, "type": "real"},
        "shape": {"col": 5, "type": "categ"},
    },
]
QPLT_FILENAMES = [
    f"s{SUB_NUM}_quality_with_cdna",
    f"s{SUB_NUM}_quality_with_time",
]


def main():
    sub_0814 = Sub_0814()
    sub_0814.cpm_norm = sub_0814.cpm_norm()
    sub_0814.q90_norm = sub_0814.quant_norm(q=0.9)

    for norm_method, norm_df in zip(
        ("cpm", "qbr"), (sub_0814.cpm_norm, sub_0814.q90_norm)
    ):
        for red_method in ("pca", "umap"):
            dim_reduction_plot(
                sub=sub_0814,
                norm_df=norm_df,
                norm_method=norm_method,
                red_method=red_method,
                q=0.9,
                plot_annot={
                    "points_legend": False,
                    "color": {"col": 1, "type": "real"},
                    "shape": {"col": 5, "type": "categ"},
                },
            )

    sub_0814.make_quality_plots(
        annotations=QPLT_ANNOT,
        filenames=[sub_0814.OUT / fn for fn in QPLT_FILENAMES],
    )


class Sub_0814(Globb):
    SUB_NUM = SUB_NUM
    DATA_DIR = DATA_DIR
    EC_FILE = EC_FILE
    ALIGNER = ALIGNER
    HOME = PROJ_HOME
    OUT = HOME / "Output"
    PREP = PREP

    def get_sample_df(self):
        self.get_sample_df_from_prep()
        self.add_index_mod8()

    def add_index_mod8(self):
        ts_indx_map = pd.read_csv(
            DATA_DIR / "TS_index_map.csv", index_col=3, header=None
        )

        def get_index_mod8(samp):
            info = self.GUP_DB.getSampInfoFromSamp(samp)
            indx_no = ts_indx_map.loc[info["index_label"], 1]
            return (indx_no - 1) % 8 + 1

        self.meta_df["ind_mod8"] = self.meta_df.index.map(get_index_mod8)

    def run_gup_submission(self):
        if list(DATA_DIR.glob(f"Sub{self.SUB_NUM}*")):
            return
        self.GUP_DB.importFlowcell(fcid=FCID)

        ct = gup.run_gup.do_miRNA(
            db=self.GUP_DB,
            fcid=FCID,
            subNum=self.SUB_NUM,
            Sample=self.GUP_DB.getAllSamples(subNum=self.SUB_NUM),
        )
        gup.run_gup.doCopy(ct.getMetadata("outdir"), DATA_DIR)

    def add_sample_id_and_time(self, df):
        """ 
        Had to custom write because PREP name (H1S1_Time01_rep2) different from
        LIMS -
        H1S1smallTime01_rep2 
        """

        def get_time(s):
            return int(s.split("Time")[1][:2])

        LIMS_sample_name_to_id_map = {
            self.GUP_DB.getSampInfoFromSamp(el)["sample_name"].replace(
                '"', ""
            ): el
            for el in self.GUP_DB.getAllSamples(subNum=self.SUB_NUM)
        }
        time_to_LIMS_sample_name_map = {
            get_time(el): el for el in LIMS_sample_name_to_id_map
        }

        df["time"] = df.index.map(get_time)
        # df["LIMS_sample_name"] = df["time"].map(time_to_LIMS_sample_name_map)
        df.index = df["time"].map(time_to_LIMS_sample_name_map)
        df["sample_id"] = df.index.map(LIMS_sample_name_to_id_map)

        return df


if __name__ == "__main__":
    main()
