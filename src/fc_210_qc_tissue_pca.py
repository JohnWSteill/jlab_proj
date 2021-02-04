"""
http://jira.morgridge.us:8080/browse/RNASEQ-780

runs with Globb v0.3
source /isitools/gup/turn_on_gup.sh
source /w5home/jsteill/PROJ/GLOBB/turn_on_globb.sh

"""

import gup

from globb import Globb, Globb_Plot
from pathlib import Path
import pandas as pd


SUB_NUM = "FC_210_QC"
ALIGNER = gup.bt_rsem.BtRsem
FCID = Globb.GUP_DB.getFcidFromFcno("210")


def main():
    sub = Sub_210_QC(aligner=ALIGNER)
    sub.add_PC12_to_meta_df(sub.expr_df)
    sub.add_UMAP_to_meta_df(sub.expr_df)
    Globb_Plot(
        sub.meta_df,
        plot_annot={
            "color": {"col": 10, "type": "categ"},
            "shape": {"col": 10},
        },
        plot_type="umap",
        filename=sub.OUT / "tissue_umap.png",
        scatter_sz = 30
    )
    Globb_Plot(
        sub.meta_df,
        plot_annot={
            "color": {"col": 10, "type": "categ"},
            "shape": {"col": 10},
        },
        plot_type="pca",
        filename=sub.OUT / "tissue_pca.png",
        scatter_sz = 30
    )


class Sub_210_QC(Globb):
    SUB_NUM = SUB_NUM
    HOME = Path(__file__).resolve().parent
    DATA_DIR = HOME / "Data"
    OUT = HOME / "Output"
    for dir in (DATA_DIR, OUT):
        Path.mkdir(dir, exist_ok=True)

    def run_gup_submission(self):
        self.GUP_DB.importFlowcell(fcid=FCID)
        self.samples = (
            self.GUP_DB.getAllSamples(fcid=FCID, subNum="820")
            + sorted(self.GUP_DB.getAllSamples(fcid=FCID, subNum="697"))[-2:]
        )
        assert len(self.samples) == 194
        try:
            self.SUB_DATA_DIR = self.get_sub_data_dir()
        except IndexError:
            ct = gup.run_gup.doCollationCalcFromSamples(
                db=self.GUP_DB,
                fcid=FCID,
                subNum=self.SUB_NUM,
                Sample=self.samples,
                ref_seq="mm10",
            )
            gup.run_gup.doCopy(ct.getMetadata("outdir"), self.DATA_DIR)
            self.SUB_DATA_DIR = self.get_sub_data_dir()
        self.TPM_FILE = self.SUB_DATA_DIR / "genes.no_mt.tpm.rescale.tab"

    def get_sample_df(self):
        """ only interested in tissue type: build meta_df from samples """
        self.meta_df = pd.DataFrame(
            {s: self.GUP_DB.getSampInfoFromSamp(s) for s in self.samples}
        ).transpose()
        self.meta_df["tissue"] = self.meta_df["sample_name"].map(
            lambda s: s.split("_")[0].lower()[1:]
        )

    def add_sample_id_1_to_1(self, df):
        # masks glob.py for now, because I don't have real subnum. But
        # might make it's way back up, this soln is more general.
        temp_dict = {
            self.GUP_DB.getSampInfoFromSamp(el)["sample_name"].replace(
                '"', ""
            ): el
            for el in self.samples
        }
        df["sample_id"] = df.index.map(temp_dict)
        return df


if __name__ == "__main__":
    main()
