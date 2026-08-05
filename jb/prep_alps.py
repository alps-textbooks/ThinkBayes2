"""Apply the small, browser-runtime adaptations used by the ALPS curriculum.

The canonical notebooks remain in ``soln/`` and ``examples/``.  ``build.sh``
copies them into this directory and then runs this script, keeping the ALPS
changes reproducible and separate from the upstream textbook sources.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MARKER = "ALPS compatibility"
PYMC_REASON = (
    "This optional PyMC example requires native PyTensor/Numba components "
    "that are not available in the ALPS browser runtime. Run the original "
    "notebook in a local CPython environment to execute this cell."
)
PYMC3_REASON = (
    "This optional comparison uses the legacy PyMC3 stack, which is not "
    "available in the ALPS browser runtime. The grid-based analysis in this "
    "notebook remains executable."
)


def load_notebook(name: str) -> dict:
    with (ROOT / name).open(encoding="utf-8") as stream:
        return json.load(stream)


def write_notebook(name: str, notebook: dict) -> None:
    with (ROOT / name).open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(notebook, stream, ensure_ascii=False, indent=1)
        stream.write("\n")


def replace_source(cell: dict, old: str, new: str, *, location: str) -> bool:
    source = "".join(cell.get("source", []))
    if new and new in source:
        return False
    if old not in source:
        if not new:
            return False
        raise ValueError(f"Expected source not found at {location}: {old!r}")
    cell["source"] = source.replace(old, new).splitlines(keepends=True)
    return True


def disable_code_cell(
    notebook: dict,
    index: int,
    *,
    expected: str,
    reason: str,
    location: str,
) -> bool:
    cell = notebook["cells"][index]
    source = "".join(cell.get("source", []))
    if cell.get("cell_type") == "markdown" and MARKER in source:
        return False
    if cell.get("cell_type") != "code" or expected not in source:
        raise ValueError(
            f"Unexpected cell at {location}; expected a code cell containing {expected!r}"
        )

    original = source.rstrip()
    note = (
        f"> **{MARKER}:** {reason}\n\n"
        "The original textbook code is preserved for reference:\n\n"
        f"```python\n{original}\n```"
    )
    metadata = cell.setdefault("metadata", {})
    metadata["alps"] = {"disabled": True, "reason": reason}
    cell["cell_type"] = "markdown"
    cell["source"] = note.splitlines(keepends=True)
    cell.pop("execution_count", None)
    cell.pop("outputs", None)
    return True


def disable_cells(
    name: str,
    notebook: dict,
    cells: dict[int, str],
    reason: str = PYMC_REASON,
) -> bool:
    changed = False
    for index, expected in cells.items():
        changed |= disable_code_cell(
            notebook,
            index,
            expected=expected,
            reason=reason,
            location=f"{name}:cell[{index}]",
        )
    return changed


def adapt_chap19() -> bool:
    name = "chap19_v3.ipynb"
    notebook = load_notebook(name)
    changed = disable_cells(
        name,
        notebook,
        {
            25: "import pymc as pm",
            28: "pm.model_to_graphviz(model)",
            31: "pm.sample_prior_predictive",
            32: "type(idata)",
            35: "idata.prior['lam']",
            38: "sample_prior_pymc",
            40: "idata.prior['goals']",
            43: "sample_prior_pred_pymc",
            46: "with pm.Model() as model",
            48: "with pm.Model() as model2",
            50: "pm.sample(500",
            52: "idata2.posterior['lam']",
            53: "sample_post_pymc.shape",
            55: "sample_post_pymc",
            58: "pm.sample_posterior_predictive",
            60: "idata2_pred.posterior_predictive",
            61: "sample_post_pred_pymc.shape",
            65: "sample_post_pred_pymc",
            86: "with pm.Model() as model3",
            89: "pm.sample(500",
            91: "az.plot_posterior",
            93: "idata3.posterior['a']",
            94: "idata3.posterior['b']",
            96: "idata3.posterior['sigma']",
            111: "with pm.Model() as model4",
            113: "pm.sample(500",
            115: "idata4.posterior['b0']",
            117: "idata4.posterior['sigma']",
            119: "param_names",
            121: "credible_interval",
            123: "Posterior mean",
            128: "with pm.Model() as model5",
            130: "with pm.Model() as model6",
            134: "with pm.Model() as model7",
            143: "with pm.Model() as model8",
            144: "az.plot_posterior(idata8)",
            151: "with pm.Model() as model9",
            152: "with model9",
            153: "az.summary(idata9)",
            154: "az.plot_posterior(idata9)",
        },
    )
    changed |= replace_source(
        notebook["cells"][70],
        "https://happiness-report.s3.amazonaws.com/2020/WHR20_DataForFigure2.1.xls",
        "https://github.com/AllenDowney/ThinkBayes2/raw/master/data/WHR20_DataForFigure2.1.xls",
        location=f"{name}:cell[70]",
    )
    if changed:
        write_notebook(name, notebook)
    return changed


def adapt_chap20() -> bool:
    name = "chap20.ipynb"
    notebook = load_notebook(name)
    changed = disable_cells(
        name,
        notebook,
        {
            65: "import pymc as pm",
            67: "with model",
            69: "with model",
            71: "with model",
            73: "with model",
            75: "pm.Binomial",
            77: "pm.sample(1000",
            79: "trace['yeast conc']",
            82: "pm.sample_prior_predictive",
            84: "prior_sample['count']",
            86: "count == 49",
            88: "prior_sample['yeast conc']",
            90: "cdf_abc",
            92: "cdf_pymc.plot",
            96: "prior_sample['yeast in chamber']",
            98: "squares_counted/total_squares",
            100: "likelihood = binom",
            101: "likelihood.shape",
            103: "plt.plot(n*p, likelihood",
            105: "qs = prior_sample['yeast conc']",
            107: "posterior_pmf.normalize",
            109: "cdf_pymc.plot",
        },
    )
    if changed:
        write_notebook(name, notebook)
    return changed


def adapt_redline_pymc() -> bool:
    name = "redline_pymc.ipynb"
    notebook = load_notebook(name)
    changed = replace_source(
        notebook["cells"][5],
        "import pymc as pm\n\n",
        "",
        location=f"{name}:cell[5]",
    )
    changed |= disable_cells(
        name,
        notebook,
        {
            20: "with pm.Model() as model",
            61: "model = make_model",
            62: "pm.model_to_graphviz",
            64: "pm.sample_prior_predictive",
            65: "idata['prior']['gap_time']",
            66: "kde_gap_time",
            67: "idata.prior['elapsed_time']",
            68: "idata.prior_predictive['n']",
            70: "pm.sample()",
            71: "az.summary(idata)",
            72: "az.plot_trace",
            74: "az.plot_posterior",
            76: "def get_sample",
            77: "get_sample(idata, 'gap_time')",
            78: "get_sample(idata, 'elapsed_time')",
            79: "get_sample(idata, 'remaining_time')",
            81: "sns.kdeplot(gap_times",
            83: "model = make_model",
            84: "pm.sample()",
            86: "get_sample(idata, 'gap_time')",
            87: "get_sample(idata, 'elapsed_time')",
            88: "get_sample(idata, 'remaining_time')",
            89: "get_sample(idata, 'rate')",
            91: "np.corrcoef",
            94: "joint_contour(elapsed_times",
            99: "run_model(make_model_func)",
            103: "def plot_cdfs",
            104: "plot_cdfs('model1'",
            108: "plot_model_results([1], 'elapsed')",
            109: "plot_model_results([1], 'remaining')",
            112: "plot_model_results(all_models, 'gap')",
            113: "plot_model_results(all_models, 'elapsed')",
            114: "plot_model_results(all_models, 'remaining')",
            115: "plot_model_results(all_models, 'rate')",
            118: "def make_figures",
            119: "make_figures('gap')",
            120: "make_figures('elapsed')",
            121: "make_figures('remaining')",
            122: "make_figures('rate')",
        },
    )
    if changed:
        write_notebook(name, notebook)
    return changed


def adapt_hospital() -> bool:
    name = "hospital.ipynb"
    notebook = load_notebook(name)
    changed = disable_cells(
        name,
        notebook,
        {
            17: "import pymc3 as pm",
            18: "make_model()",
            19: "pm.sample_prior_predictive",
            21: "az.plot_posterior",
            23: "trace['xs']",
            25: "plot_posterior(trace_xs[0])",
            30: "compare_cdf(prior_mu, pred['mu'])",
            31: "compare_cdf(prior_sigma, pred['sigma'])",
            59: "pred['xs']",
            60: "compare_cdf(marginal_x, pred_xs[0])",
            75: "trace['mu']",
            76: "trace['sigma']",
            77: "trace_xs[-1]",
            93: "trace['mu']",
            94: "trace['sigma']",
            95: "trace_xs[i]",
            100: "trace_xs[i]",
            102: "trace_xs[i].mean",
        },
        reason=PYMC3_REASON,
    )
    if changed:
        write_notebook(name, notebook)
    return changed


def adapt_bread() -> bool:
    name = "bread.ipynb"
    notebook = load_notebook(name)
    changed = replace_source(
        notebook["cells"][5],
        "import pymc as pm\nimport arviz as az\n\n",
        "",
        location=f"{name}:cell[5]",
    )
    changed |= disable_cells(
        name,
        notebook,
        {
            19: "def max_normal_logp",
            21: "def make_model",
            23: "pm.sample()",
            25: "az.summary(trace)",
            27: "def plot_posterior",
            28: "plot_posterior(trace, 'mu'",
            29: "plot_posterior(trace, 'sigma'",
            30: "plot_posterior(trace, 'k'",
            32: "az.extract(trace)",
        },
    )
    if changed:
        write_notebook(name, notebook)
    return changed


def adapt_cancer() -> bool:
    name = "cancer.ipynb"
    notebook = load_notebook(name)
    table_source = '''# Snapshot of the rows used in this example, embedded for offline ALPS use.
sites = [
    "Oral Cavity and Pharynx",
    "Esophagus",
    "Stomach",
    "Colon and Rectum",
    "Liver and Intrahepatic Bile Duct",
]
columns = ["Localized", "Regional", "Distant", "Unknown/unstaged"]

survival = [
    ["88.2%", "70.8%", "38.2%", "58.8%"],
    ["48.0%", "28.1%", "5.6%", "13.6%"],
    ["73.8%", "35.2%", "6.5%", "27.7%"],
    ["91.4%", "74.0%", "15.8%", "42.8%"],
    ["37.3%", "13.0%", "3.2%", "8.1%"],
]
cases = [
    [16225, 32154, 9328, 3138],
    [3954, 7071, 8722, 2062],
    [10613, 9131, 13737, 3371],
    [72619, 72115, 46247, 10184],
    [24061, 13361, 10622, 5459],
]

def make_raw_table(values):
    table = pd.DataFrame(values, columns=columns)
    table.insert(0, "Unnamed: 0", sites)
    return table

tables = [make_raw_table(survival), make_raw_table(cases)]'''
    cell = notebook["cells"][11]
    old_source = "".join(cell.get("source", []))
    if "Snapshot of the rows used" not in old_source:
        if "pd.read_html('CanQues Results.html')" not in old_source:
            raise ValueError(f"Unexpected source at {name}:cell[11]")
        cell["source"] = table_source.splitlines(keepends=True)
        changed = True
    else:
        changed = False

    markov_source = '''class MarkovChain:
    """Small NumPy replacement for the QuantEcon API used in this example."""

    def __init__(self, transition_matrix, state_values):
        self.P = np.asarray(transition_matrix, dtype=float)
        self.state_values = np.asarray(state_values)
        self.state_to_index = {
            state: index for index, state in enumerate(self.state_values)
        }

    def simulate(self, ts_length, init):
        state = self.state_to_index.get(init, init)
        result = np.empty(ts_length, dtype=self.state_values.dtype)
        for i in range(ts_length):
            result[i] = self.state_values[state]
            state = np.random.choice(len(self.state_values), p=self.P[state])
        return result


mc = MarkovChain(P, state_values=states)'''
    cell = notebook["cells"][32]
    old_source = "".join(cell.get("source", []))
    if "Small NumPy replacement" not in old_source:
        if "from quantecon import MarkovChain" not in old_source:
            raise ValueError(f"Unexpected source at {name}:cell[32]")
        cell["source"] = markov_source.splitlines(keepends=True)
        changed = True

    changed |= disable_cells(
        name,
        notebook,
        {
            60: "import pymc as pm",
            61: "pmx.DiscreteMarkovChain",
            62: "idx_to_state[traj]",
        },
    )
    if changed:
        write_notebook(name, notebook)
    return changed


def adapt_frog() -> bool:
    name = "frog.ipynb"
    notebook = load_notebook(name)
    source = "".join(notebook["cells"][10].get("source", []))
    replacement = '''from sympy import init_printing

# ALPS provides a lightweight IPython shell without the ``colors`` attribute
# expected by SymPy's pretty-printer initialization.
from IPython import get_ipython
shell = get_ipython()
if shell is None or hasattr(shell, "colors"):
    init_printing(use_latex=False)'''
    if "hasattr(shell, \"colors\")" in source:
        changed = False
    elif "init_printing(use_latex=False)" in source:
        notebook["cells"][10]["source"] = replacement.splitlines(keepends=True)
        changed = True
    else:
        raise ValueError(f"Unexpected source at {name}:cell[10]")
    if changed:
        write_notebook(name, notebook)
    return changed


def adapt_hepatitis() -> bool:
    name = "hepatitis.ipynb"
    notebook = load_notebook(name)
    changed = disable_cells(
        name,
        notebook,
        {
            4: "pip install pymc arviz",
            12: "with pm.Model(coords=coords)",
            15: "pm.model_to_graphviz(model)",
            17: "pm.sample(2000",
            18: "az.summary(idata",
            20: "az.plot_posterior(idata",
            22: "az.summary(idata",
            29: "def make_heterogeneity_model",
            30: "make_heterogeneity_model()",
            31: "pm.sample(2000",
            33: "az.summary",
            34: "az.plot_posterior",
            35: "az.summary",
            37: "az.plot_density",
            38: "summary_ind",
            40: "def fit_heterogeneity",
            41: "fit_heterogeneity(scale)",
        },
    )
    changed |= replace_source(
        notebook["cells"][5],
        "import pymc as pm\nimport arviz as az\n\n",
        "",
        location=f"{name}:cell[5]",
    )
    if changed:
        write_notebook(name, notebook)
    return changed


ADAPTERS = (
    adapt_chap19,
    adapt_chap20,
    adapt_redline_pymc,
    adapt_hospital,
    adapt_bread,
    adapt_cancer,
    adapt_frog,
    adapt_hepatitis,
)


def validate() -> None:
    forbidden_code = (
        "import pymc",
        "import arviz",
        "import pymc_extras",
        "from quantecon import MarkovChain",
        "pd.read_html('CanQues Results.html')",
    )
    for name in (
        "chap19_v3.ipynb",
        "chap20.ipynb",
        "redline_pymc.ipynb",
        "hospital.ipynb",
        "bread.ipynb",
        "cancer.ipynb",
        "hepatitis.ipynb",
    ):
        notebook = load_notebook(name)
        if not any(
            MARKER in "".join(cell.get("source", []))
            for cell in notebook["cells"]
        ):
            raise ValueError(f"{name} has no ALPS compatibility cells")
        for index, cell in enumerate(notebook["cells"]):
            if cell.get("cell_type") != "code":
                continue
            source = "".join(cell.get("source", []))
            for blocked in forbidden_code:
                if blocked in source:
                    raise ValueError(
                        f"{name}:cell[{index}] still contains unsupported {blocked!r}"
                    )

    cancer = load_notebook("cancer.ipynb")
    if "Small NumPy replacement" not in "".join(cancer["cells"][32]["source"]):
        raise ValueError("cancer.ipynb still depends on QuantEcon")
    if "pd.read_html" in "".join(cancer["cells"][11]["source"]):
        raise ValueError("cancer.ipynb still depends on an external HTML file")

    frog = load_notebook("frog.ipynb")
    if "hasattr(shell, \"colors\")" not in "".join(frog["cells"][10]["source"]):
        raise ValueError("frog.ipynb is missing the ALPS SymPy guard")

    chap19 = load_notebook("chap19_v3.ipynb")
    if "happiness-report.s3.amazonaws.com" in "".join(
        source
        for cell in chap19["cells"]
        for source in cell.get("source", [])
        if cell.get("cell_type") == "code"
    ):
        raise ValueError("chap19_v3.ipynb still uses the blocked S3 data URL")

    chap17 = load_notebook("chap17.ipynb")
    warning_cell = "".join(chap17["cells"][74].get("source", []))
    if "xlabel=r'$\\sigma$" not in warning_cell:
        raise ValueError("chap17.ipynb is missing raw LaTeX label strings")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check", action="store_true", help="validate already-prepared notebooks"
    )
    args = parser.parse_args()

    if not args.check:
        for adapter in ADAPTERS:
            changed = adapter()
            print(f"{'adapted' if changed else 'checked'} {adapter.__name__[6:]}")
    validate()
    print("ALPS notebook adaptations validated")


if __name__ == "__main__":
    main()
