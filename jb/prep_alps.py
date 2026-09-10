"""Adapt the built Jupyter Book notebooks for the in-browser Python runtime.

``build.sh`` copies the canonical notebooks out of ``soln/`` and ``examples/``
into this directory and then runs this script, so every adaptation below is
reapplied on each build and the upstream sources stay untouched.

Two kinds of change are made, and no others.

Marking
    A cell that needs a package with no WebAssembly build gets
    ``"alps_reference_only": true`` in its metadata. It stays a real code cell
    with its source intact -- the flag says the code is worth reading but
    cannot execute in a browser, and the wording shown to the reader belongs to
    whatever renders the notebook rather than to the notebook itself. The
    packages in question are PyMC and PyMC3, which reach Numba through PyTensor
    and Theano, and QuantEcon, which uses Numba directly; Numba compiles
    through LLVM and has no WebAssembly build.

Salvage
    Marking on its own would take whole notebooks down with the marked cells.
    Three notebooks import PyMC in the middle of the same cell that imports
    NumPy, Pandas and ``utils``: the failed import costs them ``decorate`` and
    ``set_pyplot_params``, and every later figure fails with a NameError. Those
    imports are split into a cell of their own so the rest of the cell still
    binds. ``cancer.ipynb`` needs two more, both described where they are
    applied.

Cells that merely *mention* an unavailable package are left alone. A ``def``
whose body calls ``pm.Model`` executes perfectly well -- it just cannot be
called -- and marking it would claim a failure that does not happen.

Run ``python prep_alps.py`` to apply, ``python prep_alps.py --check`` to verify
an already-prepared directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent

REFERENCE_ONLY = "alps_reference_only"


def load_notebook(name: str) -> dict:
    with (ROOT / name).open(encoding="utf-8") as stream:
        return json.load(stream)


def ensure_cell_ids(notebook: dict) -> None:
    """Give every cell an ``id`` when the notebook's format requires one.

    nbformat 4.5 made cell ids mandatory, so a cell inserted into a 4.5
    notebook without one leaves the file failing schema validation. The id is
    derived from the cell's own source rather than drawn at random, so a
    rebuild of an unchanged notebook produces a byte-identical file.
    """
    if notebook.get("nbformat_minor", 0) < 5:
        return
    taken = {cell["id"] for cell in notebook["cells"] if "id" in cell}
    for cell in notebook["cells"]:
        if "id" in cell:
            continue
        digest = hashlib.sha1(source_of(cell).encode("utf-8")).hexdigest()
        candidate = digest[:8]
        salt = 0
        while candidate in taken:
            salt += 1
            candidate = hashlib.sha1(
                f"{salt}:{source_of(cell)}".encode("utf-8")
            ).hexdigest()[:8]
        cell["id"] = candidate
        taken.add(candidate)


def write_notebook(name: str, notebook: dict) -> None:
    ensure_cell_ids(notebook)
    with (ROOT / name).open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(notebook, stream, ensure_ascii=False, indent=1)
        stream.write("\n")


def source_of(cell: dict) -> str:
    return "".join(cell.get("source", []))


def as_lines(source: str) -> list:
    return source.splitlines(keepends=True)


def already_prepared(notebook: dict) -> bool:
    """True once any cell carries the flag, so a rerun is a no-op.

    The adapters address cells by index, and the salvage steps insert cells, so
    a second pass over an already-prepared notebook would be reading shifted
    positions. Every adapter checks this first.
    """
    return any(
        cell.get("metadata", {}).get(REFERENCE_ONLY) for cell in notebook["cells"]
    )


def mark_reference_only(notebook: dict, name: str, cells: dict) -> None:
    """Flag each listed cell, asserting it is the code cell we expect.

    ``cells`` maps a cell index to a substring that must appear in that cell's
    source. The substring is the whole point: an index alone would silently
    drift the first time an upstream notebook gains or loses a cell, and a
    silent drift here mismarks a runnable cell as unrunnable.
    """
    for index, expected in cells.items():
        cell = notebook["cells"][index]
        source = source_of(cell)
        if cell.get("cell_type") != "code" or expected not in source:
            raise ValueError(
                f"{name}:cell[{index}] is not the expected code cell "
                f"containing {expected!r}"
            )
        cell.setdefault("metadata", {})[REFERENCE_ONLY] = True


def new_code_cell(source: str, *, reference_only: bool = False, tags=None) -> dict:
    metadata: dict = {}
    if tags:
        metadata["tags"] = list(tags)
    if reference_only:
        metadata[REFERENCE_ONLY] = True
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": metadata,
        "outputs": [],
        "source": as_lines(source),
    }


def new_markdown_cell(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": as_lines(source)}


def split_out_imports(
    notebook: dict,
    name: str,
    index: int,
    *,
    moved: str,
    keep: str,
) -> None:
    """Move ``moved`` out of the import cell at ``index`` into its own cell.

    The import that cannot resolve goes last so that everything the rest of the
    notebook needs is already bound when it fails. The new cell is flagged and
    inherits the original's tags, so the built book hides it exactly as it hid
    the lines it came from.
    """
    cell = notebook["cells"][index]
    source = source_of(cell)
    if cell.get("cell_type") != "code" or moved not in source:
        raise ValueError(
            f"{name}:cell[{index}] is not the expected import cell "
            f"containing {moved!r}"
        )
    cell["source"] = as_lines(keep)
    tags = cell.get("metadata", {}).get("tags")
    notebook["cells"].insert(
        index + 1, new_code_cell(moved, reference_only=True, tags=tags)
    )


def adapt_hepatitis() -> bool:
    name = "hepatitis.ipynb"
    notebook = load_notebook(name)
    if already_prepared(notebook):
        return False

    mark_reference_only(
        notebook,
        name,
        {
            4: "import pymc",
            12: "with pm.Model(coords=coords) as model:",
            15: "pm.model_to_graphviz(model)",
            17: "idata = pm.sample(2000",
            18: 'az.summary(idata, var_names=["N", "p", "mu", "k000"])',
            20: "az.plot_posterior(idata,",
            22: 'az.summary(idata, var_names=["N"], hdi_prob=0.9)',
            30: "heterogeneity_model = make_heterogeneity_model()",
            31: "with heterogeneity_model:",
            33: "az.summary(\n    idata_het,",
            34: "az.plot_posterior(idata_het,",
            35: "az.summary(idata_het,",
            37: "az.plot_density(",
            38: "summary_ind = az.summary(",
            41: "idata_s = fit_heterogeneity(scale)",
        },
    )
    split_out_imports(
        notebook,
        name,
        5,
        moved="import pymc as pm\nimport arviz as az",
        keep=(
            "import numpy as np\n"
            "import matplotlib.pyplot as plt\n"
            "import pandas as pd\n"
            "\n"
            "from scipy.stats import norm\n"
            "\n"
            "from utils import decorate, set_pyplot_params"
        ),
    )
    write_notebook(name, notebook)
    return True


def adapt_bread() -> bool:
    name = "bread.ipynb"
    notebook = load_notebook(name)
    if already_prepared(notebook):
        return False

    mark_reference_only(
        notebook,
        name,
        {
            23: "trace = pm.sample()",
            25: "az.summary(trace)",
            28: "plot_posterior(trace, 'mu', mu_true)",
            29: "plot_posterior(trace, 'sigma', sigma_true)",
            30: "plot_posterior(trace, 'k', k_true",
            32: "az.extract(trace)['k']",
        },
    )
    split_out_imports(
        notebook,
        name,
        5,
        moved="import pymc as pm\nimport arviz as az",
        keep=(
            "import numpy as np\n"
            "import matplotlib.pyplot as plt\n"
            "import seaborn as sns\n"
            "\n"
            "from utils import decorate, set_pyplot_params"
        ),
    )
    write_notebook(name, notebook)
    return True


def adapt_redline_pymc() -> bool:
    name = "redline_pymc.ipynb"
    notebook = load_notebook(name)
    if already_prepared(notebook):
        return False

    mark_reference_only(
        notebook,
        name,
        {
            20: "with pm.Model() as model:",
            61: "model = make_model(n_obs=10)",
            62: "graph = pm.model_to_graphviz(model)",
            64: "idata = pm.sample_prior_predictive()",
            65: "gap_times = idata['prior']['gap_time']",
            66: "kde_gap_time = gaussian_kde(gap_times)",
            67: "elapsed_times = idata.prior['elapsed_time']",
            68: "ns = idata.prior_predictive['n']",
            70: "idata = pm.sample()",
            71: "az.summary(idata)",
            72: "az.plot_trace(idata,",
            74: "az.plot_posterior(idata,",
            77: "gap_times = get_sample(idata, 'gap_time')",
            78: "elapsed_times = get_sample(idata, 'elapsed_time')",
            79: "remaining_times = get_sample(idata, 'remaining_time')",
            81: "sns.kdeplot(gap_times, label='gap')",
            83: "make_model = make_model4",
            84: "idata = pm.sample()",
            86: "gap_times = get_sample(idata, 'gap_time')",
            87: "elapsed_times = get_sample(idata, 'elapsed_time')",
            88: "remaining_times = get_sample(idata, 'remaining_time')",
            89: "rates = get_sample(idata, 'rate')",
            91: "np.corrcoef([gap_times,",
            94: "joint_contour(elapsed_times, rates)",
            99: "run_model(make_model_func)",
            104: "plot_cdfs('model1', 'remaining')",
            108: "plot_model_results([1], 'elapsed')",
            109: "plot_model_results([1], 'remaining')",
            112: "plot_model_results(all_models, 'gap')",
            113: "plot_model_results(all_models, 'elapsed')",
            114: "plot_model_results(all_models, 'remaining')",
            115: "plot_model_results(all_models, 'rate')",
            119: "make_figures('gap')",
            120: "make_figures('elapsed')",
            121: "make_figures('remaining')",
            122: "make_figures('rate')",
        },
    )
    split_out_imports(
        notebook,
        name,
        5,
        moved="import pymc as pm",
        keep=(
            "import matplotlib.pyplot as plt\n"
            "import numpy as np\n"
            "import pandas as pd\n"
            "\n"
            "from utils import decorate"
        ),
    )
    write_notebook(name, notebook)
    return True


def adapt_hospital() -> bool:
    name = "hospital.ipynb"
    notebook = load_notebook(name)
    if already_prepared(notebook):
        return False

    mark_reference_only(
        notebook,
        name,
        {
            17: "import pymc3 as pm",
            18: "%time model = make_model()",
            19: "pred = pm.sample_prior_predictive(1000)",
            21: "az.plot_posterior(trace, var_names=['mu', 'sigma'])",
            23: "trace_xs = trace['xs'].transpose()",
            25: "az.plot_posterior(trace_xs[0])",
            30: "compare_cdf(prior_mu, pred['mu'])",
            31: "compare_cdf(prior_sigma, pred['sigma'])",
            59: "pred_xs = pred['xs'].transpose()",
            60: "compare_cdf(marginal_x, pred_xs[0])",
            75: "compare_cdf(marginal_mu, trace['mu'])",
            76: "compare_cdf(marginal_sigma, trace['sigma'])",
            77: "compare_cdf(marginal_x, trace_xs[-1])",
            93: "compare_cdf(marginal_mu, trace['mu'])",
            94: "compare_cdf(marginal_sigma, trace['sigma'])",
            95: "compare_cdf(marginal_x, trace_xs[i])",
            100: "compare_cdf(pmf, trace_xs[i])",
            102: "trace_xs[i].mean()",
        },
    )
    write_notebook(name, notebook)
    return True


def adapt_chap19_v3() -> bool:
    name = "chap19_v3.ipynb"
    notebook = load_notebook(name)
    if already_prepared(notebook):
        return False

    mark_reference_only(
        notebook,
        name,
        {
            25: "import pymc as pm",
            28: "pm.model_to_graphviz(model)",
            31: "idata = pm.sample_prior_predictive(1000)",
            32: "type(idata)",
            35: "sample_prior_pymc = get_values(idata.prior['lam'])",
            38: "plot_cdf(sample_prior_pymc,",
            40: "sample_prior_pred_pymc = get_values(idata.prior['goals'])",
            43: "plot_pred(sample_prior_pred_pymc,",
            46: "with pm.Model() as model:",
            48: "with pm.Model() as model2:",
            50: "idata2 = pm.sample(500, **options)",
            52: "sample_post_pymc = get_values(idata2.posterior['lam'])",
            53: "sample_post_pymc.shape",
            55: "plot_cdf(sample_post_pymc,",
            58: "idata2_pred = pm.sample_posterior_predictive(idata2)",
            60: "sample_post_pred_pymc = get_values(",
            61: "sample_post_pred_pymc.shape",
            65: "plot_pred(sample_post_pred_pymc,",
            86: "with pm.Model() as model3:",
            89: "idata3 = pm.sample(500, **options)",
            91: "az.plot_posterior(idata3,",
            93: "get_values(idata3.posterior['a'].mean())",
            94: "get_values(idata3.posterior['b'].mean())",
            96: "az.plot_posterior(get_values(idata3.posterior['sigma']))",
            111: "with pm.Model() as model4:",
            113: "idata4 = pm.sample(500, **options)",
            115: "get_values(idata4.posterior['b0']).mean()",
            117: "get_values(idata4.posterior['sigma']).mean()",
            119: "means = [get_values(idata4.posterior[name]).mean()",
            121: "cis = [credible_interval(get_values(idata4.posterior[name]))",
            123: "table['Posterior mean'] = np.round(means, 3)",
            128: "with pm.Model() as model5:",
            130: "with pm.Model() as model6:",
            134: "with pm.Model() as model7:",
            143: "with pm.Model() as model8:",
            144: "az.plot_posterior(idata8)",
            151: "with pm.Model() as model9:",
            152: "idata9 = pm.sample(1000)",
            153: "az.summary(idata9)",
            154: "az.plot_posterior(idata9)",
        },
    )
    write_notebook(name, notebook)
    return True


def adapt_chap20() -> bool:
    name = "chap20.ipynb"
    notebook = load_notebook(name)
    if already_prepared(notebook):
        return False

    mark_reference_only(
        notebook,
        name,
        {
            65: "import pymc as pm",
            67: 'yeast_slurry_vol = pm.Normal("yeast slurry vol"',
            69: "dilution_shaker1 = (yeast_slurry_vol /",
            71: 'chamber_vol = pm.Gamma("chamber_vol"',
            73: 'yeast_in_chamber = pm.Poisson("yeast in chamber"',
            75: 'count = pm.Binomial("count"',
            77: "trace = pm.sample(1000, **options)",
            79: "posterior_sample = trace['yeast conc'] / billion",
            82: "prior_sample = pm.sample_prior_predictive(10000, **options)",
            84: "count = prior_sample['count']",
            86: "mask = (count == 49)",
            88: "posterior_sample2 = prior_sample['yeast conc'][mask]",
            90: "cdf_abc = Cdf.from_seq(posterior_sample2)",
            92: "cdf_abc.plot(label='ABC')",
            96: "n = prior_sample['yeast in chamber']",
            100: "likelihood = binom(n, p).pmf(yeast_counted)",
            101: "likelihood.shape",
            103: "plt.plot(n*p, likelihood",
            105: "qs = prior_sample['yeast conc'] / billion",
            107: "posterior_pmf.sort_index(inplace=True)",
            109: "posterior_pmf.make_cdf().plot(label='ABC2')",
        },
    )
    write_notebook(name, notebook)
    return True


# The SEER export the notebook reads is not in the repository, so `read_html`
# raises FileNotFoundError wherever it runs and takes the rest of the analysis
# with it. These are the rows the notebook actually displays -- `head()` of each
# of the two tables -- transcribed in the same shape `clean_table` expects, so
# the cells that follow produce the numbers the surrounding text quotes.
CANCER_TABLE = '''# Five-year survival and case counts by stage at diagnosis, SEER 17.
# The notebook shows the head of each table, so these are those rows.
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

CANCER_MARKOV_NOTE = """QuantEcon compiles its Markov chain with Numba, which has no WebAssembly
build, so the cell above cannot run in a browser. The next cell defines an
equivalent `MarkovChain` in NumPy alone, with the same constructor and the
same `simulate` method, so the rest of the notebook runs either way."""

CANCER_MARKOV = '''class MarkovChain:
    """The part of QuantEcon's MarkovChain this example uses, in NumPy alone."""

    def __init__(self, transition_matrix, state_values):
        self.P = np.asarray(transition_matrix, dtype=float)
        self.state_values = np.asarray(state_values)
        self.state_to_index = {
            state: index for index, state in enumerate(self.state_values)
        }
        # Sample by inverse transform on each row's CDF: one array of draws and
        # a search per step, rather than a categorical draw per step, which is
        # what keeps 10,000 simulated lifetimes to a few seconds.
        self._cdf = self.P.cumsum(axis=1)
        # The last bin has to reach 1.0 exactly, or a draw above the accumulated
        # total would index past the end of the row.
        self._cdf[:, -1] = 1.0

    def simulate(self, ts_length, init):
        state = self.state_to_index.get(init, init)
        draws = np.random.random(ts_length)
        indices = np.empty(ts_length, dtype=int)
        for i in range(ts_length):
            indices[i] = state
            state = int(np.searchsorted(self._cdf[state], draws[i], side="right"))
        return self.state_values[indices]


mc = MarkovChain(P, state_values=states)'''


def adapt_cancer() -> bool:
    name = "cancer.ipynb"
    notebook = load_notebook(name)
    if already_prepared(notebook):
        return False

    mark_reference_only(
        notebook,
        name,
        {
            32: "from quantecon import MarkovChain",
            60: "import pymc as pm",
            61: 'pmx.DiscreteMarkovChain("chain"',
            62: "traj_labels = idx_to_state[traj]",
        },
    )

    table_cell = notebook["cells"][11]
    if "pd.read_html('CanQues Results.html')" not in source_of(table_cell):
        raise ValueError(f"{name}:cell[11] does not read the SEER export")
    table_cell["source"] = as_lines(CANCER_TABLE)

    notebook["cells"].insert(33, new_markdown_cell(CANCER_MARKOV_NOTE))
    notebook["cells"].insert(34, new_code_cell(CANCER_MARKOV))

    write_notebook(name, notebook)
    return True


ADAPTERS = (
    adapt_hepatitis,
    adapt_bread,
    adapt_redline_pymc,
    adapt_hospital,
    adapt_chap19_v3,
    adapt_chap20,
    adapt_cancer,
)

# Where the flagged cells end up, salvage insertions included. Positions rather
# than a count: a count still matches when a flag lands on the wrong cell, and a
# flag on the wrong cell is the failure this catches.
EXPECTED_MARKS = {
    "hepatitis.ipynb": [4, 6, 13, 16, 18, 19, 21, 23, 31, 32, 34, 35, 36, 38, 39, 42],
    "bread.ipynb": [6, 24, 26, 29, 30, 31, 33],
    "redline_pymc.ipynb": [
        6, 21, 62, 63, 65, 66, 67, 68, 69, 71, 72, 73, 75, 78, 79, 80, 82, 84,
        85, 87, 88, 89, 90, 92, 95, 100, 105, 109, 110, 113, 114, 115, 116,
        120, 121, 122, 123,
    ],
    "hospital.ipynb": [
        17, 18, 19, 21, 23, 25, 30, 31, 59, 60, 75, 76, 77, 93, 94, 95, 100, 102,
    ],
    "chap19_v3.ipynb": [
        25, 28, 31, 32, 35, 38, 40, 43, 46, 48, 50, 52, 53, 55, 58, 60, 61, 65,
        86, 89, 91, 93, 94, 96, 111, 113, 115, 117, 119, 121, 123, 128, 130,
        134, 143, 144, 151, 152, 153, 154,
    ],
    "chap20.ipynb": [
        65, 67, 69, 71, 73, 75, 77, 79, 82, 84, 86, 88, 90, 92, 96, 100, 101,
        103, 105, 107, 109,
    ],
    "cancer.ipynb": [32, 62, 63, 64],
}


def validate() -> None:
    for name, expected in EXPECTED_MARKS.items():
        notebook = load_notebook(name)
        marked = [
            index
            for index, cell in enumerate(notebook["cells"])
            if cell.get("metadata", {}).get(REFERENCE_ONLY)
        ]
        if marked != expected:
            raise ValueError(
                f"{name}: cells flagged {REFERENCE_ONLY} are {marked}, "
                f"expected {expected}"
            )
        for index in marked:
            if notebook["cells"][index].get("cell_type") != "code":
                raise ValueError(
                    f"{name}:cell[{index}] is flagged but is not a code cell"
                )

    # The salvaged import cells must keep everything the rest of the notebook
    # needs and hand off only what cannot resolve.
    for name, index, must_keep, must_not in (
        ("hepatitis.ipynb", 5, "from utils import decorate", "import pymc"),
        ("bread.ipynb", 5, "from utils import decorate", "import pymc"),
        ("redline_pymc.ipynb", 5, "from utils import decorate", "import pymc"),
    ):
        source = source_of(load_notebook(name)["cells"][index])
        if must_keep not in source or must_not in source:
            raise ValueError(f"{name}:cell[{index}] is not the split import cell")

    cancer = load_notebook("cancer.ipynb")
    if "pd.read_html" in source_of(cancer["cells"][11]):
        raise ValueError("cancer.ipynb still reads the SEER export from disk")
    if "class MarkovChain" not in source_of(cancer["cells"][34]):
        raise ValueError("cancer.ipynb is missing the NumPy Markov chain")

    # Two upstream corrections live in the canonical notebooks, not here. Assert
    # them so a stale copy in soln/ cannot quietly undo them on the next build.
    chap17 = source_of(load_notebook("chap17.ipynb")["cells"][74])
    if (
        "xlabel=r'$\\sigma$" not in chap17
        or "title=r'Posterior marginal distribution of $\\sigma$'" not in chap17
    ):
        raise ValueError("chap17.ipynb lost the raw LaTeX label strings")
    chap19 = "".join(
        source_of(cell)
        for cell in load_notebook("chap19_v3.ipynb")["cells"]
        if cell.get("cell_type") == "code"
    )
    if "happiness-report.s3.amazonaws.com" in chap19:
        raise ValueError("chap19_v3.ipynb is back on the dead happiness URL")

    for path in sorted(ROOT.glob("*.ipynb")):
        notebook = json.loads(path.read_text(encoding="utf-8"))
        if notebook.get("nbformat") != 4:
            raise ValueError(f"{path.name} is not nbformat 4")
        if notebook.get("nbformat_minor", 0) >= 5:
            missing = [
                index
                for index, cell in enumerate(notebook["cells"])
                if "id" not in cell
            ]
            if missing:
                raise ValueError(f"{path.name}: cells {missing} have no id")

    print("notebook adaptations validated")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="validate an already-prepared directory without changing it",
    )
    args = parser.parse_args()

    if not args.check:
        for adapter in ADAPTERS:
            changed = adapter()
            state = "adapted" if changed else "already prepared"
            print(f"{state}: {adapter.__name__[6:]}")
    validate()


if __name__ == "__main__":
    main()
