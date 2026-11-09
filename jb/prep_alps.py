"""Adapt the built Jupyter Book notebooks for the in-browser Python runtime.

``build.sh`` copies the canonical notebooks out of ``soln/`` and ``examples/``
into this directory and then runs this script, so every adaptation below is
reapplied on each build and the upstream sources stay untouched.

Three kinds of change are made, and no others.

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
    Four notebooks import PyMC in the middle of the same cell that imports
    NumPy, Pandas and ``utils``: the failed import costs them ``decorate`` and
    ``set_pyplot_params``, and every later figure fails with a NameError. Those
    imports are split into a cell of their own so the rest of the cell still
    binds. ``cancer.ipynb`` needs two more, both described where they are
    applied.

Text
    Prose written for the book's other editions -- the print book, the
    upstream Jupyter Book site -- is rewritten where it cannot work in this
    one, such as a cross-reference in the print book's syntax. Each rule is a
    function registered with ``@text_rule`` and says what it rewrites. The
    rules run over every notebook the table of contents lists, on every build,
    after the adapters above: each one matches nothing once it has been
    applied, so a second run changes nothing, and ``--check`` fails when a
    rule would still change something.

Cells that merely *mention* an unavailable package are left alone. A ``def``
whose body calls ``pm.Model`` executes perfectly well -- it just cannot be
called -- and marking it would claim a failure that does not happen.

Run ``python prep_alps.py`` to apply, ``python prep_alps.py --check`` to verify
an already-prepared directory.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
from pathlib import Path
from typing import NamedTuple

ROOT = Path(__file__).resolve().parent

REFERENCE_ONLY = "alps_reference_only"


def load_notebook(name: str, root: Path = ROOT) -> dict:
    with (root / name).open(encoding="utf-8") as stream:
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


def write_notebook(name: str, notebook: dict, root: Path = ROOT) -> None:
    ensure_cell_ids(notebook)
    with (root / name).open("w", encoding="utf-8", newline="\n") as stream:
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
            17: "import pymc as pm",
            18: "model = make_model()",
            19: "with model:",
            21: "post = az.extract(idata)",
            23: "az.plot_dist(idata, var_names=['mu', 'sigma'])",
            25: "post_xs = post['xs']",
            27: "Cdf.from_seq(post_xs[0]).plot()",
            32: "compare_cdf(prior_mu, prior_sample['mu'])",
            33: "compare_cdf(prior_sigma, prior_sample['sigma'])",
            61: "pred_xs = prior_sample['xs']",
            62: "compare_cdf(marginal_x, pred_xs[0])",
            77: "marginal_mu = Pmf(marginal(posterior, 0), mus)",
            78: "marginal_sigma = Pmf(marginal(posterior, 1), sig",
            79: "marginal_x = Pmf(marginal(posterior, 2), xs)",
            95: "compare_cdf(marginal_mu, post['mu'])",
            96: "compare_cdf(marginal_sigma, post['sigma'])",
            97: "compare_cdf(marginal_x, post_xs[i])",
            102: "compare_cdf(pmf, post_xs[i])",
            104: "diff = abs(pmf.mean() - float(post_xs[i].mean())",
        },
    )
    write_notebook(name, notebook)
    return True


def adapt_chap19_v3(name: str = "chap19_v3.ipynb") -> bool:
    notebook = load_notebook(name)
    if already_prepared(notebook):
        return False

    mark_reference_only(
        notebook,
        name,
        {
            26: "import pymc as pm",
            29: "pm.model_to_graphviz(model)",
            32: "with model:",
            33: "type(idata)",
            36: "sample_prior_pymc = get_values(idata.prior['lam'",
            39: "plot_cdf(sample_prior,",
            41: "sample_prior_pred_pymc = get_values(idata.prior[",
            44: "plot_pred(sample_prior_pred,",
            47: "with pm.Model() as model:",
            49: "with pm.Model() as model2:",
            51: "with model2:",
            53: "sample_post_pymc = get_values(idata2.posterior['",
            54: "sample_post_pymc.shape",
            56: "posterior.make_cdf().plot(label='posterior grid'",
            59: "with model2:",
            61: "sample_post_pred_pymc = get_values(idata2_pred.p",
            62: "sample_post_pred_pymc.shape",
            66: "plot_pred(sample_post_pred,",
            87: "with pm.Model() as model3:",
            90: "with model3:",
            92: "with model3:",
            94: "print('Sample mean:', get_values(idata3.posterio",
            95: "print('Sample mean:', get_values(idata3.posterio",
            97: "az.plot_dist(idata3, var_names=['sigma']);",
            112: "with pm.Model() as model4:",
            114: "with model4:",
            116: "get_values(idata4.posterior['b0']).mean()",
            118: "get_values(idata4.posterior['sigma']).mean()",
            120: "means = [get_values(idata4.posterior[name]).mean",
            122: "cis = [credible_interval(get_values(idata4.poste",
            124: "index = columns[1:]",
            129: "with pm.Model() as model5:",
            131: "with pm.Model() as model6:",
            135: "with pm.Model() as model7:",
            144: "with pm.Model() as model8:",
            145: "with model8:",
            152: "with pm.Model() as model9:",
            153: "with model9:",
            154: "az.summary(idata9)",
            155: "with model9:",
        },
    )
    write_notebook(name, notebook)
    return True


def adapt_diabetes() -> bool:
    name = "diabetes.ipynb"
    notebook = load_notebook(name)
    if already_prepared(notebook):
        return False

    mark_reference_only(
        notebook,
        name,
        {
            4: "import pymc as pm",
            13: "with pm.Model(coords=coords) as model:",
            16: "pm.model_to_graphviz(model)",
            18: "with model:",
            19: 'az.summary(idata, var_names=["N", "p", "mu", "k0',
            21: 'az.plot_posterior(idata, var_names=["N", "p"])',
            23: 'az.summary(idata, var_names=["N"], hdi_prob=0.9)',
            29: "heterogeneity_model = make_heterogeneity_model()",
            30: "with heterogeneity_model:",
            32: "az.summary(",
            33: 'az.plot_posterior(idata_het, var_names=["N", "si',
            34: 'az.summary(idata_het, var_names=["N"], hdi_prob=',
            36: "az.plot_density(",
            37: 'summary_ind = az.summary(idata, var_names=["N"],',
            40: "idata_s = fit_heterogeneity(scale)",
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


def adapt_ordinal() -> bool:
    name = "ordinal.ipynb"
    notebook = load_notebook(name)
    if already_prepared(notebook):
        return False

    mark_reference_only(
        notebook,
        name,
        {
            35: "import pymc as pm",
            37: "pm.model_to_graphviz(model)",
            40: "with model:",
            41: "az.summary(idata, var_names=var_names, round_to=",
            43: "az.plot_dist(idata, var_names=['cutpoints', 'bet",
            45: "with model:",
            48: "plot_bands(age_range, predicted, colors)",
            51: "with pm.Model() as model2:",
            52: "with model2:",
            53: "az.summary(idata2, var_names=['cutpoints', 'beta",
        },
    )
    write_notebook(name, notebook)
    return True


def adapt_chap19_v3_soln() -> bool:
    """The teachers' copy, marked as the chapter was before its solutions went."""
    return adapt_chap19_v3("chap19_v3_soln.ipynb")


def adapt_chap20(name: str = "chap20.ipynb") -> bool:
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


def adapt_chap20_soln() -> bool:
    return adapt_chap20("chap20_soln.ipynb")


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


# The text pass. Unlike the adapters it is not keyed to cell positions, so it
# runs on every build, over every notebook the table of contents lists.

TOC_ENTRY = re.compile(
    r"""^\s*(?:-\s+)?(?:root|file):\s*["']?([^"'\s#]+)["']?\s*(?:#.*)?$""",
    re.MULTILINE,
)


def toc_pages(root: Path = ROOT) -> list:
    """The pages ``_toc.yml`` lists, root first, in reading order."""
    return TOC_ENTRY.findall((root / "_toc.yml").read_text(encoding="utf-8"))


TOC_KEY = re.compile(r"""^(\s*)(-\s+)?([A-Za-z_]+):\s*["']?([^"'\s#]*)["']?""")


def toc_audiences(root: Path = ROOT) -> dict:
    """Map each page ``_toc.yml`` gives an ``audience:`` to that audience.

    An entry's keys sit in one column, the column of its ``file:`` key, and
    an entry in a list starts at its ``-``, so an ``audience:`` line belongs
    to the latest entry whose keys start in its column, whether the entry's
    ``file:`` comes before it or after.
    """
    audiences: dict = {}
    entries: list = []  # [column, page, audience], the last two None until read
    for line in (root / "_toc.yml").read_text(encoding="utf-8").split("\n"):
        match = TOC_KEY.match(line)
        if not match:
            continue
        indent, dash, key, value = match.groups()
        column = len(indent) + len(dash or "")
        if dash:
            entries.append([column, None, None])
        entry = next((e for e in reversed(entries) if e[0] == column), None)
        if key in ("file", "root"):
            if entry is None or entry[1] is not None:
                entry = [column, None, None]
                entries.append(entry)
            entry[1] = value
            if entry[2] is not None:
                audiences[value] = entry[2]
        elif key == "audience" and entry is not None:
            entry[2] = value
            if entry[1] is not None:
                audiences[entry[1]] = value
    return audiences


class Section(NamedTuple):
    chapter: int
    level: int
    title: str
    notebook: str


class Book:
    """The pages the table of contents lists, as the text rules see them.

    ``notebooks`` maps each listed notebook's file name to its contents and
    ``markdown_pages`` each listed Markdown page's file name to its text, both
    in reading order. A listed page with no file in ``root`` is left out, so a
    directory holding part of the book still loads.
    """

    def __init__(self, root: Path = ROOT):
        self.root = root
        self.pages = toc_pages(root)
        self.audience = toc_audiences(root)
        self.notebooks: dict = {}
        self.markdown_pages: dict = {}
        for page in self.pages:
            if (root / f"{page}.ipynb").is_file():
                self.notebooks[f"{page}.ipynb"] = load_notebook(f"{page}.ipynb", root)
            elif (root / f"{page}.md").is_file():
                self.markdown_pages[f"{page}.md"] = (root / f"{page}.md").read_text(
                    encoding="utf-8"
                )
        self.sections = section_index(
            {name: nb for name, nb in self.notebooks.items() if not self.for_teachers(name)}
        )

    def for_teachers(self, name: str) -> bool:
        """True for a page listed with ``audience: instructor``."""
        return self.audience.get(name.removesuffix(".ipynb").removesuffix(".md")) == "instructor"

    def markdown_texts(self):
        """Yield ``(where, text)`` for every Markdown cell and Markdown page."""
        for name, notebook in self.notebooks.items():
            for index, cell in enumerate(notebook["cells"]):
                if cell.get("cell_type") == "markdown":
                    yield f"{name}:cell[{index}]", source_of(cell)
        yield from self.markdown_pages.items()


TEXT_RULES: list = []
BOOK_CHECKS: list = []


def text_rule(rule):
    """Register ``rule(book, name, notebook) -> int``.

    The rule edits ``notebook`` in place and returns how many changes it made,
    which must be zero once it has been applied. Rules run in the order they
    are defined.
    """
    TEXT_RULES.append(rule)
    return rule


def book_check(check):
    """Register ``check(book)``, which raises ValueError on a defect.

    For what a rule's own idempotence cannot show: text no rule recognises,
    or a property of the book as a whole.
    """
    BOOK_CHECKS.append(check)
    return check


def apply_text_rules(root: Path = ROOT) -> dict:
    """Apply every text rule to every listed notebook and save the changed ones.

    Returns ``{notebook: {rule: changes}}`` for the notebooks that changed.
    """
    book = Book(root)
    report = {}
    for name, notebook in book.notebooks.items():
        changes = {}
        for rule in TEXT_RULES:
            count = rule(book, name, notebook)
            if count:
                changes[rule.__name__] = count
        if changes:
            write_notebook(name, notebook, root)
            report[name] = changes
    return report


def check_text(root: Path = ROOT) -> None:
    book = Book(root)
    for name, notebook in book.notebooks.items():
        for rule in TEXT_RULES:
            if rule(book, name, copy.deepcopy(notebook)):
                raise ValueError(
                    f"{name}: text rule {rule.__name__} would still change it; "
                    f"run prep_alps.py"
                )
    for check in BOOK_CHECKS:
        check(book)


def markdown_cells(notebook: dict) -> list:
    return [cell for cell in notebook["cells"] if cell.get("cell_type") == "markdown"]


def set_source(cell: dict, source: str) -> None:
    cell["source"] = as_lines(source)


FENCE = re.compile(r" {0,3}(`{3,}|~{3,})(.*)$")


def unfenced_lines(lines: list):
    """Yield ``(index, line)`` for the Markdown lines outside fenced code blocks.

    A fence closes only on a line of the same character, at least as long as
    the one that opened it, with nothing after it. The lines may keep their
    line endings.
    """
    fence = None
    for index, line in enumerate(lines):
        marker = FENCE.match(line)
        if fence is not None:
            if (
                marker
                and marker[1][0] == fence[0]
                and len(marker[1]) >= len(fence)
                and not marker[2].strip()
            ):
                fence = None
        elif marker and (marker[1][0] != "`" or "`" not in marker[2]):
            fence = marker[1]
        else:
            yield index, line


def markdown_lines(source: str):
    """The lines of a Markdown source that lie outside fenced code blocks."""
    for _, line in unfenced_lines(source.split("\n")):
        yield line


HEADING = re.compile(r"(#{1,6})\s+(.*?\S)(?:[ \t]+#+)?\s*$")
CHAPTER = re.compile(r"chap(\d+)(?:_\w+)?\.ipynb")


def section_key(title: str) -> str:
    return re.sub(r"[^0-9A-Za-z]", "", title)


def section_index(notebooks: dict) -> dict:
    """Map each section key to the chapter headings that produce it.

    The print book names a section ``_`` followed by its title with everything
    but letters and digits removed: "Bayes's Rule" is ``_BayessRule``. The
    chapter number is the one in the file name, ``chap19_v3`` being Chapter
    19. Chapters are read in reading order, so each key's list starts with
    the first heading that has that title.
    """
    index: dict = {}
    for name, notebook in notebooks.items():
        chapter = CHAPTER.fullmatch(name)
        if not chapter:
            continue
        for cell in markdown_cells(notebook):
            for line in markdown_lines(source_of(cell)):
                heading = HEADING.match(line)
                if heading:
                    level, title = len(heading.group(1)), heading.group(2)
                    index.setdefault(section_key(title), []).append(
                        Section(int(chapter.group(1)), level, title, name)
                    )
    return index


CROSS_REFERENCE = re.compile(r"<<(_[^<>,]*)>>")


def cross_reference_text(section: Section) -> str:
    link = f"[Chapter {section.chapter}]({section.notebook})"
    if section.level == 1:
        return link
    return f"“{section.title}” in {link}"


@text_rule
def link_cross_references(book: Book, name: str, notebook: dict) -> int:
    """Turn the print book's cross-references into links to their chapter.

    The chapters' Markdown carries AsciiDoc cross-references such as
    ``<<_TheBinomialDistribution>>``, which a notebook renderer prints as they
    stand. A reference to a chapter's title becomes ``[Chapter 4](chap04.ipynb)``
    and a reference to a section becomes ``“The Binomial Distribution” in
    [Chapter 4](chap04.ipynb)``. A title used by more than one heading -- a
    chapter whose first section repeats its title, or a section title that
    recurs in later chapters -- resolves to its first heading in reading order,
    the one an id generator gives the unsuffixed id, and the one each such
    reference in this book means. A reference that names no heading raises
    rather than becoming a link to the wrong page. A reference with its own
    label, ``<<_Id,label>>``, is not rewritten, so --check reports it.
    """

    def replace(match: re.Match) -> str:
        sections = book.sections.get(section_key(match.group(1)))
        if not sections:
            raise ValueError(
                f"{name}: cross-reference {match.group(0)} names no heading "
                f"in the chapters"
            )
        if "<<" in sections[0].title:
            raise ValueError(
                f"{name}: cross-reference {match.group(0)} names a heading that "
                f"holds a cross-reference itself"
            )
        return cross_reference_text(sections[0])

    changes = 0
    for cell in markdown_cells(notebook):
        source = source_of(cell)
        if "<<_" not in source:
            continue
        rewritten, count = CROSS_REFERENCE.subn(replace, source)
        if count:
            set_source(cell, rewritten)
            changes += count
    return changes


@book_check
def no_cross_references_left(book: Book) -> None:
    for where, text in book.markdown_texts():
        if "<<_" in text:
            raise ValueError(f"{where} still has an AsciiDoc cross-reference")


IN_BOOK_LINK = re.compile(r"\]\(([^()\s:/#]+\.ipynb)(?:#[^()\s]*)?\)")


@book_check
def in_book_links_resolve(book: Book) -> None:
    """A link to another notebook by file name must name a page of the book."""
    for where, text in book.markdown_texts():
        for target in IN_BOOK_LINK.findall(text):
            if target not in book.notebooks:
                raise ValueError(f"{where} links to {target}, which is not a page")


WORK_CELL = "# Solution goes here"
HIDING_TAGS = ("hide-cell", "remove-cell", "hide-input", "remove-input")


def teachers_copy(book: Book, name: str):
    """The teachers' copy of a students' page, if the book lists one."""
    stem = name.removesuffix(".ipynb")
    if book.for_teachers(name) or book.audience.get(f"{stem}_soln") != "instructor":
        return None
    return f"{stem}_soln.ipynb"


@text_rule
def scrub_student_solutions(book: Book, name: str, notebook: dict) -> int:
    """Give the students' copy of a page blank work cells for its solutions.

    A page whose teachers' copy (``<page>_soln``) is listed for teachers only
    is the students' page, and gets what upstream's soln/scrub_code.py does to
    make its student notebooks: a code cell that starts with ``# Solution``
    reads ``# Solution goes here``. The work cell loses its outputs and
    execution count, which would give the answer away, and stays visible and
    runnable: no hiding tag, no reference-only flag. Every other cell keeps
    what the teachers' copy has, display tags and stored outputs included,
    which upstream's student notebooks drop.

    A code cell that follows a solution with no Markdown between them goes the
    same way: upstream leaves the last step of a worked answer unlabelled
    (hospital_birth_rate's closing probability), and that cell's source and
    stored output are the answer. An empty code cell holds no answer, so it is
    left alone and does not end the run.
    """
    if teachers_copy(book, name) is None:
        return 0
    changes = 0
    after_solution = False
    for cell in notebook["cells"]:
        if cell.get("cell_type") != "code":
            after_solution = False
            continue
        if not source_of(cell).strip() and not cell.get("outputs"):
            continue
        if not after_solution and not source_of(cell).startswith("# Solution"):
            continue
        after_solution = True
        metadata = cell.setdefault("metadata", {})
        tags = metadata.get("tags", [])
        if (
            source_of(cell) == WORK_CELL
            and not cell.get("outputs")
            and cell.get("execution_count") is None
            and not any(tag in HIDING_TAGS for tag in tags)
            and not metadata.get(REFERENCE_ONLY)
        ):
            continue
        set_source(cell, WORK_CELL)
        cell["outputs"] = []
        cell["execution_count"] = None
        kept = [tag for tag in tags if tag not in HIDING_TAGS]
        if kept:
            metadata["tags"] = kept
        else:
            metadata.pop("tags", None)
        metadata.pop(REFERENCE_ONLY, None)
        changes += 1
    return changes


NEJM_FIGURE = re.compile(
    r'<img\b[^>]*\bsrc="https://www\.nejm\.org/[^"]*/nejmoa1806828_f1\.jpeg"[^>]*>'
)
NEJM_ARTICLE_LINK = (
    "[Figure 1 of the article, on the journal's site]"
    "(https://www.nejm.org/doi/10.1056/NEJMoa1806828)"
)


@text_rule
def link_nejm_figure(book: Book, name: str, notebook: dict) -> int:
    """Link to the ADHD article's figure instead of embedding the journal's image.

    chap16's ADHD exercise embeds Figure 1 of the NEJM article from the
    journal's own servers. The image is the journal's, and its site refuses
    the importer's request for it, so the page showed nothing between "The
    paper includes this figure:" and the paragraphs that discuss the figure.
    A link to the article, where the figure is published, takes its place.
    """
    changes = 0
    for cell in markdown_cells(notebook):
        rewritten, count = NEJM_FIGURE.subn(NEJM_ARTICLE_LINK, source_of(cell))
        if count:
            set_source(cell, rewritten)
            changes += count
    return changes


@book_check
def no_journal_images(book: Book) -> None:
    for where, text in book.markdown_texts():
        if re.search(r'<img\b[^>]*\bsrc="https?://www\.nejm\.org/', text):
            raise ValueError(f"{where} still embeds an image from nejm.org")


ABOUT_PAGE = "about-this-book"


def book_counts(book: Book) -> tuple:
    """The page and code-cell counts About This Book quotes.

    Pages are the table of contents' entries other than About This Book
    itself; code cells are every code cell in the listed notebooks, which is
    what the importer turns into runnable items. Both count what a student
    sees: pages listed for teachers only are left out.
    """
    pages = [
        page for page in book.pages
        if page != ABOUT_PAGE and book.audience.get(page) != "instructor"
    ]
    code = sum(
        1
        for name, notebook in book.notebooks.items()
        if not book.for_teachers(name)
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )
    return len(pages), code


@book_check
def about_page_counts_the_book(book: Book) -> None:
    """About This Book's "At a glance" numbers must be the book's own."""
    pages, code = book_counts(book)
    text = (book.root / f"{ABOUT_PAGE}.md").read_text(encoding="utf-8")
    for expected in (f"**{pages} pages**", f"**{code} code cells**"):
        if expected not in text:
            raise ValueError(
                f"{ABOUT_PAGE}.md does not say {expected}; the book has "
                f"{pages} pages and {code} code cells"
            )


PYMC_NOTE = (
    "> **On Alps:** PyMC does not run in the browser, so this chapter's PyMC "
    "cells are shown for reference and these exercises are for your own computer."
)


@text_rule
def note_pymc_exercises(book: Book, name: str, notebook: dict) -> int:
    """Say before chap19_v3's exercises that they need PyMC.

    Every exercise in the MCMC chapter asks for a PyMC model, and PyMC has no
    WebAssembly build. The note goes under the Exercises heading, in the
    heading's own cell, so no cell moves.
    """
    if name != "chap19_v3.ipynb":
        return 0
    for cell in markdown_cells(notebook):
        source = source_of(cell)
        if source.startswith("## Exercises"):
            if PYMC_NOTE in source:
                return 0
            set_source(cell, f"{source.rstrip()}\n\n{PYMC_NOTE}")
            return 1
    return 0


@book_check
def pymc_exercises_noted(book: Book) -> None:
    chapter = book.notebooks.get("chap19_v3.ipynb")
    if chapter and not any(PYMC_NOTE in source_of(cell) for cell in markdown_cells(chapter)):
        raise ValueError("chap19_v3.ipynb has no Exercises heading to put the PyMC note under")


@text_rule
def drop_trailing_empty_code_cells(book: Book, name: str, notebook: dict) -> int:
    """Remove the empty code cells a notebook ends with.

    Many examples end with the blank cell Jupyter leaves for the next input,
    which the reader shows as an editable cell after the license footer. A
    cell with outputs or a reference-only flag stays, however blank its
    source. --check fails if one remains.
    """
    cells = notebook["cells"]
    changes = 0
    while (
        cells
        and cells[-1].get("cell_type") == "code"
        and not source_of(cells[-1]).strip()
        and not cells[-1].get("outputs")
        and not cells[-1].get("metadata", {}).get(REFERENCE_ONLY)
    ):
        cells.pop()
        changes += 1
    return changes


@text_rule
def hide_solution_cells(book: Book, name: str, notebook: dict) -> int:
    """Collapse every ``# Solution`` code cell, as the chapters' already are.

    Upstream's prep_notebooks.py tags only ``chap*.ipynb``, so an example
    with an exercise showed its solution in full under the exercise. The test
    is upstream's: a code cell whose source starts with ``# Solution``. A
    learner's work cell, which upstream's student copies open with
    ``# Solution goes here``, holds no solution and stays visible.
    """
    changes = 0
    for cell in notebook["cells"]:
        source = source_of(cell)
        if (
            cell.get("cell_type") != "code"
            or not source.startswith("# Solution")
            or source.startswith("# Solution goes here")
        ):
            continue
        metadata = cell.setdefault("metadata", {})
        tags = metadata.get("tags")
        if tags is None:
            # Keep the metadata's keys in the sorted order nbformat writes.
            cell["metadata"] = dict(sorted({**metadata, "tags": ["hide-cell"]}.items()))
        elif "hide-cell" not in tags:
            tags.append("hide-cell")
        else:
            continue
        changes += 1
    return changes


SECTION_HEADING = re.compile(r" {0,3}(#{1,3})[ \t]+(.+)")


@text_rule
def show_output_of_hidden_code(book: Book, name: str, notebook: dict) -> int:
    """Fold only the code of a hidden cell in a chapter's teaching text.

    Upstream tags helper and setup code ``hide-cell``, which folds the cell's
    output along with its code. The prose around these cells discusses what
    they print and plot, so the reader needs the output after Run All:
    ``hide-input`` folds the code and leaves the output showing.

    A ``# Solution`` cell stays ``hide-cell``, and so does every cell of an
    Exercises section: some of those are unlabelled steps of a worked answer,
    and their output gives it away. A heading is read from any unfenced line
    of a Markdown cell. A level 1 or 2 heading starts or ends an Exercises
    section by its title; a level 3 heading can start one, as in "### Exercise
    1", but never ends one, so a subsection of the exercises stays protected.
    """
    changes = 0
    in_exercises = False
    for cell in notebook["cells"]:
        source = source_of(cell)
        if cell.get("cell_type") == "markdown":
            for line in markdown_lines(source):
                heading = SECTION_HEADING.match(line)
                if not heading:
                    continue
                is_exercise = heading.group(2).lower().startswith("exercise")
                if len(heading.group(1)) <= 2 or is_exercise:
                    in_exercises = is_exercise
            continue
        tags = cell.get("metadata", {}).get("tags") or []
        if (
            cell.get("cell_type") != "code"
            or in_exercises
            or "hide-cell" not in tags
            or source.startswith("# Solution")
        ):
            continue
        if "hide-input" in tags:
            tags.remove("hide-cell")
        else:
            tags[tags.index("hide-cell")] = "hide-input"
        changes += 1
    return changes


ADAPTERS = (
    adapt_hepatitis,
    adapt_bread,
    adapt_redline_pymc,
    adapt_hospital,
    adapt_chap19_v3,
    adapt_chap19_v3_soln,
    adapt_chap20,
    adapt_chap20_soln,
    adapt_cancer,
    adapt_diabetes,
    adapt_ordinal,
)

# Where the flagged cells end up, salvage insertions included. Positions rather
# than a count: a count still matches when a flag lands on the wrong cell, and a
# flag on the wrong cell is the failure this catches.
# Imports no WebAssembly build can satisfy: each reaches Numba, which compiles
# through LLVM. A cell containing one of these cannot execute in the browser
# whatever its position in the notebook.
UNAVAILABLE_IMPORTS = (
    "import pymc",
    "import pymc3",
    "import arviz",
    "import pymc_extras",
    "from quantecon import",
    "import quantecon",
)

EXPECTED_MARKS = {
    "hepatitis.ipynb": [4, 6, 13, 16, 18, 19, 21, 23, 31, 32, 34, 35, 36, 38, 39, 42],
    "bread.ipynb": [6, 24, 26, 29, 30, 31, 33],
    "redline_pymc.ipynb": [
        6, 21, 62, 63, 65, 66, 67, 68, 69, 71, 72, 73, 75, 78, 79, 80, 82, 84,
        85, 87, 88, 89, 90, 92, 95, 100, 105, 109, 110, 113, 114, 115, 116,
        120, 121, 122, 123,
    ],
    "hospital.ipynb": [
        17, 18, 19, 21, 23, 25, 27, 32, 33, 61, 62, 77, 78, 79, 95, 96, 97, 102,
        104,
    ],
    # The students' copy: its nine PyMC solutions are work cells now.
    "chap19_v3.ipynb": [
        26, 29, 32, 33, 36, 39, 41, 44, 47, 49, 51, 53, 54, 56, 59, 61, 62, 66,
        87, 90, 92, 94, 95, 97, 112, 114, 116, 118, 120, 122, 124,
    ],
    "chap19_v3_soln.ipynb": [
        26, 29, 32, 33, 36, 39, 41, 44, 47, 49, 51, 53, 54, 56, 59, 61, 62, 66,
        87, 90, 92, 94, 95, 97, 112, 114, 116, 118, 120, 122, 124, 129, 131,
        135, 144, 145, 152, 153, 154, 155,
    ],
    "chap20.ipynb": [
        65, 67, 69, 71, 73, 75, 77, 79, 82, 84, 86, 88, 90, 92, 96, 100, 101,
        103, 105, 107, 109,
    ],
    "chap20_soln.ipynb": [
        65, 67, 69, 71, 73, 75, 77, 79, 82, 84, 86, 88, 90, 92, 96, 100, 101,
        103, 105, 107, 109,
    ],
    "cancer.ipynb": [32, 62, 63, 64],
    "diabetes.ipynb": [4, 6, 14, 17, 19, 20, 22, 24, 30, 31, 33, 34, 35, 37, 38, 41],
    "ordinal.ipynb": [35, 37, 40, 41, 43, 45, 48, 51, 52, 53],
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
        ("diabetes.ipynb", 5, "from utils import decorate", "import pymc"),
    ):
        source = source_of(load_notebook(name)["cells"][index])
        if must_keep not in source or must_not in source:
            raise ValueError(f"{name}:cell[{index}] is not the split import cell")

    cancer = load_notebook("cancer.ipynb")
    if "pd.read_html" in source_of(cancer["cells"][11]):
        raise ValueError("cancer.ipynb still reads the SEER export from disk")

    # Positions alone do not catch an upstream edit that puts one of these
    # imports into a cell that was runnable before: the cell count is
    # unchanged, so the expected marks still match while the build ships a
    # cell that cannot run. Read what each cell imports as well as where it is.
    for name in EXPECTED_MARKS:
        notebook = load_notebook(name)
        for index, cell in enumerate(notebook["cells"]):
            if cell.get("cell_type") != "code":
                continue
            source = source_of(cell)
            needs = next((p for p in UNAVAILABLE_IMPORTS if p in source), None)
            if needs and not cell.get("metadata", {}).get(REFERENCE_ONLY):
                raise ValueError(
                    f"{name}:cell[{index}] uses {needs!r}, which this runtime "
                    f"cannot install, but is not flagged {REFERENCE_ONLY}"
                )
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

    check_text()

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
        # After the adapters: they address cells by position in the copies
        # build.sh makes, and a text rule may remove a cell.
        for name, changes in apply_text_rules().items():
            summary = ", ".join(f"{rule} {count}" for rule, count in changes.items())
            print(f"rewrote: {name} ({summary})")
    validate()


if __name__ == "__main__":
    main()
