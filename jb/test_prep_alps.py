"""Tests for prep_alps.py's text pass.

Run from this directory: ``python -m unittest test_prep_alps``. The tests that
read ``../soln`` and ``../examples`` copy the notebooks exactly as build.sh
does, so they exercise the rules on the text a build starts from; the tests
that read this directory check the notebooks as committed.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import prep_alps

JB = Path(__file__).resolve().parent


def md(source: str, **metadata) -> dict:
    return {"cell_type": "markdown", "metadata": metadata, "source": prep_alps.as_lines(source)}


def code(source: str, outputs=(), **metadata) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": metadata,
        "outputs": list(outputs),
        "source": prep_alps.as_lines(source),
    }


def write_book(root: Path, notebooks: dict, markdown_pages: dict = None) -> None:
    """Write a small book: a table of contents listing the pages in order."""
    markdown_pages = markdown_pages or {}
    pages = [name.rsplit(".", 1)[0] for name in [*markdown_pages, *notebooks]]
    toc = f"format: jb-book\nroot: {pages[0]}\nchapters:\n"
    toc += "".join(f"- file: {page}\n" for page in pages[1:])
    (root / "_toc.yml").write_text(toc, encoding="utf-8")
    for name, text in markdown_pages.items():
        (root / name).write_text(text, encoding="utf-8")
    for name, cells in notebooks.items():
        notebook = {"cells": cells, "metadata": {}, "nbformat": 4, "nbformat_minor": 4}
        prep_alps.write_notebook(name, notebook, root)


def copy_build_sources(root: Path) -> None:
    """Copy what build.sh copies, from the same places, into ``root``."""
    shutil.copy(JB / "_toc.yml", root)
    for page in prep_alps.toc_pages(JB):
        if (JB / f"{page}.md").is_file():
            shutil.copy(JB / f"{page}.md", root)
    # Run build.sh's own copy block, its ../ paths pointed at this repository.
    build = (JB / "build.sh").read_text(encoding="utf-8")
    block = build[build.index("# copy notebooks"):build.index("# add tags")]
    subprocess.run(
        ["bash", "-c", block.replace("../", f"{JB.parent}/")],
        cwd=root, check=True,
    )


def copy_built_book(root: Path) -> None:
    """Copy this directory's table of contents and the pages it lists."""
    shutil.copy(JB / "_toc.yml", root)
    for page in prep_alps.toc_pages(JB):
        for suffix in (".ipynb", ".md"):
            if (JB / f"{page}{suffix}").is_file():
                shutil.copy(JB / f"{page}{suffix}", root)


def snapshot(root: Path) -> dict:
    return {path.name: path.read_bytes() for path in sorted(root.iterdir())}


def cell_text(root: Path, name: str, index: int) -> str:
    notebook = json.loads((root / name).read_text(encoding="utf-8"))
    return prep_alps.source_of(notebook["cells"][index])


class TempBook(unittest.TestCase):
    def setUp(self):
        self._temp = tempfile.TemporaryDirectory()
        self.root = Path(self._temp.name)

    def tearDown(self):
        self._temp.cleanup()


class CrossReferenceTest(TempBook):
    def write_chapters(self, reference: str) -> None:
        write_book(
            self.root,
            {
                "chap01.ipynb": [md("# Probability\n\n## Probability"), md("## Marginal Distributions")],
                "chap02.ipynb": [
                    md("# Bayes's Theorem"),
                    md("```\n## Not A Heading\n```"),
                    md("## The Cookie Problem"),
                    md("## Marginal Distributions"),
                ],
                "chap19_v3.ipynb": [md("# MCMC"), md(reference)],
            },
        )

    def test_chapter_and_section_references_become_links(self):
        self.write_chapters("From <<_Probability>> and <<_TheCookieProblem>>.")
        report = prep_alps.apply_text_rules(self.root)
        self.assertEqual(report, {"chap19_v3.ipynb": {"link_cross_references": 2}})
        self.assertEqual(
            cell_text(self.root, "chap19_v3.ipynb", 1),
            "From [Chapter 1](chap01.ipynb) and "
            "“The Cookie Problem” in [Chapter 2](chap02.ipynb).",
        )

    def test_repeated_title_resolves_to_its_first_heading(self):
        self.write_chapters("Which we saw in <<_MarginalDistributions>>.")
        prep_alps.apply_text_rules(self.root)
        self.assertEqual(
            cell_text(self.root, "chap19_v3.ipynb", 1),
            "Which we saw in “Marginal Distributions” in [Chapter 1](chap01.ipynb).",
        )

    def test_chapter_number_comes_from_the_file_name(self):
        self.write_chapters("See <<_MCMC>>.")
        prep_alps.apply_text_rules(self.root)
        self.assertEqual(
            cell_text(self.root, "chap19_v3.ipynb", 1), "See [Chapter 19](chap19_v3.ipynb)."
        )

    def test_unknown_or_fenced_target_raises(self):
        for reference in ("See <<_NoSuchSection>>.", "See <<_NotAHeading>>."):
            with self.subTest(reference=reference):
                self.write_chapters(reference)
                with self.assertRaisesRegex(ValueError, "names no heading"):
                    prep_alps.apply_text_rules(self.root)

    def test_fences_and_closing_hashes(self):
        write_book(
            self.root,
            {
                "chap01.ipynb": [md("# One\n\n````\n```\n## Fenced\n````\n\n## Closed ##\n\n~~~\n## Tilde\n```\n~~~")],
                "chap02.ipynb": [md("# Two"), md("See <<_Closed>>.")],
            },
        )
        prep_alps.apply_text_rules(self.root)
        self.assertEqual(cell_text(self.root, "chap02.ipynb", 1), "See “Closed” in [Chapter 1](chap01.ipynb).")
        sections = prep_alps.Book(self.root).sections
        self.assertNotIn("Fenced", sections)
        self.assertNotIn("Tilde", sections)

    def test_labelled_reference_is_left_for_check_to_report(self):
        self.write_chapters("See <<_Probability,the first chapter>>.")
        self.assertEqual(prep_alps.apply_text_rules(self.root), {})
        with self.assertRaisesRegex(ValueError, "still has an AsciiDoc cross-reference"):
            prep_alps.no_cross_references_left(prep_alps.Book(self.root))

    def test_heading_that_holds_a_reference_raises(self):
        write_book(
            self.root,
            {"chap01.ipynb": [md("# One\n\n## See <<_Other>>\n\n## Other"), md("Go to <<_SeeOther>>.")]},
        )
        with self.assertRaisesRegex(ValueError, "holds a cross-reference itself"):
            prep_alps.apply_text_rules(self.root)

    def test_second_pass_changes_nothing(self):
        self.write_chapters("From <<_Probability>> and <<_TheCookieProblem>>.")
        prep_alps.apply_text_rules(self.root)
        before = snapshot(self.root)
        self.assertEqual(prep_alps.apply_text_rules(self.root), {})
        self.assertEqual(snapshot(self.root), before)

    def test_upstream_chapters_lose_every_cross_reference(self):
        copy_build_sources(self.root)
        report = prep_alps.apply_text_rules(self.root)
        counts = {
            name: changes["link_cross_references"]
            for name, changes in report.items()
            if "link_cross_references" in changes and not name.endswith("_soln.ipynb")
        }
        teachers = {
            name.replace("_soln", ""): changes["link_cross_references"]
            for name, changes in report.items()
            if "link_cross_references" in changes and name.endswith("_soln.ipynb")
        }
        self.assertEqual(teachers, counts)
        self.assertEqual(
            counts,
            {
                "chap03.ipynb": 1, "chap04.ipynb": 2, "chap06.ipynb": 2,
                "chap07.ipynb": 1, "chap08.ipynb": 4, "chap10.ipynb": 7,
                "chap13.ipynb": 2, "chap14.ipynb": 1, "chap15.ipynb": 1,
                "chap16.ipynb": 1, "chap17.ipynb": 3, "chap18.ipynb": 4,
                "chap19_v3.ipynb": 8, "chap20.ipynb": 1,
            },
        )
        book = prep_alps.Book(self.root)
        self.assertFalse([where for where, text in book.markdown_texts() if "<<_" in text])
        self.assertIn(
            "which we defined in “The Binomial Distribution” in [Chapter 4](chap04.ipynb), "
            "to make a `Pmf`",
            cell_text(self.root, "chap06.ipynb", 66),
        )
        self.assertIn(
            "the Elvis problem from [Chapter 3](chap03.ipynb):",
            cell_text(self.root, "chap06.ipynb", 94),
        )
        self.assertIn(
            "which we saw in “Marginal Distributions” in [Chapter 11](chap11.ipynb)",
            cell_text(self.root, "chap13.ipynb", 43),
        )


class NejmFigureTest(TempBook):
    LINKED = (
        "The paper includes this figure:\n\n"
        "[Figure 1 of the article, on the journal's site]"
        "(https://www.nejm.org/doi/10.1056/NEJMoa1806828)\n\n"
        "In my opinion, this representation of the data"
    )

    def test_upstream_figure_becomes_a_link_to_the_article(self):
        copy_build_sources(self.root)
        self.assertIn("nejmoa1806828_f1.jpeg", cell_text(self.root, "chap16.ipynb", 148))
        report = prep_alps.apply_text_rules(self.root)
        self.assertEqual(report["chap16.ipynb"].get("link_nejm_figure"), 1)
        text = cell_text(self.root, "chap16.ipynb", 148)
        self.assertIn(self.LINKED, text)
        self.assertNotIn("<img", text)

    def test_committed_chapter_links_the_article(self):
        text = cell_text(JB, "chap16.ipynb", 148)
        self.assertIn(self.LINKED, text)
        self.assertNotIn("<img", text)


def colab_links(book: prep_alps.Book) -> dict:
    """How many Colab links each listed page's Markdown has, for those with any."""
    counts: dict = {}
    for where, text in book.markdown_texts():
        page = where.split(":")[0]
        counts[page] = counts.get(page, 0) + text.count("colab.research.google.com")
    return {page: count for page, count in counts.items() if count}


def page_text(book: prep_alps.Book, name: str) -> str:
    return "\n".join(text for where, text in book.markdown_texts() if where.split(":")[0] == name)


class ColabTest(TempBook):
    """The Colab links upstream wrote stay on the page."""

    LINK = "[Click here to run this notebook on Colab](https://colab.research.google.com/github/x/y.ipynb)"

    def test_text_rules_leave_colab_links_alone(self):
        cells = [
            md(f"# Title\n\n{self.LINK}\n\nFirst paragraph."),
            md(self.LINK),
            md(f"Or you can {self.LINK}, in a sentence."),
        ]
        write_book(self.root, {"example.ipynb": cells})
        before = snapshot(self.root)
        self.assertEqual(prep_alps.apply_text_rules(self.root), {})
        self.assertEqual(snapshot(self.root), before)

    def test_upstream_pages_keep_every_colab_link(self):
        copy_build_sources(self.root)
        before = colab_links(prep_alps.Book(self.root))
        self.assertGreater(sum(before.values()), 0)
        prep_alps.apply_text_rules(self.root)
        book = prep_alps.Book(self.root)
        self.assertEqual(colab_links(book), before)
        prep_alps.in_book_links_resolve(book)
        self.assertIn(
            "My analysis of this problem is in `redline.ipynb`, which is in the repository for "
            "this book.  [Click here to run this notebook on Colab]"
            "(https://colab.research.google.com/github/AllenDowney/ThinkBayes2/blob/master/"
            "notebooks/redline.ipynb).",
            page_text(book, "chap09.ipynb"),
        )
        self.assertIn(
            "[click here to run my analysis in a Jupyter notebook on Colab]",
            page_text(book, "cancer.ipynb"),
        )
        self.assertIn("[run that chapter on Colab]", page_text(book, "redline_pymc.ipynb"))
        for name in ("hospital_birth_rate.ipynb", "hospital_birth_rate_soln.ipynb"):
            self.assertIn(
                "[Run this notebook on Colab](https://colab.research.google.com/github/"
                "AllenDowney/ThinkBayes2/blob/master/examples/hospital_birth_rate.ipynb)",
                page_text(book, name),
            )

    def test_committed_pages_keep_every_colab_link(self):
        copy_build_sources(self.root)
        self.assertEqual(colab_links(prep_alps.Book(JB)), colab_links(prep_alps.Book(self.root)))

    def test_landing_page_links_every_example_in_the_book(self):
        index = (JB / "index.md").read_text(encoding="utf-8")
        self.assertNotIn("links on the left", index)
        self.assertNotIn("## Run the notebooks", index)
        examples = index[index.index("## Examples"):]
        audiences = prep_alps.toc_audiences(JB)
        toc = [page for page in prep_alps.toc_pages(JB) if audiences.get(page) != "instructor"]
        listed = re.findall(r"\]\((\w+)\.ipynb\)", examples)
        self.assertEqual(listed, toc[toc.index("redline"):toc.index("ordinal") + 1])

    def test_preface_says_the_code_runs_on_the_page(self):
        preface = (JB / "preface.md").read_text(encoding="utf-8")
        self.assertIn("## Working with the code\n\n> **On Alps:** the code runs in this page", preface)


class AboutCountsTest(TempBook):
    def write_about(self, pages: int, code_cells: int) -> None:
        write_book(
            self.root,
            {"chap01.ipynb": [md("# One"), code("1"), code("")], "chap02.ipynb": [code("2")]},
            {"index.md": "# Book\n"},
        )
        toc = self.root / "_toc.yml"
        toc.write_text(toc.read_text() + "- file: about-this-book\n", encoding="utf-8")
        (self.root / "about-this-book.md").write_text(
            f"| **{pages} pages** | |\n| **{code_cells} code cells** | |\n", encoding="utf-8"
        )

    def test_counts_every_page_but_about_and_every_code_cell(self):
        self.write_about(3, 3)
        book = prep_alps.Book(self.root)
        self.assertEqual(prep_alps.book_counts(book), (3, 3))
        prep_alps.about_page_counts_the_book(book)

    def test_stale_counts_fail(self):
        for pages, code_cells in ((2, 3), (3, 2)):
            with self.subTest(pages=pages, code_cells=code_cells):
                self.write_about(pages, code_cells)
                with self.assertRaisesRegex(ValueError, "the book has 3 pages and 3 code cells"):
                    prep_alps.about_page_counts_the_book(prep_alps.Book(self.root))

    def test_committed_about_page_counts_the_book(self):
        book = prep_alps.Book(JB)
        pages, code_cells = prep_alps.book_counts(book)
        about = (JB / "about-this-book.md").read_text(encoding="utf-8")
        self.assertIn(f"**{pages} pages**", about)
        self.assertIn(f"**{code_cells} code cells**", about)


class PresentationTest(TempBook):
    def test_trailing_empty_code_cells_go(self):
        output = {"output_type": "stream", "name": "stdout", "text": ["x\n"]}
        write_book(
            self.root,
            {
                "a.ipynb": [md("# A"), code(""), code("1"), code(""), code("  \n")],
                "b.ipynb": [md("# B"), code("", outputs=[output])],
                "c.ipynb": [md("# C"), code("", alps_reference_only=True)],
                "d.ipynb": [md("# D"), code(""), md("")],
            },
        )
        kept = {name: (self.root / name).read_bytes() for name in ("b.ipynb", "c.ipynb", "d.ipynb")}
        self.assertEqual(
            prep_alps.apply_text_rules(self.root), {"a.ipynb": {"drop_trailing_empty_code_cells": 2}}
        )
        self.assertEqual({name: (self.root / name).read_bytes() for name in kept}, kept)
        notebook = json.loads((self.root / "a.ipynb").read_text(encoding="utf-8"))
        self.assertEqual([prep_alps.source_of(cell) for cell in notebook["cells"]], ["# A", "", "1"])
        self.assertEqual(prep_alps.apply_text_rules(self.root), {})

    def test_hidden_code_that_is_not_a_solution_keeps_its_output(self):
        write_book(
            self.root,
            {
                "example.ipynb": [
                    md("# Prose", tags=["hide-cell"]),
                    code("import numpy as np", tags=["hide-cell"]),
                    code("x = 1", tags=["remove-output", "hide-cell"]),
                    code("y = 2", tags=["hide-cell", "hide-input"]),
                    code("# Solution\nz = 3", tags=["hide-cell"]),
                    code("w = 4", tags=["hide-input"]),
                    code("v = 5"),
                ]
            },
        )
        report = prep_alps.apply_text_rules(self.root)
        self.assertEqual(report, {"example.ipynb": {"show_output_of_hidden_code": 3}})
        notebook = json.loads((self.root / "example.ipynb").read_text(encoding="utf-8"))
        self.assertEqual(
            [cell["metadata"].get("tags") for cell in notebook["cells"]],
            [["hide-cell"], ["hide-input"], ["remove-output", "hide-input"], ["hide-input"],
             ["hide-cell"], ["hide-input"], None],
        )
        self.assertEqual(prep_alps.apply_text_rules(self.root), {})

    def test_hidden_code_in_an_exercises_section_keeps_its_output_folded(self):
        write_book(
            self.root,
            {
                "example.ipynb": [
                    code("teaching = 1", tags=["hide-cell"]),
                    md("## Exercises"),
                    code("results.params", tags=["hide-cell"]),
                    md("### A step of the exercise"),
                    code("results.resid.std()", tags=["hide-cell"]),
                    md("## Summary"),
                    code("after = 2", tags=["hide-cell"]),
                ]
            },
        )
        report = prep_alps.apply_text_rules(self.root)
        self.assertEqual(report, {"example.ipynb": {"show_output_of_hidden_code": 2}})
        notebook = json.loads((self.root / "example.ipynb").read_text(encoding="utf-8"))
        self.assertEqual(
            [cell["metadata"].get("tags") for cell in notebook["cells"]],
            [["hide-input"], None, ["hide-cell"], None, ["hide-cell"], None, ["hide-input"]],
        )
        self.assertEqual(prep_alps.apply_text_rules(self.root), {})

    def tags_after_rules(self, *cells) -> list:
        write_book(self.root, {"example.ipynb": list(cells)})
        prep_alps.apply_text_rules(self.root)
        self.assertEqual(prep_alps.apply_text_rules(self.root), {})
        notebook = json.loads((self.root / "example.ipynb").read_text(encoding="utf-8"))
        return [cell["metadata"].get("tags") for cell in notebook["cells"] if cell["cell_type"] == "code"]

    def test_exercises_heading_after_prose_in_the_same_cell_counts(self):
        self.assertEqual(
            self.tags_after_rules(
                code("before = 1", tags=["hide-cell"]),
                md("Some closing prose.\n\n## Exercises"),
                code("answer = 2", tags=["hide-cell"]),
            ),
            [["hide-input"], ["hide-cell"]],
        )

    def test_level_three_exercise_heading_counts(self):
        self.assertEqual(
            self.tags_after_rules(md("### Exercise 1"), code("answer = 2", tags=["hide-cell"])),
            [["hide-cell"]],
        )

    def test_level_three_heading_does_not_end_an_exercises_section(self):
        self.assertEqual(
            self.tags_after_rules(
                md("## Exercises"),
                code("answer = 1", tags=["hide-cell"]),
                md("### A worked step"),
                code("answer = 2", tags=["hide-cell"]),
            ),
            [["hide-cell"], ["hide-cell"]],
        )

    def test_level_three_heading_outside_exercises_starts_nothing(self):
        self.assertEqual(
            self.tags_after_rules(md("### A subsection"), code("teaching = 1", tags=["hide-cell"])),
            [["hide-input"]],
        )

    def test_later_level_two_heading_ends_the_section(self):
        self.assertEqual(
            self.tags_after_rules(
                md("## Exercises"),
                code("answer = 1", tags=["hide-cell"]),
                md("Closing prose.\n\n## Summary"),
                code("teaching = 2", tags=["hide-cell"]),
            ),
            [["hide-cell"], ["hide-input"]],
        )

    def test_headings_inside_a_code_fence_are_ignored(self):
        self.assertEqual(
            self.tags_after_rules(
                md("```\n## Exercises\n```"),
                code("teaching = 1", tags=["hide-cell"]),
                md("## Exercises"),
                code("answer = 2", tags=["hide-cell"]),
                md("```\n## Summary\n```"),
                code("answer = 3", tags=["hide-cell"]),
            ),
            [["hide-input"], ["hide-cell"], ["hide-cell"]],
        )

    def test_solution_cells_are_hidden(self):
        write_book(
            self.root,
            {
                "example.ipynb": [
                    md("# Solution"),
                    code("# Solution\n\nx = 1", z=True),
                    code("# Solution\ny = 2", tags=["remove-output"]),
                    code("# Solution\nz = 3", tags=["hide-cell"]),
                    code("w = 4  # Solution"),
                    code("# Solution goes here"),
                ]
            },
        )
        report = prep_alps.apply_text_rules(self.root)
        self.assertEqual(report, {"example.ipynb": {"hide_solution_cells": 2}})
        notebook = json.loads((self.root / "example.ipynb").read_text(encoding="utf-8"))
        self.assertEqual(
            [cell["metadata"] for cell in notebook["cells"]],
            [{}, {"tags": ["hide-cell"], "z": True}, {"tags": ["remove-output", "hide-cell"]},
             {"tags": ["hide-cell"]}, {}, {}],
        )
        self.assertEqual(list(notebook["cells"][1]["metadata"]), ["tags", "z"])
        self.assertEqual(prep_alps.apply_text_rules(self.root), {})

    def test_pymc_note_goes_under_the_exercises_heading(self):
        write_book(self.root, {"chap19_v3.ipynb": [md("# MCMC"), md("## Exercises"), md("**Exercise:** x")]})
        with self.assertRaisesRegex(ValueError, "PyMC note"):
            prep_alps.pymc_exercises_noted(prep_alps.Book(self.root))
        self.assertEqual(
            prep_alps.apply_text_rules(self.root), {"chap19_v3.ipynb": {"note_pymc_exercises": 1}}
        )
        self.assertEqual(
            cell_text(self.root, "chap19_v3.ipynb", 1), f"## Exercises\n\n{prep_alps.PYMC_NOTE}"
        )
        self.assertEqual(prep_alps.apply_text_rules(self.root), {})
        prep_alps.pymc_exercises_noted(prep_alps.Book(self.root))
        write_book(self.root, {"chap19_v3.ipynb": [md("# MCMC")]})
        self.assertEqual(prep_alps.apply_text_rules(self.root), {})
        with self.assertRaisesRegex(ValueError, "no Exercises heading"):
            prep_alps.pymc_exercises_noted(prep_alps.Book(self.root))

    def test_upstream_examples(self):
        copy_build_sources(self.root)
        report = prep_alps.apply_text_rules(self.root)
        self.assertEqual(
            sorted(name for name, changes in report.items() if "drop_trailing_empty_code_cells" in changes),
            sorted(
                f"{name}.ipynb"
                for name in (
                    "redline", "redline_pymc", "vaccine2", "usb", "sister", "bayes_dice",
                    "radiation", "ok", "bookstore", "beta_binomial", "zipf", "cancer", "frog",
                )
            ),
        )
        self.assertEqual(report["hospital_birth_rate.ipynb"]["scrub_student_solutions"], 8)
        self.assertNotIn("hide_solution_cells", report["hospital_birth_rate.ipynb"])
        self.assertEqual(report["hospital_birth_rate_soln.ipynb"]["hide_solution_cells"], 7)
        self.assertEqual(report["chap19_v3.ipynb"]["note_pymc_exercises"], 1)

    def test_committed_book(self):
        book = prep_alps.Book(JB)
        for name, notebook in book.notebooks.items():
            last = notebook["cells"][-1]
            self.assertFalse(
                last["cell_type"] == "code"
                and not prep_alps.source_of(last).strip()
                and not last.get("outputs")
                and not last.get("metadata", {}).get(prep_alps.REFERENCE_ONLY),
                name,
            )
            for index, cell in enumerate(notebook["cells"]):
                source = prep_alps.source_of(cell)
                if (
                    cell["cell_type"] == "code"
                    and source.startswith("# Solution")
                    and not source.startswith("# Solution goes here")
                ):
                    self.assertIn("hide-cell", cell["metadata"].get("tags", []), f"{name}:{index}")
        self.assertEqual(sum(text.count(prep_alps.PYMC_NOTE) for _, text in book.markdown_texts()), 1)
        toc = (JB / "_toc.yml").read_text(encoding="utf-8")
        self.assertIn("- file: redline_pymc\n    title: The Red Line Problem with PyMC\n", toc)


class BuiltBookTest(unittest.TestCase):
    def test_every_page_is_a_top_level_unit(self):
        # Alps imports a captioned part as a unit of its own, a level above
        # its chapters; caption-less parts put every page at the top level.
        toc = (JB / "_toc.yml").read_text(encoding="utf-8")
        self.assertIn("parts:", toc)
        self.assertNotRegex(toc, r"(?m)^\s*-?\s*caption:")

    def test_check_passes(self):
        result = subprocess.run(
            [sys.executable, "prep_alps.py", "--check"],
            cwd=JB, capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_text_rules_change_nothing_more(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            copy_built_book(root)
            before = snapshot(root)
            self.assertEqual(prep_alps.apply_text_rules(root), {})
            self.assertEqual(snapshot(root), before)

    def test_no_cross_references_left(self):
        book = prep_alps.Book(JB)
        self.assertEqual([where for where, text in book.markdown_texts() if "<<_" in text], [])


class StudentEditionTest(TempBook):
    """The students' pages carry work cells; the teachers' copies, the solutions."""

    TOC = (
        "format: jb-book\nroot: index\nparts:\n- caption: Chapters\n  chapters:\n"
        "  - file: chap01\n    sections:\n    - file: chap01_soln\n"
        '      title: "Solution: One"\n      audience: instructor\n'
        "  - file: chap02\n"
    )

    def write(self) -> None:
        output = {"output_type": "stream", "name": "stdout", "text": ["42\n"]}
        solution = code("# Solution\n\nx = 42\nx", outputs=[output], tags=["hide-cell"])
        solution["execution_count"] = 7
        setup = code("import numpy as np", outputs=[output], tags=["hide-cell"])
        cells = [md("# One"), setup, md("**Exercise:** find x."), solution, md("Next."), code("y = 1")]
        write_book(self.root, {"chap01.ipynb": cells, "chap01_soln.ipynb": cells, "chap02.ipynb": [md("# Two")]})
        (self.root / "_toc.yml").write_text(self.TOC, encoding="utf-8")
        (self.root / "index.md").write_text("# Book\n", encoding="utf-8")

    def test_audiences_come_from_the_entry_they_sit_under(self):
        self.write()
        self.assertEqual(prep_alps.toc_audiences(self.root), {"chap01_soln": "instructor"})
        # The column says whose the key is: one after a chapter's sections is
        # the chapter's, one written before its entry's file: is that entry's.
        (self.root / "_toc.yml").write_text(
            "format: jb-book\nroot: index\nparts:\n- caption: Chapters\n  audience: instructor\n"
            "  chapters:\n  - file: chap01\n    sections:\n    - file: chap01_soln\n"
            "    - audience: instructor\n      file: chap02_soln\n    audience: instructor\n"
            "  - file: chap02\n",
            encoding="utf-8",
        )
        self.assertEqual(
            prep_alps.toc_audiences(self.root),
            {"chap01": "instructor", "chap02_soln": "instructor"},
        )
        (self.root / "_toc.yml").write_text(self.TOC, encoding="utf-8")
        book = prep_alps.Book(self.root)
        self.assertTrue(book.for_teachers("chap01_soln.ipynb"))
        self.assertFalse(book.for_teachers("chap01.ipynb"))
        self.assertEqual(prep_alps.teachers_copy(book, "chap01.ipynb"), "chap01_soln.ipynb")
        self.assertIsNone(prep_alps.teachers_copy(book, "chap02.ipynb"))
        self.assertIsNone(prep_alps.teachers_copy(book, "chap01_soln.ipynb"))
        self.assertEqual(prep_alps.book_counts(book), (3, 3))

    def test_students_get_a_visible_work_cell_and_teachers_keep_the_solution(self):
        self.write()
        # A _soln page listed without audience: instructor is no teachers' copy:
        # its chapter keeps its solutions.
        cells = json.loads((self.root / "chap01.ipynb").read_text(encoding="utf-8"))["cells"]
        for name in ("chap02.ipynb", "chap02_soln.ipynb"):
            notebook = {"cells": cells, "metadata": {}, "nbformat": 4, "nbformat_minor": 4}
            prep_alps.write_notebook(name, notebook, self.root)
        (self.root / "_toc.yml").write_text(
            self.TOC + "    sections:\n    - file: chap02_soln\n", encoding="utf-8"
        )
        self.assertIsNone(prep_alps.teachers_copy(prep_alps.Book(self.root), "chap02.ipynb"))
        self.assertEqual(
            prep_alps.apply_text_rules(self.root),
            {
                "chap01.ipynb": {"scrub_student_solutions": 1, "show_output_of_hidden_code": 1},
                "chap01_soln.ipynb": {"show_output_of_hidden_code": 1},
                "chap02.ipynb": {"show_output_of_hidden_code": 1},
                "chap02_soln.ipynb": {"show_output_of_hidden_code": 1},
            },
        )
        student = json.loads((self.root / "chap01.ipynb").read_text(encoding="utf-8"))["cells"]
        teacher = json.loads((self.root / "chap01_soln.ipynb").read_text(encoding="utf-8"))["cells"]
        work = student[3]
        self.assertEqual(prep_alps.source_of(work), prep_alps.WORK_CELL)
        self.assertEqual((work["outputs"], work["execution_count"], work["metadata"]), ([], None, {}))
        self.assertEqual(student[1], teacher[1], "a cell that is not a solution keeps its tags and outputs")
        self.assertIn("x = 42", prep_alps.source_of(teacher[3]))
        self.assertEqual(prep_alps.apply_text_rules(self.root), {})

    def test_hidden_or_flagged_work_cells_are_made_visible_again(self):
        self.write()
        prep_alps.apply_text_rules(self.root)
        notebook = json.loads((self.root / "chap01.ipynb").read_text(encoding="utf-8"))
        notebook["cells"][3]["metadata"] = {"tags": ["hide-cell", "keep-me"], prep_alps.REFERENCE_ONLY: True}
        prep_alps.write_notebook("chap01.ipynb", notebook, self.root)
        self.assertEqual(
            prep_alps.apply_text_rules(self.root), {"chap01.ipynb": {"scrub_student_solutions": 1}}
        )
        self.assertEqual(cell_metadata(self.root, "chap01.ipynb", 3), {"tags": ["keep-me"]})

    def test_teachers_copies_do_not_answer_cross_references(self):
        write_book(
            self.root,
            {
                "chap01.ipynb": [md("# One\n\n## Target")],
                "chap01_soln.ipynb": [md("# One\n\n## Target")],
                "chap02.ipynb": [md("# Two"), md("See <<_Target>>.")],
            },
        )
        (self.root / "_toc.yml").write_text(self.TOC, encoding="utf-8")
        (self.root / "index.md").write_text("# Book\n", encoding="utf-8")
        self.assertEqual(
            [section.notebook for section in prep_alps.Book(self.root).sections["Target"]],
            ["chap01.ipynb"],
        )

    def test_upstream_students_copies_match_upstream_student_notebooks(self):
        copy_build_sources(self.root)
        prep_alps.apply_text_rules(self.root)
        book = prep_alps.Book(self.root)
        raw_strings = {("chap08.ipynb", 58), ("chap08.ipynb", 79), ("chap17.ipynb", 74)}
        chapters = [
            name for name in book.notebooks
            if prep_alps.teachers_copy(book, name) and name.startswith("chap")
        ]
        self.assertEqual(len(chapters), 20)
        scrubbed = 0
        for name in chapters:
            student = book.notebooks[name]["cells"]
            teacher = book.notebooks[prep_alps.teachers_copy(book, name)]["cells"]
            upstream = json.loads((JB.parent / "notebooks" / name).read_text(encoding="utf-8"))["cells"]
            self.assertEqual(len(student), len(upstream), name)
            for index, (ours, theirs, soln) in enumerate(zip(student, upstream, teacher)):
                if ours["cell_type"] != "code":
                    continue
                expected = soln if (name, index) in raw_strings else theirs
                self.assertEqual(
                    prep_alps.source_of(ours), prep_alps.source_of(expected), f"{name}:{index}"
                )
                if prep_alps.source_of(soln).startswith("# Solution"):
                    scrubbed += 1
                    self.assertNotEqual(prep_alps.source_of(soln), prep_alps.WORK_CELL)
        self.assertEqual(scrubbed, 359)

    def test_an_unlabelled_step_after_a_solution_is_scrubbed_too(self):
        output = {"output_type": "stream", "name": "stdout", "text": ["42\n"]}
        solution = code("# Solution\n\nx = 42", outputs=[output])
        step = code("# And the answer\n\nx + 1", outputs=[output])
        empty = code("")
        after_empty = code("x + 2", outputs=[output])
        teaching = code("y = 1", outputs=[output])
        cells = [md("# One"), md("**Exercise:** find x."), solution, step, empty, after_empty, md("Next."), teaching]
        write_book(self.root, {"chap01.ipynb": cells, "chap01_soln.ipynb": cells, "chap02.ipynb": [md("# Two")]})
        (self.root / "_toc.yml").write_text(self.TOC, encoding="utf-8")
        (self.root / "index.md").write_text("# Book\n", encoding="utf-8")
        report = prep_alps.apply_text_rules(self.root)
        self.assertEqual(report["chap01.ipynb"]["scrub_student_solutions"], 3)
        student = json.loads((self.root / "chap01.ipynb").read_text(encoding="utf-8"))["cells"]
        teacher = json.loads((self.root / "chap01_soln.ipynb").read_text(encoding="utf-8"))["cells"]
        for index in (2, 3, 5):
            self.assertEqual(prep_alps.source_of(student[index]), prep_alps.WORK_CELL, index)
            self.assertEqual(student[index]["outputs"], [], index)
        self.assertEqual(prep_alps.source_of(student[4]), "", "an empty cell holds no answer")
        self.assertEqual(student[7], teaching, "Markdown ends the run of solution cells")
        self.assertEqual(prep_alps.source_of(teacher[3]), "# And the answer\n\nx + 1")
        self.assertEqual(teacher[3]["outputs"], [output])
        self.assertEqual(prep_alps.apply_text_rules(self.root), {})

    ANSWER_MARKS = (
        "posterior1", "posterior2", "likelihood1", "likelihood2", "gt_dist", "linspace(0, 25",
        "0.5511810168369614", "14.794330239819137", "14.327038448986379",
        "4.201483589848796", "4.259983296308155",
    )
    SOLUTION_CELLS = range(16, 24)

    def assert_birth_rate_student_page_holds_no_answer(self, book: prep_alps.Book) -> None:
        notebook = book.notebooks["hospital_birth_rate.ipynb"]
        text = json.dumps(notebook, ensure_ascii=False)
        for mark in self.ANSWER_MARKS:
            self.assertNotIn(mark, text)
        for index in self.SOLUTION_CELLS:
            cell = notebook["cells"][index]
            self.assertEqual(prep_alps.source_of(cell), prep_alps.WORK_CELL, index)
            self.assertEqual((cell["outputs"], cell["execution_count"]), ([], None), index)
            self.assertFalse(set(cell["metadata"].get("tags", [])) & set(prep_alps.HIDING_TAGS), index)
            self.assertNotIn(prep_alps.REFERENCE_ONLY, cell["metadata"], index)

    def assert_birth_rate_teachers_copy_holds_the_answer(self, book: prep_alps.Book) -> None:
        cells = book.notebooks["hospital_birth_rate_soln.ipynb"]["cells"]
        upstream = json.loads(
            (JB.parent / "examples" / "hospital_birth_rate.ipynb").read_text(encoding="utf-8")
        )["cells"]
        self.assertEqual(len(cells), len(upstream))
        for index in self.SOLUTION_CELLS:
            self.assertEqual(prep_alps.source_of(cells[index]), prep_alps.source_of(upstream[index]), index)
            self.assertEqual(cells[index]["outputs"], upstream[index]["outputs"], index)
        for index in range(16, 23):
            self.assertIn("hide-cell", cells[index]["metadata"].get("tags", []), index)

    def test_the_example_with_solutions_is_split_like_a_chapter(self):
        copy_build_sources(self.root)
        prep_alps.apply_text_rules(self.root)
        book = prep_alps.Book(self.root)
        self.assertEqual(prep_alps.teachers_copy(book, "hospital_birth_rate.ipynb"), "hospital_birth_rate_soln.ipynb")
        self.assert_birth_rate_student_page_holds_no_answer(book)
        self.assert_birth_rate_teachers_copy_holds_the_answer(book)
        student = book.notebooks["hospital_birth_rate.ipynb"]["cells"]
        upstream = json.loads(
            (JB.parent / "examples" / "hospital_birth_rate.ipynb").read_text(encoding="utf-8")
        )["cells"]
        self.assertEqual(len(student), len(upstream))
        for index, (ours, theirs) in enumerate(zip(student, upstream)):
            if index not in self.SOLUTION_CELLS:
                self.assertEqual(prep_alps.source_of(ours), prep_alps.source_of(theirs), index)
                self.assertEqual(ours.get("outputs"), theirs.get("outputs"), index)

    def test_committed_example_is_split_like_a_chapter(self):
        book = prep_alps.Book(JB)
        self.assertTrue(book.for_teachers("hospital_birth_rate_soln.ipynb"))
        self.assert_birth_rate_student_page_holds_no_answer(book)
        self.assert_birth_rate_teachers_copy_holds_the_answer(book)

    def test_only_listed_pages_with_solutions_are_the_ones_split(self):
        book = prep_alps.Book(JB)
        for name, notebook in book.notebooks.items():
            has_solution = any(
                cell["cell_type"] == "code"
                and prep_alps.source_of(cell).startswith("# Solution")
                and prep_alps.source_of(cell) != prep_alps.WORK_CELL
                for cell in notebook["cells"]
            )
            if has_solution:
                self.assertTrue(book.for_teachers(name), f"{name} shows students a solution")

    def test_committed_book(self):
        book = prep_alps.Book(JB)
        toc = (JB / "_toc.yml").read_text(encoding="utf-8")
        chapters = [name for name in book.notebooks if prep_alps.teachers_copy(book, name)]
        self.assertEqual(len(chapters), 21)
        for name in chapters:
            stem = name.removesuffix(".ipynb")
            # The teachers' copy is the chapter's only, and so last, entry.
            self.assertRegex(
                toc,
                rf"- file: {stem}\n(?:\s+title: .*\n)?\s+sections:\n\s+- file: {stem}_soln\n"
                rf'\s+title: "Solution: [^"\n]+"\n\s+audience: instructor\n(?!\s{{4,}}- file:)',
            )
            work = [
                cell for cell in book.notebooks[name]["cells"]
                if cell["cell_type"] == "code" and prep_alps.source_of(cell) == prep_alps.WORK_CELL
            ]
            self.assertTrue(work, name)
            solutions = [
                cell for cell in book.notebooks[name]["cells"]
                if cell["cell_type"] == "code"
                and prep_alps.source_of(cell).startswith("# Solution")
            ]
            self.assertEqual(solutions, work, f"{name} still holds a solution")
            for cell in work:
                self.assertEqual(cell["outputs"], [], name)
                self.assertIsNone(cell["execution_count"], name)
                self.assertFalse(
                    set(cell["metadata"].get("tags", [])) & set(prep_alps.HIDING_TAGS), name
                )
                self.assertNotIn(prep_alps.REFERENCE_ONLY, cell["metadata"], name)
        pages, _ = prep_alps.book_counts(book)
        self.assertEqual(pages, len([p for p in book.pages if p != "about-this-book"]) - 21)


def cell_metadata(root: Path, name: str, index: int) -> dict:
    return json.loads((root / name).read_text(encoding="utf-8"))["cells"][index]["metadata"]


if __name__ == "__main__":
    unittest.main()
