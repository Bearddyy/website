"""Flask app for the personal website: pages, images, and AI benchmark charts."""

import csv
import io
import os

import plotly.graph_objects as go
import requests
from flask import Flask, render_template, send_file
from flask.typing import ResponseReturnValue
from PIL import Image
from plotly.basedatatypes import BaseFigure
from plotly.io import to_html

app = Flask(__name__)

FONT_PATH = "https://fonts.googleapis.com/css?family=Electrolize|Libre+Barcode+39+Text&display=swap"

# Chart theme, kept in sync with the --bg/--fg/--accent/--muted tokens in
# static/styles/main.css (aged beige). Charts sit on the page background rather
# than a separate panel, so no extra surface colour is needed.
BG_COLOR = "#d9d1ba"  # aged beige, matches the site-wide * background
INK_COLOR = "#3a3733"  # warm charcoal: text, ticks, bar outlines
ACCENT_COLOR = "#a33f2d"  # palette red: bar fill
GRID_COLOR = "rgba(107, 103, 93, 0.45)"  # bright black, for the x grid

# Benchmark CSV column -> (chart title, axis label) for each metric chart.
BENCH_METRICS: dict[str, tuple[str, str]] = {
    "pp_tok_s": ("Prompt Processing Speed", "tokens / second"),
    "tg_tok_s": ("Generation Speed", "tokens / second"),
    "ttft_s": ("Time to First Token", "seconds (lower is better)"),
    "load_s": ("Model Load Time", "seconds (lower is better)"),
}

# Benchmark CSV column -> the paragraph printed under that chart on the
# benchmarks page. Keep the tone informal and free of AI speech patterns.
BENCH_EXPLANATIONS: dict[str, str] = {
    "pp_tok_s": (
        "Prompt processing is how fast the model reads what you send it, and "
        "for agentic work this is the number that actually runs your wall "
        "clock. Every tool call resubmits the whole conversation, so an agent "
        "spends most of its time rereading a prompt tens of thousands of "
        "tokens long. The MoE model chews through that at 1401 tokens per "
        "second while the dense 27B crawls at around 240. The little 4B chat "
        "model tops the chart, but you are not getting 125B quality out of it."
    ),
    "tg_tok_s": (
        "Generation speed is the tokens per second you watch stream out while "
        "the model writes. Same story as above: the MoE model keeps only 6B "
        "parameters active per token, so it reads less from memory and writes "
        "almost twice as fast as the dense 27B. The 4B model is still the "
        "quickest thing here, it is just a different class of assistant."
    ),
    "ttft_s": (
        "Time to first token is how long you stare at the cursor before "
        "anything appears. The dense 27B runs sit near 14 seconds because "
        "thinking is on and the prompt is long. The flash next models start "
        "talking in 2 to 3 seconds. If you are chatting interactively this is "
        "the number that makes a model feel snappy or sluggish, mid agent "
        "loop you will barely notice it."
    ),
    "load_s": (
        "Load time is how long llama-swap needs to pull a model into memory "
        "when nothing is loaded yet. This is where halogen shows off: 0.7 "
        "seconds for a 125B model against 35.8 for the same weights in the "
        "standard container. When you swap models all day, that difference "
        "decides whether swapping is free or annoying."
    ),
}

# Rows whose model name contains any of these substrings are excluded from
# the public benchmarks (uncensored variants stay off the website).
EXCLUDED_MODEL_SUBSTRINGS: tuple[str, ...] = ("uncensored", "nothink")

# Bench model id -> short display name used on the charts. Any id missing
# here falls back to the raw bench name.
MODEL_DISPLAY_NAMES: dict[str, str] = {
    "chat-qwen3-4b": "qwen3 4b chat",
    "code-fim-qwen25-coder-14b": "coder 14b fim",
    "code-fim-qwen25-coder-7b": "coder 7b fim",
    "general-low-qwen38-27b": "dense 27b low",
    "general-medium-qwen38-27b": "dense 27b med",
    "general-vision-qwen38-27b": "dense 27b vision",
    "general-xhigh-qwen38-27b": "dense 27b xhigh",
    "general-qwen38-flash-next": "moe 125b flash next",
    "general-qwen38-flash-next-halogen": "moe 125b halogen",
}


def display_name(model_id: str) -> str:
    """Map a bench model id to its short chart label, or the raw id."""
    return MODEL_DISPLAY_NAMES.get(model_id, model_id)


def parse_float(value: str) -> float | None:
    """Parse a CSV field into a float, returning None when it is not numeric."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_bench_rows(csv_path: str) -> list[dict[str, str]]:
    """Read benchmark rows from a CSV file, dropping uncensored model variants.

    Rows without a model name or with a non-empty error field are also
    dropped, so only successful runs of censored models are charted.
    """
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [
        row
        for row in rows
        if row.get("model")
        and not row.get("error")
        and not any(sub in row["model"] for sub in EXCLUDED_MODEL_SUBSTRINGS)
    ]


def make_bench_fig(
    rows: list[dict[str, str]], col: str, title: str, axis_label: str
) -> BaseFigure:
    """Build a horizontal bar chart for one benchmark metric column.

    Models are sorted ascending so the best result sits at the top of the
    chart; the height scales with the model count so the layout stays
    correct as benchmark data grows.
    """
    data = [
        (display_name(row["model"]), value)
        for row in rows
        if (value := parse_float(row.get(col, ""))) is not None
    ]
    data.sort(key=lambda item: item[1])
    models = [model for model, _ in data]
    values = [value for _, value in data]

    fig = go.Figure(
        go.Bar(
            x=values,
            y=models,
            orientation="h",
            marker={
                "color": ACCENT_COLOR,
                "line": {"color": INK_COLOR, "width": 0.5},
            },
            text=[f"{value:g}" for value in values],
            textposition="outside",
            textfont={"color": INK_COLOR, "size": 12},
            hovertemplate="%{y}<br>%{x:g}<extra></extra>",
        )
    )
    fig.update_layout(
        title={"text": title, "font": {"color": INK_COLOR, "size": 20}},
        template=None,
        paper_bgcolor=BG_COLOR,
        plot_bgcolor=BG_COLOR,
        font={"color": INK_COLOR, "family": "Electrolize"},
        margin={"l": 170, "r": 80, "t": 60, "b": 70},
        height=max(300, 120 + 42 * len(models)),
        xaxis={
            "title": {"text": axis_label, "font": {"color": INK_COLOR, "size": 15}},
            "gridcolor": GRID_COLOR,
            "zerolinecolor": INK_COLOR,
            "tickfont": {"color": INK_COLOR},
            "range": [0, max(values) * 1.18] if values else None,
        },
        yaxis={"tickfont": {"color": INK_COLOR, "size": 13}},
    )
    return fig


@app.route("/")
def root() -> ResponseReturnValue:
    """Serve the site index page."""
    return render_template("index.html", font_path=FONT_PATH)


@app.route("/projects/<string:project>")
def projects(project: str) -> ResponseReturnValue:
    """Serve a project page, or the error page when it does not exist."""
    # strip the project name of any slashes and other characters
    project = project.replace("/", "").replace("\\", "") + ".html"
    # if project is not in the list of projects, return the error page
    if project not in os.listdir("templates/projects"):
        print(f"project {project} not found in {os.listdir('templates/projects')}")
        return page_not_found(project, " not found")
    return render_template("projects/" + project, font_path=FONT_PATH)


@app.route("/ascention")
def ascention() -> ResponseReturnValue:
    """Serve the ascention page."""
    return render_template("ascention.html", font_path=FONT_PATH)


@app.route("/ai-inference")
def ai_inference() -> ResponseReturnValue:
    """Serve the AI inference landing page."""
    return render_template("ai/index.html", font_path=FONT_PATH)


@app.route("/ai-inference/benchmarks")
def benchmarks() -> ResponseReturnValue:
    """Render the AI inference benchmarks page.

    Reads the checked-in benchmark CSV, excludes uncensored model
    variants, and renders one website-themed chart per metric column.
    """
    csv_path = os.path.join(app.root_path, "data", "benchmarks.csv")
    rows = load_bench_rows(csv_path)
    charts = []
    for col, (title, axis_label) in BENCH_METRICS.items():
        if not any(parse_float(row.get(col, "")) is not None for row in rows):
            continue
        fig = make_bench_fig(rows, col, title, axis_label)
        # The .chart CSS container has no fixed height, so the embed div
        # must carry the figure's pixel height or it collapses to zero.
        charts.append(
            (
                to_html(
                    fig,
                    include_plotlyjs=False,
                    full_html=False,
                    default_width="100%",
                    default_height=f"{fig.layout.height}px",
                    config={
                        "displaylogo": False,
                        "responsive": True,
                        "modeBarButtonsToRemove": ["lasso2d", "select2d"],
                    },
                ),
                BENCH_EXPLANATIONS[col],
            )
        )
    run_date = rows[0]["date"] if rows else "unknown"
    return render_template(
        "ai/benchmarks.html",
        charts=charts,
        run_date=run_date,
        model_count=len(rows),
        font_path=FONT_PATH,
    )


@app.route("/ai-inference/<string:page>")
def ai_page(page: str) -> ResponseReturnValue:
    """Serve an AI inference subpage, or the error page when it does not exist."""
    # strip the page name of any slashes and other characters
    page = page.replace("/", "").replace("\\", "") + ".html"
    # if page is not in the list of ai pages, return the error page
    if page not in os.listdir("templates/ai"):
        print(f"page {page} not found in {os.listdir('templates/ai')}")
        return page_not_found(page, " not found")
    return render_template("ai/" + page, font_path=FONT_PATH)


# route for get requests for an image with width and height
@app.route("/image/<int:width>/<int:height>")
def image(width: int, height: int) -> ResponseReturnValue:
    """Fetch a grayscale photo from picsum and return it as a PNG."""
    print(f"width: {width}, height: {height}")
    # use https://picsum.photos/ to get an image
    response = requests.get(
        f"https://picsum.photos/{width}/{height}?grayscale", timeout=30
    )

    # convert the image to grayscale png
    with Image.open(io.BytesIO(response.content)) as img:
        img.convert("L").save("image.png")

    # return the image as png
    return send_file("image.png", mimetype="image/png")


# fallback route returns not found page
@app.errorhandler(404)
def page_not_found(e: object, extra_text: str = "") -> tuple[str, int]:
    """Render the error page for missing routes."""
    return (
        render_template(
            "error.html", error=e, extra_text=extra_text, font_path=FONT_PATH
        ),
        200,
    )
