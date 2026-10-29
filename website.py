"""Flask app for the personal website: pages, images, and AI benchmark charts."""

import csv
import hmac
import io
import os
import re

import plotly.graph_objects as go
import requests
from flask import (
    Flask,
    jsonify,
    render_template,
    request,
    send_file,
    send_from_directory,
)
from flask.typing import ResponseReturnValue
from PIL import Image, ImageOps
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

# Photography: uploads land in data/photos (full resolution originals) and
# data/photos/display (page-sized copies), named "<seq>-<slug>.jpg" so sorting
# by name is sorting by upload order. The upload key lives only in the server
# environment; the repo is public, so it must never appear in source.
PHOTO_DIR = os.path.join(app.root_path, "data", "photos")
PHOTO_DISPLAY_DIR = os.path.join(PHOTO_DIR, "display")
UPLOAD_KEY_ENV = "PHOTO_UPLOAD_KEY"
UPLOAD_HEADER = "X-Upload-Key"
SEEN_COOKIE = "seen_up_to"
DISPLAY_MAX_EDGE = 1600
UPLOAD_MAX_BYTES = 40 * 1024 * 1024
# MPO is the multi-frame JPEG variant Sony/Samsung cameras write; PIL reads
# its first frame, so it re-encodes to plain JPEG like any other input.
ALLOWED_UPLOAD_FORMATS: frozenset[str] = frozenset(
    {"JPEG", "PNG", "WEBP", "TIFF", "BMP", "MPO"}
)
PHOTO_ID_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")

app.config["MAX_CONTENT_LENGTH"] = UPLOAD_MAX_BYTES

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


def slugify(name: str) -> str:
    """Reduce an uploaded filename to a safe lowercase slug for a photo id."""
    stem = os.path.splitext(os.path.basename(name))[0].lower()
    slug = re.sub(r"[^a-z0-9]+", "-", stem).strip("-")[:40].rstrip("-")
    return slug or "photo"


def list_photo_ids() -> list[str]:
    """Stored photo ids, newest (highest sequence) first."""
    if not os.path.isdir(PHOTO_DIR):
        return []
    return sorted(
        (
            name[: -len(".jpg")]
            for name in os.listdir(PHOTO_DIR)
            if name.endswith(".jpg") and not name.endswith("_display.jpg")
        ),
        reverse=True,
    )


def photo_seq(photo_id: str) -> int:
    """Sequence number encoded in the leading digits of a photo id."""
    try:
        return int(photo_id.split("-", 1)[0])
    except ValueError:
        return 0


def next_photo_id(filename: str) -> str:
    """Build the next sequential photo id from the highest id on disk."""
    seq = max((photo_seq(stored) for stored in list_photo_ids()), default=0) + 1
    return f"{seq:06d}-{slugify(filename)}"


def seen_watermark(raw: str | None) -> int:
    """Parse the seen cookie; anything unparseable means nothing has been seen."""
    try:
        return int(raw or "0")
    except ValueError:
        return 0


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


@app.route("/projects/tithe-map/app")
def tithe_map_app() -> ResponseReturnValue:
    """Serve the raw tithe map viewer for the page's iframe.

    The viewer is a self-contained HTML file (the scanned map is embedded
    as base64), so it is served straight from disk instead of through
    Jinja. The wrapper page at /projects/tithe-map shows it alongside the
    site navigation.
    """
    return send_from_directory(
        os.path.join(app.root_path, "pages"),
        "tithe-map-app.html",
        max_age=86400,
    )


@app.route("/projects/tithe-map/interactive_map_full.jpg")
def tithe_map_full() -> ResponseReturnValue:
    """Serve the full-resolution map scan the viewer lazy-loads.

    The viewer embeds a half-scale scan for the zoomed-out overview and
    swaps to this full-size file once the user zooms in past 0.5x, so
    labels stay sharp at max zoom. Long cache: the file only changes when
    the viewer is regenerated.
    """
    return send_from_directory(
        os.path.join(app.root_path, "pages"),
        "interactive_map_full.jpg",
        max_age=86400,
    )


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


@app.route("/photography")
def photography() -> ResponseReturnValue:
    """Serve the photography page: newest first, unseen photos above a divider.

    The seen_up_to cookie holds the highest photo sequence the visitor has
    acknowledged with the SEEN button, so every newer photo renders in the
    new section at the top until it is acknowledged.
    """
    watermark = seen_watermark(request.cookies.get(SEEN_COOKIE))
    photo_ids = list_photo_ids()
    photos = [
        {"id": photo_id, "is_new": photo_seq(photo_id) > watermark}
        for photo_id in photo_ids
    ]
    new_count = sum(1 for photo in photos if photo["is_new"])
    max_seq = photo_seq(photo_ids[0]) if photo_ids else 0
    return render_template(
        "photography.html",
        photos=photos,
        new_count=new_count,
        max_seq=max_seq,
        seen_cookie=SEEN_COOKIE,
        font_path=FONT_PATH,
    )


def photo_response(photo_id: str, display: bool) -> ResponseReturnValue:
    """Send one stored photo file, or the error page when it is not on disk."""
    if PHOTO_ID_RE.fullmatch(photo_id) is None:
        return page_not_found(photo_id, " is not a valid photo id")
    directory = PHOTO_DISPLAY_DIR if display else PHOTO_DIR
    name = photo_id + ("_display.jpg" if display else ".jpg")
    if not os.path.isfile(os.path.join(directory, name)):
        return page_not_found(photo_id, " not found")
    # Photo ids are never reused, so the copies can be cached for a year.
    return send_from_directory(directory, name, max_age=31536000)


@app.route("/photos/<string:photo_id>/original")
def photo_original(photo_id: str) -> ResponseReturnValue:
    """Serve the full resolution copy of a stored photo."""
    return photo_response(photo_id, display=False)


@app.route("/photos/<string:photo_id>/display")
def photo_display(photo_id: str) -> ResponseReturnValue:
    """Serve the page-sized copy of a stored photo."""
    return photo_response(photo_id, display=True)


@app.route("/photos/upload", methods=["POST"])
def upload_photo() -> ResponseReturnValue:
    """Store an uploaded photo, authenticated by the PHOTO_UPLOAD_KEY env var.

    curl -F photo=@shot.jpg -H "X-Upload-Key: $PHOTO_UPLOAD_KEY" \\
        https://example.com/photos/upload

    The image is re-encoded to a full resolution JPEG (EXIF orientation
    applied) plus a display-sized copy for the grid, so the originals stay
    web-safe whatever format they were shot in.
    """
    key = os.environ.get(UPLOAD_KEY_ENV, "")
    if not key:
        return jsonify(error=f"uploads disabled: {UPLOAD_KEY_ENV} is not set"), 503
    if not hmac.compare_digest(request.headers.get(UPLOAD_HEADER, ""), key):
        return jsonify(error="invalid upload key"), 401
    upload = request.files.get("photo")
    if upload is None or not upload.filename:
        return jsonify(error="send the image as multipart field 'photo'"), 400
    try:
        with Image.open(upload.stream) as probe:
            image_format = probe.format
            probe.verify()
    except OSError:
        return jsonify(error="not a readable image file"), 400
    if image_format not in ALLOWED_UPLOAD_FORMATS:
        return jsonify(error=f"unsupported image format: {image_format}"), 400
    upload.stream.seek(0)
    with Image.open(upload.stream) as raw:
        original = ImageOps.exif_transpose(raw).convert("RGB")
    photo_id = next_photo_id(upload.filename)
    os.makedirs(PHOTO_DISPLAY_DIR, exist_ok=True)
    original.save(
        os.path.join(PHOTO_DIR, photo_id + ".jpg"),
        format="JPEG",
        quality=92,
        optimize=True,
    )
    display = original.copy()
    display.thumbnail((DISPLAY_MAX_EDGE, DISPLAY_MAX_EDGE))
    display.save(
        os.path.join(PHOTO_DISPLAY_DIR, photo_id + "_display.jpg"),
        format="JPEG",
        quality=85,
        optimize=True,
    )
    return jsonify(id=photo_id), 201


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
