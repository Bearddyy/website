from flask import Flask
from flask import send_file
from flask import render_template
from flask.typing import ResponseReturnValue
import requests
from PIL import Image
import os
import io

app = Flask(__name__)

FONT_PATH = "https://fonts.googleapis.com/css?family=Inter|Libre+Barcode+39+Text|Overpass+Mono:wght@700&display=swap"

@app.route("/")
def root() -> ResponseReturnValue:
    return render_template("index.html", font_path=FONT_PATH)

@app.route("/projects/<string:project>")
def projects(project: str) -> ResponseReturnValue:
    #strip the project name of any slashes and other characters
    project = project.replace("/", "")
    project = project.replace("\\", "")
    project = project + ".html"
    # if project is not in the list of projects, return the error page
    if project not in os.listdir("templates/projects"):
        print(f"project {project} not found in {os.listdir('templates/projects')}")
        return page_not_found(project, " not found")
    else:
        return render_template("projects/" + project, font_path=FONT_PATH)

@app.route("/ascention")
def ascention() -> ResponseReturnValue:
    return render_template("ascention.html", font_path=FONT_PATH)

# route for get requests for an image with width and height
@app.route("/image/<int:width>/<int:height>")
def image(width: int, height: int) -> ResponseReturnValue:
    print(f"width: {width}, height: {height}")
    # use https://picsum.photos/ to get an image
    response = requests.get(f"https://picsum.photos/{width}/{height}?grayscale")

    # convert the image to grayscale png
    with Image.open(io.BytesIO(response.content)) as img:
        img.convert("L").save("image.png")

    # return the image as png
    return send_file("image.png", mimetype="image/png")

#fallback route returns not found page
@app.errorhandler(404)
def page_not_found(e: object, extra_text: str = "") -> tuple[str, int]:
    return render_template("error.html", error=e, extra_text=extra_text, font_path=FONT_PATH), 200