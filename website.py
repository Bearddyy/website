from flask import Flask
from flask import send_file
from flask import render_template
import requests
from PIL import Image
import os
import random
import numpy as np

app = Flask(__name__)

FONT_PATH = "https://fonts.googleapis.com/css?family=Inter|Libre+Barcode+39+Text|Overpass+Mono:wght@700&display=swap"

@app.route("/")
def root():
    return render_template("index.html", font_path=FONT_PATH)

@app.route("/projects/<string:project>")
def projects(project):
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
def ascention():
    return render_template("ascention.html", font_path=FONT_PATH)

# route for get requests for an image with width and height
@app.route("/image/<int:width>/<int:height>")
def image(width, height):
    print(f"width: {width}, height: {height}")
    # use https://picsum.photos/ to get an image
    response = requests.get(f"https://picsum.photos/{width}/{height}?grayscale", stream=True)

    #if the image is jpeg, convert it to png
    if response.headers['Content-Type'] == 'image/jpeg':
        img = Image.open(response.raw)
        img.save("image.png")
        response.raw = open("image.png", "rb")
    else:
        img.save("image.png")

    #convert the image to 16 bit grayscale
    img = Image.open(response.raw)
    img = img.convert("L")
    img.save("image.png")
    response.raw = open("image.png", "rb")
    
    # return the image in binary in the .content field
    return send_file(response.raw, mimetype="image/png")

@app.route("/log/<string:id>", methods=["POST", "GET"])
def log(id):
    if request.method == "POST":
        # create a new log with the id and the data
        return "POST", 200
    elif request.method == "GET":
        # return the log with the id
        return "GET", 200

@app.route("/log/list/<string:id>", methods=["GET"])
def log_list(id):
    # return a list of all the logs currently stored
    return str(os.listdir()), 200

@app.route("/julia/<int:width>/<int:height>")
def julia(width, height):
    print(f"width: {width}, height: {height}")
    width = int(width)
    height = int(height)
    ratio = width / height
    c = complex(random.uniform(-0.1, 0.1), random.uniform(0.65, 0.75))
    n = 50
    x_min = -0.5 * ratio
    x_max = 0.5 * ratio
    y_min = -0.5 * (1/ratio)
    y_max = 0.5 * (1/ratio)


    x = np.linspace(x_min, x_max, width)
    y = np.linspace(y_min, y_max, height)
    z = x + y[:, None] * 1j
    image = np.zeros((height, width))
    for k in range(n):
        z = z**2 + c
        mask = (np.abs(z) > 2) & (image == 0)
        image[mask] = k
        z[mask] = np.nan
    image = np.log(image + 1)
    image[np.isnan(image)] = 0
    image = image / np.nanmax(image)
    image = np.uint8(image * 255)
    img = Image.fromarray(image)
    img.save("julia.png")
    return send_file("julia.png", mimetype="image/png")

#fallback route returns not found page
@app.errorhandler(404)
def page_not_found(e, extra_text=""):
    return render_template("error.html", error=e, extra_text=extra_text, font_path=FONT_PATH), 200
