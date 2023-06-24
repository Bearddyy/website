from flask import Flask
from flask import send_file
from flask import render_template
import requests
from PIL import Image

app = Flask(__name__)

@app.route("/")
def root():
    return render_template("index.html")



# route for get requests for an image with width and height
@app.route("/image/<int:width>/<int:height>")
def image(width, height):
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
