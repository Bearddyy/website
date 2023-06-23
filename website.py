from flask import Flask
import requests

app = Flask(__name__)

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"



# route for get requests for an image with width and height
@app.route("/image/<int:width>/<int:height>")
def image(width, height):
    # use https://picsum.photos/ to get an image
    response = requests.get(f"https://picsum.photos/{width}/{height}?grayscale")
    # provide the image in the response
    return response.content
