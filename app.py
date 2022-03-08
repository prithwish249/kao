from curses.panel import bottom_panel
import json
import face_recognition as fr
from flask import Flask, render_template, request, redirect, jsonify
import base64
from PIL import Image, ImageDraw
import os, io, sys
import numpy as np
import cv2


ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif"}

app = Flask(__name__)


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def detect_face_from_image(train, test, label):
    res = {
        "face_location": [],
        "unknown": False,
    }
    train_img = fr.load_image_file(train)
    train_enc = fr.face_encodings(train_img)[0]

    test_img = fr.load_image_file(test)
    test_enc = fr.face_encodings(test_img)[0]

    results = fr.compare_faces([train_enc], test_enc)

    pil_image = Image.fromarray(test_img)

    draw = ImageDraw.Draw(pil_image)
    if results[0]:
        (top, right, bottom, left) = fr.face_locations(test_img)[0]
        draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 0))
        text_width, text_height = draw.textsize(label)
        draw.rectangle(
            ((left, bottom - text_height - 10), (right, bottom)),
            fill=(0, 0, 0),
            outline=(0, 0, 0),
        )
        draw.text(
            (left + 6, bottom - text_height - 5), label, fill=(255, 255, 255, 255)
        )
        del draw
        pil_image.show()
        rawBytes = io.BytesIO()
        pil_image.save(rawBytes, "JPEG")
        rawBytes.seek(0)
        img_base64 = base64.b64encode(rawBytes.read())
        return img_base64
    else:
        res["unknown"] = True

    return res


@app.route("/", methods=["GET", "POST"])
def home():

    if request.method == "POST":
        if "face_train" not in request.files or "face_test" not in request.files:
            return redirect(request.url)

        train = request.files["face_train"]
        test = request.files["face_test"]

        if train.filename == "" or test.filename == "":
            return redirect(request.url)

        print(request.form)
        if not request.form.get("label"):
            return redirect(request.url)

        label = request.form.get("label")

        if (
            train
            and test
            and allowed_file(train.filename)
            and allowed_file(test.filename)
        ):
            res = detect_face_from_image(train, test, label)

            return jsonify({"res": str(res)})

    else:
        return render_template("index.html")
