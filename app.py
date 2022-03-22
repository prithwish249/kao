import face_recognition as fr
from flask import Flask, render_template, request, redirect, jsonify
import base64
from PIL import Image, ImageDraw
import io
import numpy as np


ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}

app = Flask(__name__)


def resize_image(img, basewidth=300):
    wpercent = basewidth / float(img.size[0])
    hsize = int((float(img.size[1]) * float(wpercent)))
    img = img.resize((basewidth, hsize), Image.ANTIALIAS)
    return img


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def detect_face_from_image(train, test, label):
    train = Image.open(train)
    t_io = io.BytesIO()

    test = Image.open(test)
    tst_io = io.BytesIO()

    train = resize_image(train)
    test = resize_image(test)

    train_img = np.array(train)
    train_enc = fr.face_encodings(train_img)
    if len(train_enc) == 0:
        return {"res": "", "error": True}
    else:
        train_enc = train_enc[0]
    test_img = np.array(test)
    test_enc = fr.face_encodings(test_img)
    if len(test_enc) == 0:
        return {"res": "", "error": True}
    else:
        test_enc = test_enc[0]

    results = fr.compare_faces([train_enc], test_enc)

    pil_image = Image.fromarray(test_img)

    draw = ImageDraw.Draw(pil_image)
    print(results)
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
        # pil_image.show()
        rawBytes = io.BytesIO()
        pil_image.save(rawBytes, "JPEG")
        rawBytes.seek(0)
        img_base64 = base64.b64encode(rawBytes.read())
        return {"res": str(img_base64), "error": False}
    else:
        return {"res": "", "error": True}


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

            return jsonify(res)
        else:
            print("Here")
            return {"res": "", "error": True}

    else:
        return render_template("index.html")
