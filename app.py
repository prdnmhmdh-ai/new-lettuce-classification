import os
import base64
from flask import Flask, send_from_directory, render_template, request, jsonify
from inference_sdk import InferenceHTTPClient
from PIL import Image
from io import BytesIO
from roboflow import Roboflow
import supervision as sv
import cv2
import traceback
import numpy as np
import requests
from ultralytics import YOLO

app = Flask(__name__, static_folder='assets')
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['RESULT_FOLDER'] = 'static/results'
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024

try:
    model = YOLO("lettucemodel.pt")
    print("✅ Model 'lettucemodel.pt' berhasil dimuat.")
except Exception as e:
    print(f"❌ Gagal memuat model: {e}")
    model = None

for folder in [app.config['UPLOAD_FOLDER'], app.config['RESULT_FOLDER']]:
    if not os.path.exists(folder):
        os.makedirs(folder)

CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="6OD1amgfXBOwuFLTWdVH"
)

for folder in [app.config['UPLOAD_FOLDER'], app.config['RESULT_FOLDER']]:
    if not os.path.exists(folder):
        os.makedirs(folder)

@app.route("/")
def index():
    return send_from_directory(os.getcwd(), "index.html")

@app.route('/static/results/<path:filename>')
def serve_result_image(filename):
    return send_from_directory('static/results', filename)

@app.route('/static/uploads/<path:filename>')
def serve_upload_image(filename):
    return send_from_directory('static/uploads', filename)

@app.route("/disease", methods=['GET'])
def diseasePage():
    return render_template('disease.html')

@app.route('/disease', methods=['POST'])
def detectDisease():
    try:
        data = request.get_json()
        if 'camera_image' in data:
            data_url = data['camera_image']
            header, encoded = data_url.split(",", 1)
            binary_data = base64.b64decode(encoded)
            image = Image.open(BytesIO(binary_data))

            image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'capture_disease.jpg')
            image.save(image_path, optimize=True, quality=75)

        elif 'image' in request.files:
            file = request.files['image']
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(image_path)

        else:
            return jsonify({"reply": "❌ No image provided"}), 400

        print("📦 Loading Roboflow...")
        rf = Roboflow(api_key="6OD1amgfXBOwuFLTWdVH")
        project = rf.workspace().project("aquaponic_polygan_disease_test")
        model = project.version(5).model

        print("🔍 Predicting...")
        result_json = model.predict(image_path, confidence=40).json()
        preds = result_json["predictions"]

        if not preds:
            return jsonify({
                "predictions": [],
                "annotated_image": None,
                "reply": "✅ Tidak ada objek yang terdeteksi."
            })

        detections = sv.Detections.from_inference(result_json)

        labels = [item["class"] for item in preds]

        image_cv2 = cv2.imread(image_path)

        box_annotator = sv.BoxAnnotator(thickness=4)
        label_annotator = sv.LabelAnnotator()

        annotated_image = box_annotator.annotate(
            scene=image_cv2, detections=detections)
        annotated_image = label_annotator.annotate(
            scene=annotated_image, detections=detections, labels=labels)

        output_filename = "result_" + os.path.basename(image_path)
        output_path = os.path.join(app.config['RESULT_FOLDER'], output_filename)
        success = cv2.imwrite(output_path, annotated_image)

        if not success:
            return jsonify({"reply": "❌ Gagal menyimpan hasil gambar."}), 500

        return jsonify({
            "predictions": preds,
            "annotated_image": f"/static/results/{output_filename}"
        })
    
    except requests.exceptions.HTTPError as err:
        print(f"Error: {err}")
        if err.response:
            print(f"Response: {err.response.text}")

    except Exception as e:
        traceback.print_exc()
        return jsonify({"reply": f"❌ Error: {str(e)}"}), 500

@app.route("/harvest", methods=['GET'])
def harvestPage():
    return render_template('harvest.html')

@app.route('/harvest', methods=['POST'])
def detectHarvest():
    try:
        data = request.get_json()
        if 'camera_image' in data:
            data_url = data['camera_image']
            header, encoded = data_url.split(",", 1)
            binary_data = base64.b64decode(encoded)
            image = Image.open(BytesIO(binary_data))

            image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'capture_harvest.jpg')
            image.save(image_path, optimize=True, quality=75)

        elif 'image' in request.files:
            file = request.files['image']
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(image_path)

        else:
            return jsonify({"reply": "❌ No image provided"}), 400

        print("📦 Loading Roboflow...")
        rf = Roboflow(api_key="6OD1amgfXBOwuFLTWdVH")
        project = rf.workspace().project("aquaponic_polygan_test")
        model = project.version(2).model

        print("🔍 Predicting...")
        result_json = model.predict(image_path, confidence=40).json()
        preds = result_json["predictions"]

        if not preds:
            return jsonify({
                "predictions": [],
                "annotated_image": None,
                "reply": "✅ Tidak ada objek yang terdeteksi."
            })

        detections = sv.Detections.from_inference(result_json)

        labels = [item["class"] for item in preds]

        image_cv2 = cv2.imread(image_path)

        box_annotator = sv.BoxAnnotator(thickness=4)
        label_annotator = sv.LabelAnnotator()

        annotated_image = box_annotator.annotate(
            scene=image_cv2, detections=detections)
        annotated_image = label_annotator.annotate(
            scene=annotated_image, detections=detections, labels=labels)

        output_filename = "result_" + os.path.basename(image_path)
        output_path = os.path.join(app.config['RESULT_FOLDER'], output_filename)
        success = cv2.imwrite(output_path, annotated_image)

        if not success:
            return jsonify({"reply": "❌ Gagal menyimpan hasil gambar."}), 500

        return jsonify({
            "predictions": preds,
            "annotated_image": f"/static/results/{output_filename}"
        })
    
    except requests.exceptions.HTTPError as err:
        print(f"Error: {err}")
        if err.response:
            print(f"Response: {err.response.text}")

    except Exception as e:
        traceback.print_exc()
        return jsonify({"reply": f"❌ Error: {str(e)}"}), 500

@app.route("/ourmodel", methods=['GET'])
def ourmodelPage():
    return render_template('ourmodel.html')

@app.route('/ourmodel', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model tidak berhasil dimuat'}), 500

    try:
        # Simpan gambar dari kamera atau unggahan
        if request.is_json and 'camera_image' in request.get_json():
            data_url = request.get_json()['camera_image']
            header, encoded = data_url.split(",", 1)
            binary_data = base64.b64decode(encoded)
            image = Image.open(BytesIO(binary_data)).convert("RGB")
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'camera_input.jpg')
            image.save(image_path)
        elif 'image' in request.files:
            file = request.files['image']
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(image_path)
            image = Image.open(image_path).convert("RGB")
        else:
            return jsonify({'error': 'Tidak ada gambar yang dikirim'}), 400

        # Konversi ke NumPy array
        image_np = np.array(image)
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        # Prediksi menggunakan YOLO
        results = model(image_bgr)[0]  # YOLOv8 model inference

        if results.boxes is None or len(results.boxes) == 0:
            return jsonify({
                'predictions': [],
                'annotated_image': None,
                'reply': '✅ Tidak ada objek yang terdeteksi.'
            })

        # Ambil hasil prediksi
        boxes = results.boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
        scores = results.boxes.conf.cpu().numpy()
        class_ids = results.boxes.cls.cpu().numpy().astype(int)
        class_names = model.names

        predictions = []

        # Gambar bounding box
        for box, score, class_id in zip(boxes, scores, class_ids):
            x1, y1, x2, y2 = map(int, box)
            label = f"{class_names[class_id]} {score:.2f}"

            predictions.append({
                'class': class_names[class_id],
                'confidence': float(score),
                'box': [x1, y1, x2, y2]
            })

            # Draw box
            cv2.rectangle(image_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image_bgr, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 2)

        # Simpan hasil anotasi
        output_filename = "result_" + os.path.basename(image_path)
        output_path = os.path.join(app.config['RESULT_FOLDER'], output_filename)
        cv2.imwrite(output_path, image_bgr)

        return jsonify({
            'predictions': predictions,
            'annotated_image': f"/static/results/{output_filename}"
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500
    
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5001)
