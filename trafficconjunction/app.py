from flask import Flask, render_template, request, jsonify
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import os
import numpy as np
from ultralytics import YOLO
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import subprocess
from pathlib import Path
import threading
import uuid

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_FOLDER = BASE_DIR / "static" / "uploads"
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10MB
ALLOWED_IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

device = torch.device("cpu")
_inference_lock = threading.Lock()
YOLO_CONF = 0.15
YOLO_IMGSZ = 640


def convert_video(input_path, output_path):

    command = [
        "ffmpeg",
        "-y",
        "-i", input_path,
        "-vcodec", "libx264",
        "-acodec", "aac",
        output_path
    ]

    subprocess.run(command, check=True)  
# ================================
# GNN MODEL (SLIDER PREDICTION)
# ================================
class TrafficGNN(torch.nn.Module):

    def __init__(self):
        super(TrafficGNN,self).__init__()

        self.conv1 = GCNConv(4,32)
        self.conv2 = GCNConv(32,32)
        self.conv3 = GCNConv(32,16)
        self.conv4 = GCNConv(16,2)

    def forward(self,data):

        x,edge_index = data.x,data.edge_index

        x = self.conv1(x,edge_index)
        x = F.relu(x)

        x = self.conv2(x,edge_index)
        x = F.relu(x)

        x = self.conv3(x,edge_index)
        x = F.relu(x)

        x = self.conv4(x,edge_index)

        return F.log_softmax(x,dim=1)

gnn_model = TrafficGNN()
gnn_model.load_state_dict(torch.load(str(BASE_DIR / "traffic_gnn_model.pth"), map_location=device))
gnn_model.eval()

node_map = {"Node 1":0,"Node 2":1,"Node 3":2,"Node 4":3}

edge_index = torch.tensor([
[0,1,1,2,2,3],
[1,0,2,1,3,2]
],dtype=torch.long)

data = Data(x=torch.zeros((4,4)), edge_index=edge_index)

# ================================
# YOLO MODEL
# ================================
yolo = YOLO(str(BASE_DIR / "yolov8l.pt"))

# YOLO class-name mapping can vary slightly across weight versions
vehicle_classes = {
    "car",
    "bus",
    "truck",
    "motorcycle",
    "motorbike",
}
vehicle_class_ids = [
    cls_id
    for cls_id, name in yolo.model.names.items()
    if name in vehicle_classes
]
# If this list ends up empty, the app might never count vehicles.
# In that case we fall back to counting all detected boxes for congestion.
if len(vehicle_class_ids) == 0:
    vehicle_class_ids = None

roads = ["Top Left","Top Right","Bottom Left","Bottom Right"]

# ================================
# EGNN MODEL
# ================================
class EGNN(nn.Module):

    def __init__(self):
        super(EGNN,self).__init__()

        self.gcn1 = GCNConv(1,16)
        self.gcn2 = GCNConv(16,8)

        self.lstm = nn.LSTM(input_size=8,hidden_size=16,batch_first=True)

        self.fc = nn.Linear(16,1)

    def forward(self,x,edge_index):

        x = self.gcn1(x,edge_index)
        x = torch.relu(x)

        x = self.gcn2(x,edge_index)
        x = torch.relu(x)

        x_seq = x.unsqueeze(0)

        lstm_out,_ = self.lstm(x_seq)

        out = self.fc(lstm_out.squeeze(0))

        return out

egnn_model = EGNN()
EGNN_EDGE_INDEX = torch.tensor(
    [[0, 0, 1, 1, 2, 2, 3, 3], [1, 2, 0, 3, 0, 3, 1, 2]], dtype=torch.long
)

# ================================
# SLIDER PREDICTION
# ================================
def slider_prediction(node,r1,r2,r3,r4):

    avg = (r1+r2+r3+r4)/4

    # Return explicit 3-class congestion levels.
    # This keeps the UI consistent and avoids mixing binary GNN outputs
    # with 3-class labels.
    if avg < 40:
        return "LOW"
    if avg > 50:
        return "HIGH"
    return "MEDIUM"

# ================================
# FRAME ANALYSIS (YOLO + EGNN)
# ================================
def analyze_frame(frame):

    h,w = frame.shape[:2]

    road_density=[0,0,0,0]

    def get_road(x,y):

        mid_x=w//2
        mid_y=h//2

        if x<mid_x and y<mid_y: return 0
        elif x>=mid_x and y<mid_y: return 1
        elif x<mid_x and y>=mid_y: return 2
        else: return 3

    with _inference_lock:
        yolo_kwargs = {
            "conf": YOLO_CONF,
            "imgsz": YOLO_IMGSZ,
            "verbose": False,
        }
        if vehicle_class_ids is not None:
            yolo_kwargs["classes"] = vehicle_class_ids

        results = yolo.predict(frame, **yolo_kwargs)

        for r in results:
            if r.boxes is None or len(r.boxes) == 0:
                continue

            for box,cls in zip(r.boxes.xyxy,r.boxes.cls):

                name = yolo.model.names[int(cls)]

                if vehicle_class_ids is None or name in vehicle_classes:

                    x1,y1,x2,y2 = map(int,box)

                    cx=(x1+x2)//2
                    cy=(y1+y2)//2

                    road=get_road(cx,cy)

                    road_density[road]+=1

                    cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)

    # Normalize road density to make EGNN output less sensitive to
    # absolute counts (helps avoid constant "MEDIUM").
    road_tensor = torch.tensor(road_density, dtype=torch.float32)
    if road_tensor.sum().item() > 0:
        road_tensor = road_tensor / (road_tensor.sum() + 1e-6)
    node_features = road_tensor.view(4, 1)

    with _inference_lock, torch.inference_mode():
        pred = egnn_model(node_features,EGNN_EDGE_INDEX)

    # EGNN output is an unbounded score. Normalize to [0,1].
    pred_scores = torch.sigmoid(pred).view(-1)  # shape: [4]
    avg_score = pred_scores.mean().item()
    max_score = pred_scores.max().item()

    # Density-based congestion (more reliable than EGNN bucketing when
    # EGNN outputs are near-constant).
    total_vehicles = sum(road_density)
    # Tight thresholds to avoid collapsing into MEDIUM.
    # Tune these later using debug.total_vehicles values from the UI.
    if total_vehicles <= 2:
        density_congestion = "LOW"
    elif total_vehicles <= 4:
        density_congestion = "MEDIUM"
    else:
        density_congestion = "HIGH"

    # EGNN-based congestion.
    if max_score < 0.45:
        egcnn_congestion = "LOW"
    elif max_score > 0.60:
        egcnn_congestion = "HIGH"
    else:
        egcnn_congestion = "MEDIUM"

    # For now: use density rule as final congestion label for stability.
    congestion = density_congestion

    green_index = torch.argmax(pred_scores).item()

    green_road=roads[green_index]

    mid_x=w//2
    mid_y=h//2

    cv2.line(frame,(mid_x,0),(mid_x,h),(255,0,0),3)
    cv2.line(frame,(0,mid_y),(w,mid_y),(255,0,0),3)

    return (
        congestion,
        green_road,
        frame,
        road_density,
        total_vehicles,
        avg_score,
        max_score,
        density_congestion,
        egcnn_congestion,
    )

# ================================
# FRAME PREDICTION ROUTE
# ================================
@app.route("/predict_frame", methods=["POST"])
def predict_frame():
    if "frame" not in request.files:
        return jsonify({"error":"Missing form field 'frame'"}), 400

    f = request.files["frame"]
    original_name = f.filename or ""
    ext = os.path.splitext(original_name)[1].lower()
    mimetype = (f.mimetype or "").lower()

    if (ext and ext not in ALLOWED_IMAGE_EXTS) and not mimetype.startswith("image/"):
        return jsonify({"error":"Unsupported image type"}), 400

    raw = f.read()
    img_arr = np.frombuffer(raw, np.uint8)
    frame = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)

    if frame is None:
        return jsonify({"error":"Invalid image"}), 400

    (
        congestion,
        green_road,
        annotated_frame,
        road_density,
        total_vehicles,
        avg_score,
        max_score,
        density_congestion,
        egcnn_congestion,
    ) = analyze_frame(frame)

    uid = uuid.uuid4().hex
    out_name = f"result_{uid}.jpg"
    out_path = os.path.join(app.config["UPLOAD_FOLDER"], out_name)

    ok = cv2.imwrite(out_path, annotated_frame)
    if not ok:
        return jsonify({"error":"Failed to write result image"}), 500

    return jsonify({
        "congestion":congestion,
        "green":green_road,
        "image":f"uploads/{out_name}",
        "debug": {
            "road_density": road_density,
            "total_vehicles": total_vehicles,
            "avg_score": avg_score,
            "max_score": max_score,
            "density_congestion": density_congestion,
            "egcnn_congestion": egcnn_congestion,
            "vehicle_class_filter_used": vehicle_class_ids is not None,
        }
    })

# ================================
# HOME PAGE
# ================================
@app.route("/",methods=["GET","POST"])
def home():

    slider_result=None

    if request.method=="POST":

        node=request.form["node"]

        if node not in node_map:
            return render_template(
                "index.html",
                nodes=node_map.keys(),
                slider_result="Invalid node selected."
            )

        try:
            r1=float(request.form["r1"])
            r2=float(request.form["r2"])
            r3=float(request.form["r3"])
            r4=float(request.form["r4"])
        except (KeyError, ValueError):
            return render_template(
                "index.html",
                nodes=node_map.keys(),
                slider_result="Invalid slider input values."
            )

        slider_result=slider_prediction(node,r1,r2,r3,r4)

    return render_template(
        "index.html",
        nodes=node_map.keys(),
        slider_result=slider_result
    )

    

if __name__=="__main__":
    app.run(debug=False)