import os
from ultralytics import YOLO
from hailo_sdk_client import ClientRunner

model_name = "yolo11s"
pt_path = "yolo11s.pt"
onnx_path = "yolo11s.onnx"
hef_path = "yolo11s.hef"

def convert_pt_to_hef():
    # Step 1: Export PT to ONNX (using Ultralytics)
    print(f"--- Exporting {pt_path} to ONNX ---")
    model = YOLO(pt_path)
    model.export(format="onnx", imgsz=640, opset=11)

    # Step 2: Initialize Hailo Runner
    runner = ClientRunner()

    # Step 3: Translate ONNX to Hailo representation
    print("--- Translating ONNX to Hailo ---")
    runner.translate_onnx_model(onnx_path, model_name)

    # Step 4: Optimize
    # Note: For real accuracy, you'd load a calibration dataset here
    print("--- Optimizing ---")
    runner.optimize()

    # Step 5: Compile
    print("--- Compiling to HEF ---")
    hef = runner.compile()

    with open(hef_path, "wb") as f:
        f.write(hef)
    
    print(f"Done! {hef_path} is ready for use on the Raspberry Pi AI Kit.")

if __name__ == "__main__":
    convert_pt_to_hef()
