import numpy as np
import cv2
import os
from hailo_platform import HEF, VDevice, InferVStreams, ConfigureParams, InputVStreamParams, OutputVStreamParams, HailoStreamInterface

# --- CONFIGURATION ---
MODEL_PATH = "models/yolov8s.hef"
IMAGE_PATH = "test.jpg"      
THRESHOLD = 0.3                   

def run_test():
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file {MODEL_PATH} not found.")
        return

    # 1. Initialize Device
    hef = HEF(MODEL_PATH)
    target = VDevice()

    # 2. Configure Network
    params = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
    network_group = target.configure(hef, params)[0]
    input_v_params = InputVStreamParams.make(network_group)
    output_v_params = OutputVStreamParams.make(network_group)

    # 3. Get Input Info
    input_info = hef.get_input_vstream_infos()[0]
    height, width, _ = input_info.shape
    print(f"Model input: {width}x{height}")

    # 4. Prepare Image
    raw_img = cv2.imread(IMAGE_PATH)
    if raw_img is None:
        print("Test image not found, using dummy.")
        raw_img = np.zeros((height, width, 3), dtype=np.uint8)
    
    resized_img = cv2.resize(raw_img, (width, height))
    input_data = {input_info.name: np.expand_dims(resized_img, axis=0).astype(np.uint8)}

    # 5. Run Inference
    print("Running inference...")
    with network_group.activate():
        with InferVStreams(network_group, input_v_params, output_v_params) as pipeline:
            raw_results = pipeline.infer(input_data)

    # 6. FIXED POST-PROCESSING
    # Locate the NMS output key
    output_key = [k for k in raw_results.keys() if 'nms' in k.lower()][0]
    detections_by_class = raw_results[output_key]

    # detections_by_class is typically a list where detections_by_class[0] 
    # contains 80 arrays (one for each class).
    batch_detections = detections_by_class[0] 
    
    found_count = 0
    orig_h, orig_w = raw_img.shape[:2]

    # Iterate through each of the 80 classes
    for class_id, class_detections in enumerate(batch_detections):
        # class_detections is an array of [ymin, xmin, ymax, xmax, score]
        for det in class_detections:
            score = det[4]
            if score >= THRESHOLD:
                found_count += 1
                ymin, xmin, ymax, xmax = det[0:4]
                
                # Scale to original image
                left = int(xmin * orig_w)
                top = int(ymin * orig_h)
                right = int(xmax * orig_w)
                bottom = int(ymax * orig_h)

                cv2.rectangle(raw_img, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(raw_img, f"ID:{class_id} {score:.2f}", (left, top - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    print(f"Detected {found_count} objects above threshold.")

    # 7. Save/Show
    cv2.imwrite("output_test.jpg", raw_img)
    print("Result saved as 'output_test.jpg'.")
    try:
        cv2.imshow("Hailo Test", raw_img)
        cv2.waitKey(0)
    except:
        print("Could not open window (Headless mode). Check output_test.jpg")

if __name__ == "__main__":
    run_test()
