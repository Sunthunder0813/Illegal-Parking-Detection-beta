import numpy as np
import cv2
import os
from hailo_platform import HEF, VDevice, InferVStreams, ConfigureParams, InputVStreamParams, OutputVStreamParams, HailoStreamInterface

# --- CONFIGURATION ---
MODEL_PATH = "models/yolov8s.hef"  # Change this to your HEF path
IMAGE_PATH = "test.jpg"      # Path to an image you want to test
THRESHOLD = 0.3                   # Confidence threshold for detections

def run_test():
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file {MODEL_PATH} not found.")
        return

    # 1. Initialize the Hailo device
    hef = HEF(MODEL_PATH)
    target = VDevice()

    # 2. Configure the network group
    configure_params = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
    network_group = target.configure(hef, configure_params)[0]
    input_vstreams_params = InputVStreamParams.make(network_group)
    output_vstreams_params = OutputVStreamParams.make(network_group)

    # 3. Get input/output information
    input_info = hef.get_input_vstream_infos()[0]
    height, width, channels = input_info.shape
    print(f"Model expects input: {width}x{height}")

    # 4. Prepare the test image
    raw_img = cv2.imread(IMAGE_PATH)
    if raw_img is None:
        print(f"No image found at {IMAGE_PATH}. Creating a dummy black image for testing.")
        raw_img = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Resize and prepare for Hailo (UINT8, NHWC)
    resized_img = cv2.resize(raw_img, (width, height))
    input_data = {input_info.name: np.expand_dims(resized_img, axis=0).astype(np.uint8)}

    # 5. Run Inference
    print("Running inference...")
    with network_group.activate():
        with InferVStreams(network_group, input_vstreams_params, output_vstreams_params) as pipeline:
            raw_results = pipeline.infer(input_data)

    # 6. Post-processing (Generic handling for Hailo NMS)
    output_key = [k for k in raw_results.keys() if 'nms' in k.lower()][0]
    detections = raw_results[output_key]

    # Normalize detections to [N, 6] format
    if isinstance(detections, list):
        detections = np.array(detections[0])
    if detections.ndim == 3:
        detections = detections[0]

    print(f"Found {len(detections)} potential detections.")

    # 7. Visualize Results
    for det in detections:
        if len(det) >= 6:
            ymin, xmin, ymax, xmax, score, class_id = det
            if score > THRESHOLD:
                # Scale coordinates back to original image size
                orig_h, orig_w = raw_img.shape[:2]
                left = int(xmin * orig_w)
                top = int(ymin * orig_h)
                right = int(xmax * orig_w)
                bottom = int(ymax * orig_h)

                cv2.rectangle(raw_img, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(raw_img, f"ID:{int(class_id)} {score:.2f}", (left, top - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 8. Save and display
    cv2.imwrite("output_test.jpg", raw_img)
    print("Testing complete. Result saved as 'output_test.jpg'.")
    cv2.imshow("Hailo Detection Test", raw_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_test()
