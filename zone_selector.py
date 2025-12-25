import cv2
import numpy as np
import json
import sys

# Camera RTSP URLs
CAM1_URL = "rtsp://192.168.18.2:554/stream"
CAM2_URL = "rtsp://192.168.18.199:554/stream"

window_name = "Zone Selector"
points = []
zone_var_name = ""

def mouse_callback(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        print(f"Point added: ({x}, {y})")

def main():
    global points, zone_var_name
    # Accept camera selection from command line argument
    if len(sys.argv) > 1:
        cam_choice = sys.argv[1]
    else:
        print("Select camera to use for zone selection:")
        print("1 - Camera 1")
        print("2 - Camera 2")
        cam_choice = input("Enter 1 or 2: ").strip()
    if cam_choice == "1":
        cam_url = CAM1_URL
        zone_var_name = "ZONE_CAM1"
    elif cam_choice == "2":
        cam_url = CAM2_URL
        zone_var_name = "ZONE_CAM2"
    else:
        print("Invalid selection.")
        return

    cap = cv2.VideoCapture(cam_url)
    if not cap.isOpened():
        print("Failed to open camera or image.")
        return

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    print("Instructions:")
    print("- Click to add points for the zone polygon.")
    print("- Press 'u' to undo last point.")
    print("- Press 'c' to clear all points.")
    print("- Press 'q' to quit and print the coordinates.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # Draw points and polygon
        for pt in points:
            cv2.circle(frame, pt, 5, (0, 0, 255), -1)
        if len(points) > 1:
            cv2.polylines(frame, [np.array(points, np.int32)], isClosed=True, color=(0, 255, 0), thickness=2)

        cv2.imshow(window_name, frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('u') and points:
            points.pop()
        elif key == ord('c'):
            points = []
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if points:
        print(json.dumps(points))  # Print only the coordinates as JSON array
    else:
        print("[]")

if __name__ == "__main__":
    main()
