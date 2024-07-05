import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
import math
import time
import tempfile
from PIL import Image

def linear_equation(x1, y1, x2, y2):
    if x1 == x2:
        return float('inf'), y1
    a = (y2 - y1) / (x2 - x1)
    b = y1 - a * x1
    return a, b

# Helper function to check if a point lies on a line
def check_point_linear(x, y, x1, y1, x2, y2):
    a, b = linear_equation(x1, y1, x2, y2)
    if a == float('inf'):  # Handle vertical line case
        return abs(x - x1) < 3
    y_pred = a * x + b
    return math.isclose(y_pred, y, abs_tol=3)

def read_label(x):
    label = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    if x == 29:
        return '0'
    elif label[x + 1] == 'L':
        return 'N'
    elif label[x + 1] == 'J':
        return 'L'
    elif label[x + 1] == 'Q':
        return 'V'
    elif label[x + 1] == 'I':
        return 'K'
    elif label[x + 1] == 'S':
        return 'I'
    elif label[x + 1] == 'R':
        return 'X'
    elif label[x + 1] == 'T':
        return 'Z'
    elif label[x + 1] == 'O':
        return 'T'
    elif label[x + 1] == 'M':
        return 'P'
    elif label[x + 1] == 'K':
        return 'M'
    elif label[x + 1] == 'P':
        return 'U'
    elif label[x + 1] == 'N':
        return 'S'
    else:
        return label[x + 1]

# Hàm đọc kí tự biển số
def read_plate(yolo_license_plate, im):
    LP_type = "1"
    results = yolo_license_plate(im)
    bb_list = results[0].boxes.xyxy.cpu().numpy()  # Adjust to access the first result and get bounding boxes
    classes = results[0].boxes.cls.cpu().numpy()
    label = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

    if len(bb_list) == 0 or len(bb_list) < 7 or len(bb_list) > 10:
        return "unknown"
    center_list = []
    y_sum = 0
    for i, bb in enumerate(bb_list):
        x_c = (bb[0] + bb[2]) / 2
        y_c = (bb[1] + bb[3]) / 2
        y_sum += y_c
        center_list.append([x_c, y_c, int(classes[i])])

    # find 2 point to draw line
    l_point = min(center_list, key=lambda c: c[0])
    r_point = max(center_list, key=lambda c: c[0])

    for ct in center_list:
        if l_point[0] != r_point[0]:
            if not check_point_linear(ct[0], ct[1], l_point[0], l_point[1], r_point[0], r_point[1]):
                LP_type = "2"
                break

    y_mean = y_sum / len(center_list)

    line_1 = []
    line_2 = []
    license_plate = ""
    if LP_type == "2":
        for c in center_list:
            if c[1] > y_mean:
                line_2.append(c)
            else:
                line_1.append(c)
        for l1 in sorted(line_1, key=lambda x: x[0]):
            license_plate += read_label(l1[2])
        license_plate += "-"
        for l2 in sorted(line_2, key=lambda x: x[0]):
            license_plate += read_label(l2[2])
    else:
        for l in sorted(center_list, key=lambda x: x[0]):
            license_plate += read_label(l[2])
            if len(license_plate) == 3:
                license_plate += "-"

    return license_plate

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Load local CSS
local_css("style.css")



# Load models
plate_model = YOLO('Module/bestLP.pt')
char_model = YOLO('Module/best.pt')

# Title and description
st.title('🔍 License Plate Recognition')
st.write("This application uses YOLO models to detect and recognize license plates from images and videos. Upload an image or video to get started.")

# Sidebar for input selection
st.sidebar.title("Options")
option = st.sidebar.selectbox('Choose input type:', ('Image', 'Video'))

# Image processing
if option == 'Image':
    st.sidebar.write("### Upload an Image")
    uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "webp"])
    if uploaded_file is not None:
        st.markdown("### Detection Results:")
        img = Image.open(uploaded_file)
        img_array = np.array(img)

        # Predict license plate location
        plate_results = plate_model.predict(source=img_array)
        for plate_result in plate_results:
            plate_boxes = plate_result.boxes
            for plate_box in plate_boxes:
                x1, y1, x2, y2 = plate_box.xyxy[0].cpu().numpy().astype(int)
                cropped_plate = img_array[y1:y2, x1:x2]

                # Predict characters in the license plate
                detected_text = read_plate(char_model, cropped_plate)
                st.image(cropped_plate, caption='License Plate')
                st.write("**Detected License Plate Text:**", detected_text)

                # Draw the bounding box and detected text on the original image
                img_array = cv2.rectangle(img_array, (x1, y1), (x2, y2), (255, 0, 0), 2)
                img_array = cv2.putText(img_array, detected_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 2)
        st.markdown("#### Original Image After Detection")
        st.image(img_array, caption='Processed Image')

# Video processing
elif option == 'Video':
    st.sidebar.write("### Upload a Video")
    video_file = st.sidebar.file_uploader("Choose a video...", type=["mp4", "mov", "avi", "mkv"])
    if video_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())
        vid = cv2.VideoCapture(tfile.name)

        stframe = st.empty()
        prev_frame_time = 0

        results = set()
        cnt = 0
        while vid.isOpened():
            ret, frame = vid.read()
            if not ret:
                break
            cnt += 1
            if cnt % 3 != 0:
                continue
            plates = plate_model.predict(source=frame, imgsz=640)
            for plate in plates:
                plate_boxes = plate.boxes
                for plate_box in plate_boxes:
                    x1, y1, x2, y2 = plate_box.xyxy[0].cpu().numpy().astype(int)  # Coordinates of top-left and bottom-right corners
                    cropped_plate = frame[y1:y2, x1:x2]
                    lp = read_plate(char_model, cropped_plate)
                    if lp != "unknown" and lp not in results:
                        results.add(lp)
                        cv2.putText(frame, lp, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
                        st.image(cropped_plate, caption='Detected License Plate')
                        st.write("**Detected License Plate Text:**", lp)
                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            new_frame_time = time.time()
            fps = 1 / (new_frame_time - prev_frame_time)
            prev_frame_time = new_frame_time
            fps = int(fps)
            cv2.putText(frame, str(fps), (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (100, 255, 0), 3, cv2.LINE_AA)

            # Display the frame with bounding box and recognized text
            stframe.image(frame, channels="BGR")
