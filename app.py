# import streamlit as st
# import cv2
# import numpy as np

# def mark_answer_circles(image, answer_key, output_path="marked_answer_circles_with_fill.png"):
#     # Convert the image to grayscale for better intensity analysis
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#     # Apply adaptive thresholding to improve circle visibility
#     thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)

#     # Use Hough Circle Transform to detect circles
#     circles = cv2.HoughCircles(thresh, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30,
#                                param1=50, param2=30, minRadius=15, maxRadius=30)

#     # Initialize a list to store circle data (for debug output)
#     circle_data = []

#     if circles is not None:
#         circles = np.round(circles[0, :]).astype("int")

#         # Sort circles based on their vertical position (y-coordinate)
#         circles_sorted_by_y = sorted(circles, key=lambda x: x[1])

#         # Group circles into rows (based on y-coordinate clustering)
#         rows = []
#         threshold_y_distance = 50
#         current_row = []
#         previous_y = circles_sorted_by_y[0][1]

#         for (x, y, r) in circles_sorted_by_y:
#             if abs(y - previous_y) < threshold_y_distance:
#                 current_row.append((x, y, r))
#             else:
#                 rows.append(sorted(current_row, key=lambda x: x[0]))  # Sort by x (left to right)
#                 current_row = [(x, y, r)]  # Start a new row
#             previous_y = y

#         if current_row:
#             rows.append(sorted(current_row, key=lambda x: x[0]))

#         flattened_circles = [circle for row in rows for circle in row]
#         letters = ['A', 'B', 'C', 'D']
#         mcq_number = 1
#         marked_answers = []

#         for idx, (x, y, r) in enumerate(flattened_circles):
#             letter = letters[idx % 4]
#             mcq_number = (idx // 4) + 1

#             mask = np.zeros_like(gray)
#             cv2.circle(mask, (x, y), r, 255, -1)
#             masked_region = cv2.bitwise_and(gray, gray, mask=mask)

#             roi = masked_region[y - r:y + r, x - r:x + r]

#             non_zero_pixels = np.count_nonzero(roi)
#             mean_intensity = np.mean(roi) if non_zero_pixels > 0 else 0

#             status = "Filled" if (mean_intensity < 75) else "Unfilled"
#             circle_data.append({
#                 'MCQ Number': mcq_number,
#                 'Letter': letter,
#                 'Intensity': mean_intensity,
#                 'Non-zero Pixels': non_zero_pixels,
#                 'Status': status
#             })

#             circle_color = (0, 0, 255) if status == "Filled" else (0, 255, 0)
#             cv2.circle(image, (x, y), r, circle_color, 4)
#             cv2.putText(image, f"{letter}: {status}", (x - 10, y + 10),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

#             if status == "Filled":
#                 marked_answers.append((mcq_number, letter))

#     # Save the output image with marked circles
#     cv2.imwrite(output_path, image)

#     return marked_answers

# def calculate_score(marked_answers, answer_key):
#     score = 0
#     for mcq_number, letter in marked_answers:
#         if mcq_number <= len(answer_key):
#             if letter == answer_key[mcq_number - 1]:
#                 score += 1  # Correct answer
#             else:
#                 score += 0  # Incorrect answer
#     return score

# def calculate_negative_marking(marked_answers, answer_key):
#     total_score = 0
#     for mcq_number, selected_answers in marked_answers.items():
#         correct_answer = answer_key[mcq_number - 1]  # The correct answer for this MCQ

#         # If no answer is marked for this MCQ, skip it (no penalty)
#         if len(selected_answers) == 0:
#             continue  

#         # Handle the case where multiple options are selected for this MCQ
#         # selected_answers should only contain the answers marked for this specific row (MCQ)
#         correct_selected = [ans for ans in selected_answers if ans == correct_answer]

#         # Case where only one option is selected
#         if len(selected_answers) == 1:
#             if correct_selected:  # If the selected answer is correct
#                 total_score += 1  # Correct answer
#             else:  # Incorrect answer
#                 total_score += 0  # No points, no penalty

#         # Case where multiple options are selected for this MCQ
#         else:
#             if len(correct_selected) > 0:  # If any of the selected answers is correct
#                 total_score -= 0.25 * (len(selected_answers)-1)  # Deduct penalty for multiple answers
#                 total_score += 1  # Add 1 point for the correct answer
#             else:  # All answers are wrong
#                 total_score -= 0.25 * (len(selected_answers)-1)  # Deduct penalty for all wrong answers

#     return total_score

# def app():
#     st.title('MCQ Answer Sheet Scorer')

#     # Input: Answer key for 10 MCQs (using dropdown or radio buttons for each question)
#     st.subheader("Please select the correct answer for each MCQ:")

#     answer_key = []
#     options = ['A', 'B', 'C', 'D']

#     # Use radio buttons for each MCQ answer
#     for i in range(1, 11):
#         answer_key.append(st.radio(f"Question {i}:", options, key=f"q{i}"))

#     # Image upload
#     uploaded_image = st.file_uploader("Upload an Image of the Answer Sheet", type="png")
#     if uploaded_image is not None:
#         # Read the image
#         image = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
#         img = cv2.imdecode(image, cv2.IMREAD_COLOR)

#         # Process the image and detect filled answers
#         output_image_path = "marked_answer_circles_with_fill.png"
#         marked_answers = mark_answer_circles(img, answer_key, output_image_path)

#         # Display the marked image
#         st.image(img, caption="Marked Answer Sheet", use_container_width=True)

#         # Prepare the marked answers dictionary
#         marked_answers_dict = {i: [] for i in range(1, 11)}
#         for mcq_number, letter in marked_answers:
#             marked_answers_dict[mcq_number].append(letter)

#         # Calculate the score with negative marking
#         score = calculate_negative_marking(marked_answers_dict, answer_key)

#         st.write(f"Score: {score}/10")

# if __name__ == '__main__':
#     app()





import streamlit as st
import cv2
import numpy as np

def preprocess_image(image):
    """Prepare the image for circle detection by applying various image processing techniques."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.5)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    edges = cv2.Canny(thresh, 50, 150)
    return gray, edges

def detect_circles(edges):
    """Detect circles in the preprocessed image using Hough Circle Transform."""
    return cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30,
                            param1=50, param2=30, minRadius=15, maxRadius=30)

def group_circles_into_rows(circles):
    """Organize detected circles into rows based on their vertical positions."""
    circles_sorted_by_y = sorted(circles, key=lambda x: x[1])
    rows = []
    threshold_y_distance = 50
    current_row = []
    previous_y = circles_sorted_by_y[0][1]

    for (x, y, r) in circles_sorted_by_y:
        if abs(y - previous_y) < threshold_y_distance:
            current_row.append((x, y, r))
        else:
            rows.append(sorted(current_row, key=lambda x: x[0]))
            current_row = [(x, y, r)]
        previous_y = y
    rows.append(sorted(current_row, key=lambda x: x[0]))

    return rows

def check_circle_filled(gray, x, y, r):
    """Determine if a circle is filled by analyzing its pixel intensity."""
    mask = np.zeros_like(gray)
    cv2.circle(mask, (x, y), r, 255, -1)
    mean_intensity = cv2.mean(gray, mask=mask)[0]
    return mean_intensity < 100

def mark_answer_circles(image, answer_key, output_path="marked_answer_circles_with_fill_final3.png"):
    """Main function to process the image, detect and mark answer circles."""
    if image is None:
        st.error(f"Error: Could not load image")
        return None, None

    gray, edges = preprocess_image(image)
    circles = detect_circles(edges)
    marked_answers = {}
    total_score = 0

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        rows = group_circles_into_rows(circles)
        option_labels = ['A', 'B', 'C', 'D']

        for i, row in enumerate(rows):
            for j, (x, y, r) in enumerate(row):
                is_filled = check_circle_filled(gray, x, y, r)
                
                label = option_labels[j] if j < len(option_labels) else f"Option {j+1}"
                status = 'Filled' if is_filled else 'Unfilled'
                
                if status == 'Filled':
                    circle_color = (0, 0, 255)
                    text_color = (0, 0, 255)
                else:
                    circle_color = (0, 255, 0)
                    text_color = (0, 0, 255)
                
                cv2.circle(image, (x, y), r, circle_color, 2)
                cv2.putText(image, f"{label}: {status}", (x - r, y + r + 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2, cv2.LINE_AA)

                question_number = i + 1
                if question_number not in marked_answers:
                    marked_answers[question_number] = []

                marked_answers[question_number].append((label, is_filled))

        total_score = calculate_score(marked_answers, answer_key)
        cv2.imwrite(output_path, image)
    else:
        st.warning("No circles were detected in the image.")

    return total_score, image

def calculate_score(marked_answers, answer_key):
    """Calculate the score based on marked answers and the correct answer key."""
    total_score = 0
    for question_number, answer_data in marked_answers.items():
        correct_answer = answer_key[question_number - 1]

        for label, is_filled in answer_data:
            if label == correct_answer and is_filled:
                total_score += 1
            elif is_filled:
                total_score -= 0.25

    return total_score

def app():
    """Streamlit application for MCQ Answer Sheet Scorer."""
    st.title('MCQ Answer Sheet Scorer')
    st.subheader("Please select the correct answer for each MCQ:")

    answer_key = []
    options = ['A', 'B', 'C', 'D']

    for i in range(1, 11):
        answer_key.append(st.radio(f"Question {i}:", options, key=f"q{i}"))

    uploaded_image = st.file_uploader("Upload an Image of the Answer Sheet", type=["png", "jpg", "jpeg"])
    
    if uploaded_image is not None:
        image = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
        img = cv2.imdecode(image, cv2.IMREAD_COLOR)

        score, marked_image = mark_answer_circles(img, answer_key)

        if marked_image is not None:
            st.image(marked_image, channels="BGR", caption="Marked Answer Sheet", use_container_width=True)

            if score is not None:
                st.write(f"Score: {score}/10")

if __name__ == '__main__':
    app()
