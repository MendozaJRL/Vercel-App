import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO

# Define the label map
label_map = {
    'Flowering': 'Growing',
    'Vegetative': 'Growing',
    'Germination': 'Germination',
    'Harvesting': 'Harvesting'
}

# Function to preprocess image for Thurinus Lettuce
def preprocess_thurinus(image):
    hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv_img[:, :, 0] = 60  # Change hue to a greenish value
    green_hued_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)
    return green_hued_img

# Streamlit app
def main():
    st.title("Lumina Flora: Plant Growth Stage Detection Model")
    st.write("The plant growth stage detection model is a YOLOv8 Model that is capable of detecting plant growth stages: germination, growing, and harvesting.")

    # Load your YOLOv8 model
    model = YOLO("40 Epoch Plant Growth Stage YOLOv8 Model.pt")
    
    # Provide options for users to choose from
    option = st.selectbox("Select Level:", ["None", "Olmetie Lettuce", "Thurinus Lettuce"])
    
    # Conditional display for file uploader based on plant type selection
    if option != "None":
        st.write(f"Plant Type: {option}")
        uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
        
        if uploaded_file:
            # Read the uploaded image as a numpy array
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            original_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)  # Original image
            
            # Display the original uploaded image
            st.image(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB), caption="Uploaded Image", use_column_width=True)
            
            # Preprocess the image only if the plant type is Thurinus Lettuce
            if option == "Thurinus Lettuce":
                image = preprocess_thurinus(original_image)  # Apply preprocessing only for Thurinus Lettuce
            else:
                image = original_image  # Use original image if no preprocessing is required
            
            # Detect growth stage button and results display
            if st.button("Detect Growth Stage"):
                # Make predictions
                results = model.predict(source=image, save=False, conf=0.25)  # Adjust confidence threshold as needed
                
                # Initialize a list to store detection results
                detection_results = []
                
                # Get the original image dimensions
                annotated_image = original_image.copy()  # Use the original image for annotation
                
                # Loop through detections and draw bounding boxes with OpenCV
                for result in results[0].boxes.data.tolist():  # Assuming YOLOv8 outputs boxes as list
                    x1, y1, x2, y2, confidence, class_id = result[0], result[1], result[2], result[3], result[4], result[5]
                    confidence = round(confidence, 2)  # Round confidence to 2 decimal places
                    class_name = model.names[class_id]
                    
                    # Map the detected label using label_map
                    class_name = label_map.get(class_name, class_name)
                    
                    # Ensure the bounding box coordinates are integers
                    x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
                    
                    # Draw bounding box with increased thickness
                    cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (255, 255, 0), 4)
                    
                    # Put label with larger font size and thickness
                    label = f"{class_name} ({confidence:.2f})"
                    font_scale = 1.0  # Larger font scale
                    font_thickness = 2  # Thicker font
                    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
                    text_x, text_y = x1, y1 - 10
                    label_bg_x2 = text_x + text_size[0] + 4
                    label_bg_y2 = text_y + text_size[1] + 4
                    
                    # Draw background rectangle for text
                    cv2.rectangle(annotated_image, (text_x - 2, text_y - text_size[1] - 4), 
                                  (label_bg_x2, label_bg_y2), (255, 255, 0), -1)
                    cv2.putText(annotated_image, label, (text_x, text_y), 
                                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thickness)
                    
                    # Append the result to the list
                    detection_results.append({
                        'Label': class_name,
                        'Confidence': confidence,
                        'Bounding Box': (x1, y1, x2, y2)
                    })
                
                # Display the model results using st.write()
                if detection_results:
                    st.write("### Detection Results:")
                    for result in detection_results:
                        st.write(f"**Label**: {result['Label']}")
                        st.write(f"   **Confidence**: {result['Confidence']}")
                        st.write("-----")
                
                # Convert the annotated image to RGB for display in Streamlit
                st.image(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB), caption="Detected Growth Stages", use_column_width=True)
    
    st.write("")
    st.markdown("Made by Team 45")
                    
# Run the app
if __name__ == '__main__':
    main()
