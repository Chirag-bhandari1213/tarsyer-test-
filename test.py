import cv2

# Path to the Haar cascade XML file for vehicle detection
haar_cascade_path = '/Users/chiragbhandari/Downloads/tarsyer/cars.xml'

# Path to the input videoq
video_path = '/Users/chiragbhandari/Downloads/tarsyer/cars_on_highway (1080p).mp4'

# Load the Haar cascade classifier for vehicle detection
vehicle_cascade = cv2.CascadeClassifier(haar_cascade_path)

# Open the video file
cap = cv2.VideoCapture(video_path)

# Initialize vehicle count
vehicle_count = 0

while True:
    # Read frames from the video
    ret, frame = cap.read()

    if not ret:
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Perform vehicle detection
    vehicles = vehicle_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw rectangles around the detected vehicles
    for (x, y, w, h) in vehicles:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Update the vehicle count
    vehicle_count += len(vehicles)

    # Display the frame with detected vehicles
    cv2.imshow("Vehicle Detection", frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) == ord('q'):
        break

# Release the video capture
cap.release()
cv2.destroyAllWindows()

# Print the total vehicle count
print("Total vehicles detected:", vehicle_count)
