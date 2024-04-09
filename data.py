import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
from tkinter import ttk


true_labels = []  # List to store true labels
predicted_labels = []  # List to store predicted labels
centroid_data = []  # list na ukladanie zhlukov

# na farby
def detect_red_objects(frame):
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])
    mask1 = cv2.inRange(hsv_frame, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv_frame, lower_red2, upper_red2)
    red_mask = mask1 + mask2
    contours, _ = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours based on area
    significant_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 400]
    data_points = [cv2.moments(cnt) for cnt in significant_contours if cnt is not None]

    # Centroids
    centroids = []
    for M in data_points:
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            centroids.append([cX, cY])

    centroids_np = np.array(centroids)

    # Draw bounding boxes and label objects 
    if len(centroids) > 0:
        centroids_np = np.array(centroids)
        n_clusters = min(len(centroids), 2)  # Maximum number of clusters
        if len(centroids) > 1:
            kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(centroids_np)
            labels = kmeans.labels_

            cluster_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)]

            for i, cnt in enumerate(significant_contours):
                x, y, w, h = cv2.boundingRect(cnt)
                cluster_color = cluster_colors[labels[i] % len(cluster_colors)]
                cv2.rectangle(frame, (x, y), (x + w, y + h), cluster_color, 2)
                cv2.putText(frame, f'', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, cluster_color, 2)
        else:  # If only one cluster
            x, y, w, h = cv2.boundingRect(significant_contours[0])
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
            cv2.putText(frame, '', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    return frame, centroids_np

def process_camera_feed():
    global centroid_data
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Camera cannot be used")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        processed_frame, centroids = detect_red_objects(frame)
        centroid_data.extend(centroids.tolist())
        cv2.imshow('Object Detection', processed_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def plot_clustered_data():
    global centroid_data
    if not centroid_data:
        print("No objects were found.")
        return
    centroid_array = np.array(centroid_data)
    n_clusters = min(len(centroid_array), 2)  # Determine the number of clusters dynamically, with a maximum
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(centroid_array)

    # Get unique labels and their corresponding centroids
    unique_labels, counts = np.unique(kmeans.labels_, return_counts=True)
    centroids_per_label = [centroid_array[kmeans.labels_ == label] for label in unique_labels]

    # Scatter plot of clustered data with distinct colors for each cluster
    for label, centroids in zip(unique_labels, centroids_per_label):
        plt.scatter(centroids[:, 0], centroids[:, 1], label=f'Cluster {label}')

    # Adding legend
    plt.legend()

    # Adding title and labels
    plt.title('Clustering')
    plt.xlabel('X')
    plt.ylabel('Y')

    # Displaying the plot
    plt.show()

def close_window():
    root.destroy()

def main():
    global root
    root = tk.Tk()
    root.title("Object Detection")

    frame = tk.Frame(root)
    frame.pack(pady=20)

    def process_selected_option():
        process_camera_feed()
        plot_clustered_data()

    button = tk.Button(frame, text="Camera (HSV)", command=process_selected_option)
    button.pack(pady=20)

    close_button = tk.Button(root, text="Close", command=close_window)
    close_button.pack(side="bottom", pady=20)

    root.geometry("500x500")  # Set window size

    root.mainloop()

if __name__ == "__main__":
    main()
