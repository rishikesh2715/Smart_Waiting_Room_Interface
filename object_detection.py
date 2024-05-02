from typing import List
import cv2
import numpy as np
from ultralytics import YOLO
from utils.general import find_in_list, load_zones_config
from utils.timers import FPSBasedTimer
import supervision as sv
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
import matplotlib.pyplot as plt

COLORS = sv.ColorPalette.from_hex(["#E6194B", "#3CB44B", "#FFE119", "#3C76D1"])
COLOR_ANNOTATOR = sv.BoundingBoxAnnotator(color=COLORS)
LABEL_ANNOTATOR = sv.LabelAnnotator(color=COLORS, text_color=sv.Color.from_hex("#000000"))

def main(
    source_video_path: str,
    zone_configuration_path: str,
    weights: str,
    device: str,
    confidence: float,
    iou: float,
    classes: List[int],
) -> None:
    model = YOLO(weights)
    tracker = sv.ByteTrack(minimum_matching_threshold=0.3)
    video_info = sv.VideoInfo.from_video_path(video_path=source_video_path)
    frames_generator = sv.get_video_frames_generator(source_video_path)
    polygons = load_zones_config(file_path=zone_configuration_path)
    zones = [
        sv.PolygonZone(polygon=polygon, triggering_anchors=(sv.Position.BOTTOM_CENTER,))
        for polygon in polygons
    ]
    timers = [FPSBasedTimer(video_info.fps) for _ in zones]

    # Initialize Kalman Filters for each zone with initial estimate of 10 minutes
    initial_estimate = 10 * 60  # Convert minutes to seconds
    kfs = [KalmanFilter(dim_x=2, dim_z=1) for _ in zones]
    for kf in kfs:
        kf.x = np.array([initial_estimate, 0.])  # initial state (location and velocity)
        kf.F = np.array([[1., 1.],  # state transition matrix
                         [0., 1.]])
        kf.H = np.array([[1., 0.]])  # measurement function
        kf.P *= np.array([[1000., 0.],
                          [0., 1000.]])  # covariance matrix
        kf.R = np.array([[5.]])  # measurement uncertainty
        kf.Q = Q_discrete_white_noise(dim=2, dt=0.1, var=0.13)  # process uncertainty

    # Initialize variables for averaging approach
    total_wait_times = [0] * len(zones)
    num_detections_left = [0] * len(zones)

    # Initialize lists to store wait times for plotting
    real_wait_times = [[] for _ in zones]
    kalman_wait_times = [[] for _ in zones]
    average_wait_times = [[] for _ in zones]

    for frame in frames_generator:
        results = model(frame, verbose=False, device=device, conf=confidence)[0]
        detections = sv.Detections.from_ultralytics(results)
        detections = detections[find_in_list(detections.class_id, classes)]
        detections = detections.with_nms(threshold=iou)
        detections = tracker.update_with_detections(detections)

        annotated_frame = frame.copy()
        for idx, zone in enumerate(zones):
            annotated_frame = sv.draw_polygon(
                scene=annotated_frame, polygon=zone.polygon, color=COLORS.by_idx(idx)
            )
            detections_in_zone = detections[zone.trigger(detections)]

            if detections_in_zone is not None:
                time_in_zone = timers[idx].tick(detections_in_zone)
                custom_color_lookup = np.full(detections_in_zone.class_id.shape, idx)
                annotated_frame = COLOR_ANNOTATOR.annotate(
                    scene=annotated_frame,
                    detections=detections_in_zone,
                    custom_color_lookup=custom_color_lookup,
                )

                # Update Kalman Filter with new time_in_zone measurement
                kf = kfs[idx]
                kf.update(np.array([time_in_zone[-1]]))
                kf.predict()

                # Get the estimated next time_in_zone from Kalman Filter
                next_time_in_zone_kalman = kf.x[0]

                # Update total wait time and number of detections that have left the zone
                total_wait_times[idx] += time_in_zone[-1]
                num_detections_left[idx] += 1

                # Calculate the average wait time
                average_wait_time = total_wait_times[idx] / num_detections_left[idx] if num_detections_left[idx] > 0 else 0

                # Store wait times for plotting
                real_wait_times[idx].append(time_in_zone[-1])
                kalman_wait_times[idx].append(next_time_in_zone_kalman)
                average_wait_times[idx].append(average_wait_time)

                

                labels = [
                    f"#{tracker_id} {int(time // 60):02d}:{int(time % 60):02d} (Kalman: {int(next_time_in_zone_kalman // 60):02d}:{int(next_time_in_zone_kalman % 60):02d})"
                    for tracker_id, time in zip(detections_in_zone.tracker_id, time_in_zone)
                ]
                annotated_frame = LABEL_ANNOTATOR.annotate(
                    scene=annotated_frame,
                    detections=detections_in_zone,
                    labels=labels,
                    custom_color_lookup=custom_color_lookup,
                )

        cv2.imshow("Processed Video", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()

    # Plot the real wait time, Kalman estimate wait time, and averaging wait time for each zone
    for idx, zone in enumerate(zones):
        plt.figure(figsize=(10, 4))
        plt.plot(real_wait_times[idx], label='Real Wait Time')
        plt.plot(kalman_wait_times[idx], label='Kalman Estimate Wait Time')
        plt.plot(average_wait_times[idx], label='Average Wait Time')
        plt.legend()
        plt.xlabel('Detection Number')
        plt.ylabel('Wait Time (seconds)')
        plt.title(f'Wait Time Comparison - Zone {idx+1}')
        plt.grid(True)
        plt.show()

if __name__ == "__main__":
    # Set the parameter values directly here:
    source_video_path = "Sequence.mp4"
    zone_configuration_path = "scripts\config_sequence.json"
    weights = "yolov8s.pt"  # or path to your weights file
    device = "cuda"  # or 'cuda', 'mps'
    confidence = 0.3
    iou = 0.25
    classes = [0]  # or any array of integers representing class IDs

    main(
        source_video_path=source_video_path,
        zone_configuration_path=zone_configuration_path,
        weights=weights,
        device=device,
        confidence=confidence,
        iou=iou,
        classes=classes,
    )