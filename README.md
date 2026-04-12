

# X Detection System Documentation

This project contains three Python files that work together on the **Jetson Orin Nano** with an **IMX477 camera** and a **Pixhawk 6C**. The system detects an **X-shaped target**, shows it on a connected monitor, gives alignment guidance, and can trigger the motors through Pixhawk when the X is visible.

---

## 1. `X_5.py` — Basic X Detection and Live Display

`X_5.py` is the base script for detecting the X target using a trained YOLO model (`best_22.pt`). Its main purpose is to open the camera, run detection on live frames, and display the results on the monitor connected to the Jetson.

### Main purpose
This file is used to verify that:

- the camera is working
- the YOLO model is loading correctly
- the X target is being detected
- the detection is shown live on screen

### How it works

#### Model loading
The script loads a custom YOLO model from `best_22.pt`. If the file is missing, the program exits.

#### Camera input
It uses a GStreamer pipeline to read video from the Jetson camera. The script supports two modes:

- `4k`
- `1080p`

Although the display window is smaller, the important point is that inference is done on a **full 1920×1080 frame**, because the model was trained on full-resolution images.

#### Detection
For every frame:

1. the frame is captured from the camera
2. YOLO runs inference on the full-resolution frame
3. bounding boxes and labels are generated for detected objects
4. the frame is resized for display
5. the result is shown on screen

#### Display output
The script draws:

- a green bounding box around the detected X
- a label showing class name and confidence
- FPS information
- number of detections

### Modes available

#### Live mode
This is the default mode. It opens a display window and shows the live detection results on the monitor.

#### Headless mode
This mode does not open a video window. Instead, it prints detection results in the terminal such as:

- confidence
- center coordinates
- bounding box coordinates
- FPS

This is useful when running over SSH.

#### Snapshot mode
This mode captures one frame, runs detection once, draws the result, and saves the image to disk.

### What `X_5.py` is used for
`X_5.py` is the **basic vision script**. It answers this question:

> Can the Jetson detect the X and show it on the connected monitor?

The answer is yes. This file is only for detection and display. It does **not** guide centering and does **not** control the motors.

---

## 2. `x_detect_guide.py` — Detection with Centering Guidance

`x_detect_guide.py` builds on the first script. It not only detects the X, but also calculates where the X is located relative to the center of the camera frame. This helps the user understand how to move the camera or drone so the X becomes centered.

### Main purpose
This file is used to:

- detect the X
- find the center of the X
- compare that center with the frame center
- show guidance for alignment

### Does it move the camera?
No. This script does **not** directly move the camera or the drone.

What it does is compute and display movement guidance such as:

- move left
- move right
- move up
- move down
- centered

So this file is used to **tell how to move** so the X comes to the center of the drone camera view.

### How it finds the X center

The script supports two methods:

#### 1. Bounding-box center
This method takes the midpoint of the YOLO bounding box. It is simple and fast.

#### 2. Refined center
This method looks inside the detected bounding box, segments green pixels in HSV color space, and calculates the centroid of the green region. This is meant to estimate the crossing point of the X more accurately.

If the refined method cannot find enough green pixels, it falls back to the bounding-box center.

### Guidance calculation
The script calculates:

- `dx = cx - frame_center_x`
- `dy = cy - frame_center_y`

Where:

- `cx, cy` = center of the detected X
- `frame_center_x, frame_center_y` = center of the camera frame

Interpretation:

- `dx < 0` means the X is left of center
- `dx > 0` means the X is right of center
- `dy < 0` means the X is above center
- `dy > 0` means the X is below center

A configurable **deadzone** is used. If the X center is within the deadzone, the target is treated as centered.

### What is shown on screen
This script displays:

- bounding box around the X
- crosshair at the frame center
- colored dot at the X center
- line connecting the X center to the frame center
- offset values
- guidance text such as `LEFT`, `RIGHT`, `UP`, `DOWN`, or `CENTERED`

### Terminal output
It also prints movement guidance in the terminal, for example:

- searching for X
- move left
- move right
- centered

### Modes available

#### Live mode
Shows the annotated camera feed on the monitor and prints guidance in the terminal.

#### Headless mode
Runs without a display window and only prints guidance text in the terminal.

#### Snapshot mode
Captures one frame, computes the center and guidance, draws the annotations, and saves the result.

### What `x_detect_guide.py` is used for
`x_detect_guide.py` is the **guidance script**. It answers this question:

> How should the drone or camera move so the X is in the center?

So yes, this file is used to tell whether the X is left, right, above, or below the center, and whether alignment is correct.

It does **not** control motors. It only gives visual and terminal guidance.

---

## 3. `x_detect_motor.py` — Detection with Pixhawk Motor Control

`x_detect_motor.py` is the action script. It connects the Jetson vision system to the Pixhawk using `pymavlink`. When the X is detected, it triggers motor activity.

### Main purpose
This file is used to:

- detect the X
- connect Jetson to Pixhawk
- arm the flight controller
- spin the motors
- stop and disarm after a fixed duration

### Safety note
This script includes a strong safety warning:

> REMOVE PROPS FOR TESTING

That warning is critical because this script can actually command the motors.

### How it works

#### YOLO detection
Like the other files, this script captures frames from the camera and runs YOLO inference using `best_22.pt`.

#### Pixhawk connection
The script connects to Pixhawk through a serial port such as:

- `/dev/ttyACM0`

It uses a baud rate such as:

- `115200`

It waits for a MAVLink heartbeat before proceeding.

#### Trigger condition
If at least one X is detected and the motors are not already running, the script treats that as the trigger event.

#### Motor command flow
When triggered, the script:

1. force-arms the Pixhawk
2. sends `MAV_CMD_DO_MOTOR_TEST` to motors 1, 2, 3, and 4
3. runs them at the requested throttle percentage
4. keeps them running for the configured duration
5. force-disarms afterward

### Parameters
The file supports configurable settings such as:

- confidence threshold
- serial device
- baud rate
- throttle percentage
- motor duration

### Dry-run mode
A very useful safety feature is `--dry-run`.

In dry-run mode:

- the camera and detection still run
- the system behaves as though it detected the X
- but it does **not** connect to Pixhawk or spin the motors

This is useful for testing the logic safely.

### Display output
The monitor shows:

- live video
- bounding boxes
- center points of detections
- FPS
- status text

When motors are active, the display also shows:

- red border
- motor active status
- remaining time

### What `x_detect_motor.py` is used for
`x_detect_motor.py` is the **control script**. It answers this question:

> Can Jetson detect the X and tell Pixhawk to start the motors?

Yes. This file is the one that connects visual detection with Pixhawk motor commands.

---

## Overall Flow of the Three Files

These three files represent three stages of the system:

### `X_5.py`
**See the X**

- detects the X
- shows the result on the monitor
- useful for basic model testing

### `x_detect_guide.py`
**Center the X**

- detects the X
- finds the X center
- compares it to frame center
- tells how to move so the X becomes centered

### `x_detect_motor.py`
**React to the X**

- detects the X
- communicates with Pixhawk
- arms and runs motors when the X is visible

---

## Final Summary

This project is structured in a progressive way:

1. `X_5.py` verifies X detection and live display
2. `x_detect_guide.py` adds alignment and centering guidance
3. `x_detect_motor.py` adds Pixhawk motor control based on X visibility

In simple words:

- `X_5.py` = detect and show the X
- `x_detect_guide.py` = guide movement to center the X
- `x_detect_motor.py` = trigger motors when the X is detected

This makes the system suitable for target detection, alignment assistance, and hardware triggering in a Jetson + Pixhawk workflow.
