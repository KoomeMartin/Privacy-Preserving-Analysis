### **Summary of EDA Implementations for Adaptive Privacy Redaction**

* **Multi-Class ViPER-XML Parsing Engine**: Developed a specialized tool to extract temporal and spatial data for **Persons**, **Faces**, and **Accessories** from the PEViD-HD dataset.
* **Depth-Based Risk Zoning**: Implemented a mathematical framework using bounding box area as a proxy for physical distance to the sensor.
    * **Low Risk**: Area less than 20,000 pixels. Triggers light Gaussian Blur.
    * **Medium Risk**: Area between 20,000 and 80,000 pixels. Triggers Pixelation.
    * **High Risk**: Area greater than 80,000 pixels. Triggers full Blackout Silhouette Masking.
* **Temporal Stability & Smoothing**: Integrated a **5-frame rolling mean** to stabilize raw annotation jitter. This prevents "redaction flickering" by ensuring the filter logic follows a smooth signal rather than noisy raw data.
* **Area Change Velocity Analysis**: Analyzed the rate of area change per frame to identify "Artifact Spikes"—unrealistic jumps in annotation size (e.g., frame 360) that require smoothing to maintain visual consistency.
* **Privacy Risk Score (PRS) Calculation**: Created a normalized scoring system between 0.0 and 1.0 based on smoothed proximity data to dictate redaction strength dynamically.
    * **The PRS Formula**: The PRS is the minimum of 1.0 or the Smoothed Area divided by 80,000.
* **Utility-Privacy Conflict Detection (IoU)**: Utilized **Intersection over Union (IoU)** to quantify spatial overlaps between Faces (Privacy) and Accessories (Utility).

* **Semantic Silhouette Justification**: Demonstrated that standard box-based redaction risks hiding critical contextual items, justifying our silhouette masking approach to preserve utility.
* **Global Dataset Risk Profiling**: Generated a comprehensive distribution table across all videos to provide the quantitative "Dataset Profile" for the project manuscript.
* **Dynamic Analytics Video Engine**: Built a real-time auditing tool that overlays smoothed Area, active Redaction Mode, and PRS directly onto video frames for visual verification.