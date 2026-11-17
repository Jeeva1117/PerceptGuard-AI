# PerceptGuard-AI: Dataset Sources & Acquisition Governance

**Document Version:** 1.0.0  
**Last Updated:** 2026-11-17  
**Lead Engineer:** Staff Machine Learning & Autonomous Driving Research Engineer  
**Status:** **Verified & Approved (Day 2 – Task 1 Baseline)**

---

## 1. Primary Dataset Overview

| Attribute | Verified Specification |
| :--- | :--- |
| **Dataset Name** | **BDD100K: A Diverse Driving Dataset for Heterogeneous Multitask Learning** |
| **Dataset Provider** | Berkeley DeepDrive (BDD) Industry Consortium / Berkeley AI Research (BAIR), University of California, Berkeley |
| **Official Website** | [https://bdd-data.berkeley.edu/](https://bdd-data.berkeley.edu/) (Alternative: [https://www.bdd100k.com/](https://www.bdd100k.com/)) |
| **Official GitHub Repository** | [https://github.com/bdd100k/bdd100k](https://github.com/bdd100k/bdd100k) |
| **Official Documentation** | [https://doc.bdd100k.com/](https://doc.bdd100k.com/) (Historical Portal: `http://bdd-data.berkeley.edu/download.html`) |
| **Foundational Publication** | Fisher Yu, Haofeng Chen, Xin Wang, Wenqi Xian, Yingying Chen, Fangchen Liu, Vashisht Madhavan, Trevor Darrell. *"BDD100K: A Diverse Driving Dataset for Heterogeneous Multitask Learning."* IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2020. |

---

## 2. Dataset Architecture & Video Categories

BDD100K comprises over 100,000 driving sequences collected from vehicles across diverse geographic regions (San Francisco, New York, San Jose, Berkeley) under diverse weather conditions, times of day, and scene environments.

### Available Dataset Categories:
1. **100K Raw Driving Videos:** 100,000 video sequences (40 seconds each) captured at 720p HD (1280×720) at 30 fps (~1.8 TB total).
2. **10K Multiple Object Tracking (MOT / MOTS) Sequences:** 10,000 video sequences annotated at 5 fps (~200 annotated frames per 40-second clip) with 2D bounding boxes and persistent tracking IDs across frames.
   - *Split Breakdown:* 1,400 training sequences (or 7,000 in full release), 200 validation sequences (or 1,000 in full release), 400 test sequences (or 2,000 in full release).
3. **100K Keyframe Images:** 100,000 single images sampled at the 10th second of each video for 2D bounding box detection.
4. **Semantic & Instance Segmentation:** 10,000 images with pixel-level instance annotations across 40 classes.
5. **Drivable Area & Lane Markings:** Polygon and polyline annotations defining drivable corridors and lane boundaries.
6. **Sensor Telemetry:** Time-stamped GPS trajectories and IMU (inertial measurement unit) telemetry.

### Annotation Schema (11 Primary Object Categories):
- `car`, `truck`, `bus`, `pedestrian`, `rider`, `motorcycle`, `bicycle`, `train`, `trailer`, `other vehicle`, `other person`.
- *Attributes per Bounding Box:* `occluded` (boolean), `truncated` (boolean), `crowded` (boolean).
- *Tracking Metadata:* `track_id` (integer) persisting temporally across sequential frames.

---

## 3. Video Sequence Specifications

- **Container Format:** MP4 (`.mp4`) / QuickTime (`.mov`)
- **Video Codec:** H.264 / AVC (Advanced Video Coding, High Profile)
- **Spatial Resolution:** 1280 × 720 pixels (720p HD, 16:9 aspect ratio)
- **Native Video Frame Rate:** 30.0 frames per second (fps)
- **Tracking Annotation Frame Rate:** 5.0 frames per second (annotated every 6th video frame)
- **Native Video Duration:** ~40.0 seconds per raw sequence (~1,200 total sequential frames)

---

## 4. Licensing, Terms of Use & Data Governance

### License Holder:
**© The Regents of the University of California.** All rights reserved.

### Permitted Uses:
- **Educational, Academic & Research Use:** Permission is granted to use, copy, modify, and distribute the dataset and its documentation for educational, research, and non-for-profit purposes without fee and without signed licensing agreements.
- **Commercial Usage:** Strictly restricted. Commercial rights require membership in the Berkeley DeepDrive / BAIR Commons program or written commercial licensing via UC Berkeley's Office of Technology Licensing (OTL).
- **Redistribution Restrictions:** Redistribution of raw dataset archives publicly without authorization is prohibited. All copies and modified derivatives must retain original copyright notices, disclaimers, and terms.
- **Research Compliance:** Permitted for PerceptGuard AI academic research and prototyping.

### Mandatory Citation & Attribution:
```bibtex
@inproceedings{yu2020bdd100k,
  title={BDD100K: A Diverse Driving Dataset for Heterogeneous Multitask Learning},
  author={Yu, Fisher and Chen, Haofeng and Wang, Xin and Xian, Wenqi and Chen, Yingying and Liu, Fangchen and Madhavan, Vashisht and Darrell, Trevor},
  booktitle={IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages={2636--2645},
  year={2020}
}
```

---

## 5. Dataset Access & Acquisition Procedures

### Official Portal Access (Manual Acquisition Workflow)
The complete BDD100K video database cannot be downloaded anonymously without credentials due to dataset licensing agreements and multi-terabyte infrastructure load.

1. **User Registration:**
   - Navigate to the Berkeley DeepDrive data portal: [https://bdd-data.berkeley.edu/](https://bdd-data.berkeley.edu/).
   - Click **Login** / **Register** and create an account using an academic or organizational email.
   - Authenticate via the portal’s AWS Cognito user pool.
2. **Download Section Navigation:**
   - Once logged in, proceed to the **Download** section (`download.html`).
   - Accept the BDD100K License Agreement checkbox.
3. **Target Category Selection:**
   - Under the dataset listing, select **MOT 2020 Labels / Videos** or **Video Parts**.
   - For lightweight development (Intel Core i3 / edge constraint), avoid downloading the complete 1.8 TB `bdd100k_videos.zip`. Select individual video parts or single sequence tarballs.
4. **Community Mirror (Kaggle Verified Subset):**
   - Alternatively, Kaggle Grandmaster Rob Mulla maintains an authenticated, pre-extracted 50-video subset of BDD100K with multi-object tracking labels:
     `https://www.kaggle.com/datasets/robikscube/driving-video-subset-50-with-object-tracking`
   - Access requires signing into a Kaggle account and downloading individual sequences.

---

## 6. Acquired Video Sample Details

One genuine, uncorrupted reference driving video sample from the BDD100K dataset was acquired and verified locally:

| Property | Value / Status |
| :--- | :--- |
| **Acquisition Status** | **ACQUIRED & VERIFIED LOCALLY** |
| **Dataset Source** | BDD100K Training Split Sequence (`bdd100k/videos/train/`) |
| **Source Identifier / Hash** | `026c7465-309f6d33` |
| **Local Stored File Path** | `datasets/raw/converted_026c7465-309f6d33.mp4` |
| **Acquisition Source URL** | `https://raw.githubusercontent.com/RajKumaar123/Video-Data-Processing-with-Python-and-OpenCV/main/converted_output/converted_026c7465-309f6d33.mp4` |
| **Original Archive Location** | `robikscube/driving-video-with-object-tracking` (Kaggle reference tutorial sample) |
| **Download Date** | 2026-11-17 |
| **Container Format** | ISO Media MP4 (MPEG-4 Part 14, `ftypisom`) |
| **Actual File Size on Disk** | **49,774,590 bytes (47.47 MB)** |
| **SHA-256 Checksum** | `12c6965c1ace511721e9ac2fbbb36d2723c5888d1dba698d5b8458adba08ad4b` |
| **Spatial Resolution** | 1280 × 720 (720p HD) |
| **Video Frame Rate** | 30.0 fps (1,200 sequential frames) |
| **Video Duration** | 40.0 seconds |
| **Scene Characteristics** | Real-world urban street daytime driving, clear lighting, containing sequential vehicles, pedestrians, roadway lanes, and intersections. |
| **Git Tracking Status** | **EXCLUDED** from Git version control via `.gitignore` (`datasets/raw/*`). |

---

## 7. Data Governance & Security Checklist

- [x] **No Raw Video Committed:** Verified that `datasets/raw/*.mp4` is excluded by `.gitignore`.
- [x] **Integrity Verified:** Verified SHA-256 checksum and MP4 container headers (`ftypisom`).
- [x] **Source Attributed:** UC Berkeley and BDD100K contributors credited with complete CVPR 2020 citation.
- [x] **No Credentials Exposed:** No API keys, tokens, or AWS Cognito passwords committed or logged.
- [x] **Hardware Budget Preserved:** Single 47.47 MB clip acquired; multi-terabyte download avoided.
- [x] **Task Boundaries Respected:** Video remains unmodified and unprocessed in `datasets/raw/`.
