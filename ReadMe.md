
## Dataset Overview

This dataset contains hotel booking information.
Each row represents one hotel reservation.

### Target Variable
- **is_canceled**
  - 1 → The booking was canceled
  - 0 → The booking was not canceled

### Dataset Purpose
The main goal of this dataset is to **predict whether a hotel booking will be canceled or not**.

By using booking details such as:
- booking time
- number of guests
- stay duration
- customer type

we build a **classification model** that helps hotels:
- reduce revenue loss
- manage rooms better
- improve planning and decision making

## Dataset structure

project-root/

├── .github/
│   └── workflows/          # CI/CD pipelines (GitHub Actions)
│
├── app/                    # Application entry point (API / service)
│
├── data/                   # Datasets (raw)
│
├── models/
│   └── baseline/bestmodel  # Saved baseline models & checkpoints
│
├── notebooks/              # Jupyter notebooks (EDA, baseline_model)
│
├── scripts/                #  calling classes
│
├── src/                    # Classes
│
├── tests/                  # Testing
│
├── .dockerignore           # Exclude files from Docker image
├── .gitignore              # Exclude files from Git tracking
├── Dockerfile              # Docker build configuration
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
