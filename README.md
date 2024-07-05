# License Plate Recognition

Dự án này nhằm mục đích nhận dạng biển số xe bằng các mô hình YOLOv8 và OpenCV. Bạn có thể chạy dự án này cục bộ hoặc sử dụng Docker.

## Prerequisites

- Python 3.6 or above
- Docker (optional, for running the project using Docker)

## Installation

### Running Locally

1. **Clone the repository:**
   ```bash
   git clone <repository_url>
   cd <repository_directory>
2. **Install the required libraries:**
   ```bash
   pip install -r requirements.txt
3. **To start the application:**
   ```bash
   streamlit run app.py

### Running With Docker
1. **Pull the Docker image from Docker Hub:**
   ```bash
   docker pull nmduc24/license-plate-recognition:latest
2. **Run the Docker container:**
   ```bash
   docker run -p 8501:8501 nmduc24/license-plate-recognition:latest
### Accessing the Application
Khi ứng dụng đang chạy, hãy mở trình duyệt web của bạn và truy cập http://localhost:8501/ để truy cập giao diện người dùng.


