\# UPI Payment Screenshot Fraud Detection System



\## Overview



The UPI Payment Screenshot Fraud Detection System is a web-based application that analyzes uploaded payment screenshots and identifies suspicious or potentially fraudulent payment evidence.



The system combines OCR-based information extraction, image analysis, consistency checks, image forensics, and a trained machine learning model to generate a risk assessment for the uploaded screenshot.



\## Features



\- User login and authentication

\- Payment screenshot upload

\- OCR-based extraction of payment information

\- Detection of payment details such as:

&#x20; - UPI application

&#x20; - Amount

&#x20; - Transaction ID

&#x20; - UTR

&#x20; - Merchant

&#x20; - Payment status

\- Image forensic analysis

\- Error Level Analysis (ELA)

\- Payment information consistency checking

\- Machine learning based image analysis

\- Risk level and suspicion score

\- Analysis result and explanation

\- Dashboard for viewing analysis

\- Analysis history during the current application session

\- Profile and logout functionality



\## Technology Stack



\### Backend

\- Python

\- Flask

\- REST-style Flask routes



\### Machine Learning \& Image Processing

\- TensorFlow

\- Keras

\- OpenCV

\- NumPy

\- Pillow

\- Tesseract OCR

\- Pytesseract



\### Frontend

\- HTML5

\- CSS3

\- Jinja2 Templates



\### Model

The project uses a trained Keras model stored in:



`models/image\_fraud\_model.keras`



\## How It Works



1\. The user logs into the application.

2\. The user uploads a UPI/payment screenshot.

3\. The application validates the uploaded image.

4\. OCR extracts available payment-related information.

5\. Image analysis and forensic checks are performed.

6\. The extracted information is checked for consistency.

7\. The trained machine learning model analyzes the image.

8\. The application calculates risk-related scores.

9\. The system displays the analysis result and explanation.



\## Project Structure



```text

upi-fraud-detection/

│

├── app.py

├── fraud\_detector.py

├── behavior\_analyzer.py

├── train\_model.py

├── requirements.txt

├── .gitignore

│

├── models/

│   └── image\_fraud\_model.keras

│

├── templates/

│   ├── dashboard.html

│   ├── history.html

│   ├── index.html

│   ├── login.html

│   ├── profile.html

│   ├── result.html

│   └── upload.html

│

└── static/

&#x20;   └── style.css

```



\## Installation



\### 1. Clone the repository



```bash

git clone https://github.com/chandrikachandu0113-source/upi-fraud-detection.git

cd upi-fraud-detection

```



\### 2. Create a Python virtual environment



Python 3.12 is recommended for this project.



```bash

python -m venv venv

```



\### 3. Activate the environment



On Windows:



```powershell

venv\\Scripts\\activate

```



\### 4. Install dependencies



```bash

pip install -r requirements.txt

```



\### 5. Configure environment variables



The application uses environment variables for the login credentials and Flask secret key.



Windows PowerShell:



```powershell

$env:ADMIN\_EMAIL="admin@frauddetect.com"

$env:ADMIN\_PASSWORD="admin123"

$env:SECRET\_KEY="change-this-secret-key"

```



For production deployment, use secure values instead of the example values above.



\### 6. Run the application



```bash

python app.py

```



The application will be available locally at:



```text

http://127.0.0.1:5000

```



\## Important Notes



\- The system analyzes payment screenshots; it does not directly access or verify transactions with UPI banking systems.

\- Results should be treated as an automated risk assessment and not as a definitive financial or banking decision.

\- Analysis history is currently maintained in application memory and is cleared when the Flask application restarts.

\- Tesseract OCR must be installed separately on systems where OCR is used.



\## Future Improvements



\- Deploy the application as a production web service

\- Add persistent database storage for analysis history

\- Improve fraud detection model accuracy with a larger and more diverse dataset

\- Add additional payment application formats

\- Add API-based integration for external verification

\- Improve authentication and user management

\- Add automated model monitoring and evaluation



\## Author



\*\*Chandrika N\*\*



B.E./B.Tech – Computer Science and Engineering



GitHub: `chandrikachandu0113-source`



LinkedIn: `chandrika1513`

