# Data Science Project Boilerplate

This boilerplate is designed to kickstart data science projects by providing a basic setup for database connections, data processing, and machine learning model development. It includes a structured folder organization for your datasets and a set of pre-defined Python packages necessary for most data science tasks.

## Structure

The project is organized as follows:

- `requirements.txt` - This file contains the list of necessary python packages for deployment.
- `requirements_dev.txt` - This file contains the list of necessary python packages for studying the data and models.
- `noteboks` - Notebooks to do the exploration of the data and the model.
- `src` - This folder contains the backend code.
- `models/` - This directory should contain your SQLAlchemy model classes.
- `data/` - This directory contains the following subdirectories:
  - `interin/` - For intermediate data that has been transformed.
  - `processed/` - For the final data to be used for modeling.
  - `raw/` - For raw data without any processing.
 
    
## Setup

**Prerequisites**

Make sure to have Python 3.11+ installed.

**Installation**

Clone the project repository to your local machine.

Navigate to the project directory and install the required Python packages:

```bash
pip install -r requirements_dev.txt
```

**Environment Variables**

Create a .env file in the project root directory to store your environment variables.

```makefile
KAGGLEHUB_CACHE=../.
CORS_URL=cors_url
```

## Running the Application

There is a FastAPI backend, to run it use the command:

```bash
fastapi run .\src\main.py
```

## Data example to test the project (should return rice)

```json
{
  "N": 90,
  "P": 42,
  "K": 43,
  "temperature": 20.87974371,
  "humidity": 82.00274423,
  "ph": 6.502985292000001,
  "rainfall": 202.9355362
}
```
