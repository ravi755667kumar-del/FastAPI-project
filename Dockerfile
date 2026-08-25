# 1. Use an official, lightweight Python image
FROM python:3.10-slim

# 2. Set the working directory inside the container
WORKDIR /code

# 3. Copy just the requirements first (to cache the installations and speed up builds)
COPY requirements.txt /code/requirements.txt

# 4. Install the Python libraries
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# 5. Copy your FastAPI app and models into the container
COPY ./app /code/app
COPY ./models /code/models

# 6. Expose the port FastAPI uses
EXPOSE 8000

# 7. Start the FastAPI server using Uvicorn
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]