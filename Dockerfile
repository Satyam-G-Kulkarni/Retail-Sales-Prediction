# Use a specific Python version as the base image
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . .

# Install dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port for the Streamlit app
EXPOSE 8501

# Run the Streamlit app
CMD ["streamlit", "run", "Sales_prediction_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
