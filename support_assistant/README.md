# Zepto Support Assistant

## Run with Docker

Build the image from the project root:

docker build -t zepto-support-assistant -f support_assistant/Dockerfile .

Run the API:

docker run --rm -p 8001:8000 zepto-support-assistant

Open the FastAPI documentation at:

http://localhost:8001/docs

The /ask endpoint accepts a JSON request containing a policy question and returns the retrieved answer, source documents, and confidence score.
