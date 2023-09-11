# cat-vs-dog-classifier-demo

A cat vs dog classifier demo API that uses FastAPI for the python backend and Pytorch for the machine learning.

It accepts a random number 1-1000 for a random image in the testing data, or a custom image input encoded as a base64 string.

It returns the classification and the base64 data for the image.

The current model has a ~95% test accuracy of correctly predicting an image as a cat or a dog.

Pip installs needed: pip install fastapi "uvicorn[standard]" torch torchvision Pillow

Run command: uvicorn main:app --reload

To test the API, use http://127.0.0.1:8000/docs 