FROM public.ecr.aws/lambda/python:3.8

RUN pip install keras-image-helper

RUN  pip install tflite_runtime

COPY clothing-model.tflite .

COPY lambda_function.py .

CMD["lambda_function.lambda_handler"]
