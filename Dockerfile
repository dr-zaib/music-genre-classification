FROM public.ecr.aws/lambda/python:3.11

#RUN pip install https://github.com/alexeygrigorev/tflite-aws-lambda/blob/main/tflite/tflite_runtime-2.14.0-cp311-cp311-linux_x86_64.whl?raw=true
RUN pip install --upgrade tensorflow
RUN pip install numpy

COPY mugen-model.tflite .
COPY genres_lambda_lite.py .

CMD [ "genres_lambda_lite.lambda_handler" ]