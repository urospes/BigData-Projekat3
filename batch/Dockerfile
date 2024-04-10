FROM bitnami/spark:latest

# Copy the source code
COPY . /app
RUN cd /app && pip install numpy

ENV SPARK_MASTER_NAME spark
ENV SPARK_MASTER_PORT 7077
ENV SPARK_APPLICATION_PYTHON_LOCATION /app/batch.py

COPY submit.sh /
CMD ["/bin/bash", "/submit.sh"]