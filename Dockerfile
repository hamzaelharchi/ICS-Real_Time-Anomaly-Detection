# Use specific version instead of latest
FROM jupyter/pyspark-notebook:python-3.11

# Install system utilities
USER root
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
    iputils-ping \
    telnet && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    fix-permissions "/home/${NB_USER}"

# Switch to notebook user
USER ${NB_UID}

# Copy requirements file
COPY requirements.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.txt


#RUN spark-submit   --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1   --jars kafka-clients-3.5.0.jar   --driver-class-path kafka-clients-3.5.0.jar   --conf "spark.executor.extraClassPath=kafka-clients-3.5.0.jar" work/keep_reading.py

#docker exec -it -u root pyspark bash
