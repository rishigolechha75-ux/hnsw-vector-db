FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY vecdb ./vecdb
ENV VECDB_DATA=/data NUMBA_CACHE_DIR=/tmp/numba
VOLUME /data
EXPOSE 8000
RUN useradd -m vecdb && mkdir -p /data && chown vecdb /data
USER vecdb
CMD ["python", "-m", "vecdb", "serve", "--data", "/data", "--host", "0.0.0.0", "--port", "8000"]
