FROM python:3.9-slim

WORKDIR /app

# Cap the thread pools to the challenge's 8-CPU allocation. Without this, a
# --cpus=8 quota on a many-core eval node still exposes all cores, so numba and
# BLAS spawn one thread per physical core and thrash against the CPU quota.
ENV NUMBA_NUM_THREADS=8 \
    OMP_NUM_THREADS=8 \
    OPENBLAS_NUM_THREADS=8 \
    MKL_NUM_THREADS=8 \
    PYTHONUNBUFFERED=1

# All dependencies ship as manylinux wheels — no compiler needed.
# (Adding build-essential is what blew the disk budget on an earlier build.)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN chmod +x /app/search.py
