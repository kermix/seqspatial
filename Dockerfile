FROM python:3.9-slim-buster

WORKDIR /app

RUN apt-get update && apt-get install -y \
  gcc \
  && rm -rf /var/lib/apt/lists/*


# By copying over requirements first, we make sure that Docker will cache
# our installed requirements rather than reinstall them on every build
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
# require numpy installed before
RUN pip install --no-cache-dir scikit-bio


# Now copy in our code, and run it
COPY ./app /app

RUN echo '#!/bin/bash\npython /app/app/data/analyze.py "$@"' > /usr/bin/analyze && \
    chmod +x /usr/bin/analyze

RUN echo '#!/bin/bash\npython /app/app/data/create_dataset.py "$@"' > /usr/bin/create_dataset && \
    chmod +x /usr/bin/create_dataset

CMD [ "python", "/app/app/gui/index.py" ]
