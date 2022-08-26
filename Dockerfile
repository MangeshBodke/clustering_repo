FROM python:3.9.2
ADD app.py .
COPY . /app
WORKDIR /app

RUN pip install -r requirements.txt
ENV PORT 8080
ENV HOST 0.0.0.0
EXPOSE 8080
CMD [ "python", "./app.py" ]
