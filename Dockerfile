FROM python:3.10

WORKDIR /app

RUN apt update && apt install -y libgl1

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["python", "backend.py"]