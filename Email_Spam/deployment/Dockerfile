FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080

WORKDIR /app

COPY Email_Spam/deployment/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /tmp/requirements.txt

COPY Email_Spam/__init__.py /app/Email_Spam/__init__.py
COPY Email_Spam/email_spam_pipeline.py /app/Email_Spam/email_spam_pipeline.py
COPY Email_Spam/Email_Spam_Final_Pipeline.pkl /app/Email_Spam/Email_Spam_Final_Pipeline.pkl
COPY Email_Spam/deployment /app/Email_Spam/deployment

RUN useradd --create-home --uid 10001 apiuser \
    && chown -R apiuser:apiuser /app
USER apiuser

EXPOSE 8080

CMD ["sh", "-c", "uvicorn Email_Spam.deployment.app:app --host 0.0.0.0 --port ${PORT:-8080}"]
