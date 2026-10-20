# Email Spam API Deployment

The service accepts raw email text and applies the saved text cleaner, TF-IDF representation, classifier, and validation-selected decision threshold. Retraining is not required to start the API.

## Endpoints

- `GET /health` — readiness and model-loading status.
- `GET /metadata` — model configuration and final-test metrics.
- `POST /predict` — classify one raw email.
- `GET /docs` — interactive OpenAPI documentation.

Example payload:

```json
{
  "text": "Free money! Click here now to claim your prize."
}
```

The response contains the predicted class, a continuous spam score, the fixed decision threshold, and the selected model name.

## Local Run

Run commands from the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r .\Email_Spam\deployment\requirements.txt
uvicorn Email_Spam.deployment.app:app --host 0.0.0.0 --port 8080
```

Smoke test:

```powershell
Invoke-RestMethod http://localhost:8080/health
Invoke-RestMethod -Method Post `
  -Uri http://localhost:8080/predict `
  -ContentType "application/json" `
  -Body '{"text":"Free money! Click here now to claim your prize."}'
```

## Docker

Build from the repository root because the serialized pipeline references the `Email_Spam` Python package:

```powershell
docker build -f .\Email_Spam\deployment\Dockerfile -t email-spam-api .
docker run --rm -p 8080:8080 email-spam-api
```

## Google Cloud Run

Authenticate, select a Google Cloud project, enable the required APIs, and create an Artifact Registry repository once:

```powershell
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com
gcloud artifacts repositories create ml-models `
  --repository-format=docker `
  --location=europe-central2
```

Build and push the image:

```powershell
gcloud builds submit --config .\Email_Spam\deployment\cloudbuild.yaml .
```

Deploy the service:

```powershell
gcloud run deploy email-spam-api `
  --image europe-central2-docker.pkg.dev/YOUR_PROJECT_ID/ml-models/email-spam-api:latest `
  --region europe-central2 `
  --platform managed `
  --allow-unauthenticated `
  --memory 1Gi `
  --cpu 1 `
  --min-instances 0 `
  --max-instances 3
```

Remove `--allow-unauthenticated` and configure IAM for a private API. Cloud Run supplies the `PORT` environment variable automatically.

## Tests

```powershell
pip install -r .\Email_Spam\deployment\requirements-test.txt
pytest .\Email_Spam\deployment\tests -q
```

