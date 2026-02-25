# spaCy Setup for Unverified Feature Detection

## Installation

The `unverified_feature` flag now uses spaCy for accurate named entity recognition (NER) to detect proper nouns like company names, products, and integrations.

### Local Development

#### Step 1: Install spaCy

```bash
pip install spacy
```

#### Step 2: Download English Language Model

```bash
python -m spacy download en_core_web_sm
```

#### Alternative: Install from requirements.txt

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### Render Deployment

The spaCy model is automatically downloaded during the build process via the `buildCommand` in `render.yaml`:

```yaml
buildCommand: "pip install -r backend/requirements.txt && python -m spacy download en_core_web_sm"
```

No additional configuration needed - the model will be downloaded and cached during deployment.

### Other Cloud Platforms

#### AWS EC2
Add to your deployment steps (after `pip install -r requirements.txt`):
```bash
python -m spacy download en_core_web_sm
```

See `AWS_EC2_DEPLOYMENT.md` for complete deployment guide.

#### Heroku
Add to `Procfile`:
```
release: python -m spacy download en_core_web_sm
```

#### AWS Elastic Beanstalk
Add to `.ebextensions/01_spacy.config`:
```yaml
container_commands:
  01_download_spacy_model:
    command: "python -m spacy download en_core_web_sm"
```

#### AWS Lambda
Include the model in your deployment package or Lambda Layer:
```bash
# Create layer
mkdir python
pip install spacy -t python/
python -m spacy download en_core_web_sm -d python/
zip -r spacy-layer.zip python/
```

#### Docker
Add to `Dockerfile`:
```dockerfile
RUN pip install -r requirements.txt
RUN python -m spacy download en_core_web_sm
```

#### Railway
Add to build command in Railway dashboard:
```bash
pip install -r requirements.txt && python -m spacy download en_core_web_sm
```

## Verification

Test that spaCy is working correctly:

```bash
python backend/test_spacy_extraction.py
```

Expected output:
- ✓ Should extract: 'slack', 'clearpath' (actual proper nouns)
- ✗ Should NOT extract: 'set', 'go', 'click', 'integrations' (common words)
- ✓ Should NOT flag unverified_feature when proper nouns are in chunks

## Fallback Behavior

If spaCy is not installed, the system will:
1. Fall back to simple capitalized word extraction
2. Only extract words with CamelCase or ALL CAPS (more conservative)
3. Still use integration patterns for known tools

This ensures the system works even without spaCy, but with reduced accuracy.

## Benefits of spaCy

1. **Accurate NER**: Distinguishes between proper nouns and common words
2. **No manual stop words**: Automatically filters out "Set", "Go", "Click", etc.
3. **Context-aware**: Understands "Slack" is an organization, "Set" is a verb
4. **Reduces false positives**: Only flags actual product/company names

## Entity Types Detected

- **ORG**: Organizations, companies (e.g., "Slack", "GitHub", "Microsoft")
- **PRODUCT**: Products, services (e.g., "ClearPath", "Office 365")
- **GPE**: Geopolitical entities (e.g., "Paris", "France") - useful for detecting out-of-domain queries

## Performance

- Model size: ~12 MB (en_core_web_sm)
- Load time: ~100ms (cached after first load)
- Processing time: ~5-10ms per response
- Minimal impact on overall latency
