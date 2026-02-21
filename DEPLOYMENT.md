# Deployment Guide

This guide provides step-by-step instructions for deploying the ClearPath RAG Chatbot to production.

## Architecture

- **Frontend**: Next.js application deployed on Vercel
- **Backend**: FastAPI application deployed on Render
- **Database**: Supabase (PostgreSQL with pgvector)
- **APIs**: Groq API (LLM), Hugging Face Inference API (embeddings)

## Prerequisites

1. GitHub account with your code repository
2. Vercel account (free tier available)
3. Render account (free tier available)
4. Supabase account (free tier available)
5. Groq API key
6. Hugging Face API key

## Part 1: Deploy Backend to Render

### Step 1: Prepare Supabase Database

1. Go to [Supabase](https://supabase.com) and create a new project
2. Wait for the database to be provisioned
3. Go to Project Settings → API to get:
   - `SUPABASE_URL` (Project URL)
   - `SUPABASE_KEY` (anon/public key)
4. Go to SQL Editor and run the migration scripts:
   ```sql
   -- Enable pgvector extension
   CREATE EXTENSION IF NOT EXISTS vector;
   
   -- Run migrations from backend/migrations/
   -- 001_create_chunks_table.sql
   -- 002_create_conversations_tables.sql
   ```

### Step 2: Get API Keys

1. **Groq API Key**:
   - Go to [Groq Console](https://console.groq.com)
   - Create an account and generate an API key
   - Copy the key (starts with `gsk_`)

2. **Hugging Face API Key**:
   - Go to [Hugging Face](https://huggingface.co/settings/tokens)
   - Create a new access token (read permission is sufficient)
   - Copy the token

### Step 3: Deploy to Render

1. Go to [Render Dashboard](https://dashboard.render.com)
2. Click "New +" → "Web Service"
3. Connect your GitHub repository
4. Configure the service:
   - **Name**: `clearpath-rag-backend`
   - **Region**: Oregon (US West)
   - **Branch**: `main`
   - **Root Directory**: Leave empty (render.yaml handles this)
   - **Environment**: Python 3
   - **Build Command**: `pip install -r backend/requirements.txt`
   - **Start Command**: `cd backend && uvicorn main:app --host 0.0.0.0 --port $PORT`
   - **Plan**: Free

5. Add Environment Variables:
   - `GROQ_API_KEY`: Your Groq API key
   - `HUGGINGFACE_API_KEY`: Your Hugging Face token
   - `SUPABASE_URL`: Your Supabase project URL
   - `SUPABASE_KEY`: Your Supabase anon key
   - `PORT`: 8000
   - `LOG_LEVEL`: INFO
   - `CORS_ORIGINS`: `https://your-frontend.vercel.app,https://*.vercel.app` (update after deploying frontend)

6. Click "Create Web Service"
7. Wait for deployment to complete (5-10 minutes)
8. Copy your backend URL (e.g., `https://clearpath-rag-backend.onrender.com`)

### Step 4: Ingest Documents

After the backend is deployed, you need to ingest the PDF documents:

1. SSH into your Render service or run locally:
   ```bash
   cd backend
   python ingest_documents.py
   ```

2. This will:
   - Load all PDFs from `clearpath_docs/`
   - Chunk the documents
   - Generate embeddings
   - Store in Supabase

**Note**: On Render's free tier, you may need to run ingestion locally and point to your production Supabase database.

## Part 2: Deploy Frontend to Vercel

### Step 1: Update API URL

1. Open `frontend/app/page.tsx`
2. Find all instances of `http://localhost:8000`
3. Replace with your Render backend URL:
   ```typescript
   const API_URL = process.env.NEXT_PUBLIC_API_URL || 'https://clearpath-rag-backend.onrender.com'
   
   // Then use it in fetch calls:
   fetch(`${API_URL}/query`, ...)
   fetch(`${API_URL}/query/stream`, ...)
   ```

### Step 2: Deploy to Vercel

1. Go to [Vercel Dashboard](https://vercel.com/dashboard)
2. Click "Add New..." → "Project"
3. Import your GitHub repository
4. Configure the project:
   - **Framework Preset**: Next.js
   - **Root Directory**: `frontend`
   - **Build Command**: `npm run build`
   - **Output Directory**: `.next`
   - **Install Command**: `npm install`

5. Add Environment Variable:
   - `NEXT_PUBLIC_API_URL`: Your Render backend URL

6. Click "Deploy"
7. Wait for deployment to complete (2-3 minutes)
8. Your frontend will be live at `https://your-project.vercel.app`

### Step 3: Update CORS Settings

1. After deploying frontend to Vercel, copy your Vercel URL
2. Go to Render Dashboard → Your Service → Environment
3. Update the `CORS_ORIGINS` environment variable:
   ```
   https://your-project.vercel.app,https://*.vercel.app
   ```
4. Save changes - Render will automatically redeploy

## Part 3: Verify Deployment

### Test Backend

1. Visit `https://your-backend.onrender.com/health`
2. Should return: `{"status": "healthy", "service": "clearpath-rag-chatbot", "version": "1.0.0"}`

3. Test query endpoint:
   ```bash
   curl -X POST https://your-backend.onrender.com/query \
     -H "Content-Type: application/json" \
     -d '{"question": "What is ClearPath?"}'
   ```

### Test Frontend

1. Visit your Vercel URL
2. Try asking a question
3. Verify streaming works
4. Check debug panel shows metadata

## Troubleshooting

### Backend Issues

**Problem**: Service won't start
- Check Render logs for errors
- Verify all environment variables are set
- Ensure Python version is 3.10+

**Problem**: Database connection fails
- Verify Supabase URL and key are correct
- Check Supabase project is active
- Ensure pgvector extension is enabled

**Problem**: API rate limits
- Groq free tier: 30 requests/minute
- Hugging Face free tier: Rate limited, may need to upgrade

### Frontend Issues

**Problem**: Can't connect to backend
- Verify API_URL environment variable
- Check CORS settings in backend
- Ensure backend is deployed and healthy

**Problem**: Streaming doesn't work
- Check browser console for errors
- Verify `/query/stream` endpoint is accessible
- Test with regular mode first

### Performance Issues

**Render Free Tier Limitations**:
- Service spins down after 15 minutes of inactivity
- First request after spin-down takes 30-60 seconds
- 512 MB RAM limit
- Consider upgrading to paid tier for production

**Vercel Free Tier Limitations**:
- 100 GB bandwidth/month
- Serverless function timeout: 10 seconds
- Should be sufficient for frontend

## Monitoring

### Backend Monitoring

1. Render Dashboard → Your Service → Logs
2. Monitor for errors and performance issues
3. Check `/health` endpoint regularly

### Frontend Monitoring

1. Vercel Dashboard → Your Project → Analytics
2. Monitor page views and performance
3. Check for deployment errors

## Cost Estimation

### Free Tier Usage

- **Render**: Free (with limitations)
- **Vercel**: Free (with limitations)
- **Supabase**: Free up to 500 MB database, 2 GB bandwidth
- **Groq API**: Free tier with rate limits
- **Hugging Face**: Free tier with rate limits

### Paid Tier (Recommended for Production)

- **Render**: $7/month (Starter plan)
- **Vercel**: $20/month (Pro plan)
- **Supabase**: $25/month (Pro plan)
- **Groq API**: Pay-as-you-go
- **Hugging Face**: $9/month (PRO plan)

**Total**: ~$61/month for production-ready deployment

## Security Considerations

1. **Never commit API keys** to git
2. **Use environment variables** for all secrets
3. **Enable HTTPS** (automatic on Vercel/Render)
4. **Implement rate limiting** for production
5. **Monitor API usage** to prevent abuse
6. **Rotate API keys** regularly

## Scaling Considerations

### When to Scale

- Backend response time > 2 seconds
- Render service frequently spins down
- Database queries slow (> 500ms)
- API rate limits hit frequently

### Scaling Options

1. **Upgrade Render plan** for always-on service
2. **Add Redis caching** for embeddings
3. **Use dedicated vector database** (Pinecone, Weaviate)
4. **Implement CDN** for static assets
5. **Add load balancer** for multiple backend instances

## Maintenance

### Regular Tasks

1. **Monitor logs** weekly
2. **Check API usage** monthly
3. **Update dependencies** quarterly
4. **Rotate API keys** every 6 months
5. **Backup database** monthly

### Updates

1. Push changes to GitHub
2. Vercel auto-deploys frontend
3. Render auto-deploys backend
4. Test in production
5. Monitor for issues

## Support

For issues:
1. Check logs in Render/Vercel dashboards
2. Review this deployment guide
3. Check API provider status pages
4. Contact support if needed

## Next Steps

After successful deployment:
1. Set up monitoring and alerts
2. Configure custom domain (optional)
3. Implement analytics
4. Add user authentication (if needed)
5. Set up CI/CD pipeline
6. Create backup strategy
