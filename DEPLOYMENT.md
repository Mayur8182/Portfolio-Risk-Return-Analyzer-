# 🚀 Portfolio Analytics Platform - Render Deployment Guide

## 📋 Prerequisites

1. **GitHub Repository**: Push your code to GitHub
2. **Render Account**: Sign up at [render.com](https://render.com)
3. **MongoDB Atlas**: Set up a free cluster at [mongodb.com](https://mongodb.com)
4. **API Keys**: Ensure you have valid API keys for financial data

## 🔧 Environment Variables Setup

### Required Environment Variables in Render Dashboard:

```bash
# Core Settings
PORT=5000
FLASK_ENV=production
FLASK_DEBUG=false
PYTHON_VERSION=3.11.5

# Financial Data APIs
FINNHUB_API_KEY=d1hodf1r01qsvr2be4r0d1hodf1r01qsvr2be4rg
TWELVEDATA_API_KEY=b57491954ba64f1d9c851fb4c157bf51

# Database (MongoDB Atlas)
MONGODB_URI=mongodb+srv://username:password@cluster.mongodb.net/portfolio_analytics?retryWrites=true&w=majority

# Security
SECRET_KEY=your_generated_secret_key_here

# Performance Settings
WORKERS=4
WORKER_CONNECTIONS=1000
TIMEOUT=120

# Feature Flags
ENABLE_ENHANCED_ANALYTICS=true
ENABLE_REAL_TIME_DATA=true
ENABLE_PDF_EXPORT=true
```

## 📦 Deployment Steps

### Method 1: Using Render Dashboard

1. **Connect Repository**:
   - Go to Render Dashboard
   - Click "New +" → "Web Service"
   - Connect your GitHub repository

2. **Configure Service**:
   - **Name**: `portfolio-analytics-platform`
   - **Environment**: `Python`
   - **Region**: `Oregon` (or closest to your users)
   - **Branch**: `main`
   - **Build Command**: `./render-build.sh`
   - **Start Command**: `./render-start.sh`

3. **Set Environment Variables**:
   - Add all required environment variables from above
   - Generate a secure SECRET_KEY
   - Add your MongoDB Atlas connection string

4. **Deploy**:
   - Click "Create Web Service"
   - Wait for build and deployment to complete

### Method 2: Using render.yaml (Infrastructure as Code)

1. **Push render.yaml**: Already included in your repository
2. **Connect Repository**: Render will auto-detect the configuration
3. **Set Sensitive Variables**: Add MongoDB URI and other secrets in dashboard

## 🗄️ Database Setup (MongoDB Atlas)

1. **Create Cluster**:
   - Sign up at MongoDB Atlas
   - Create a free M0 cluster
   - Choose a region close to your Render deployment

2. **Configure Access**:
   - Add IP address `0.0.0.0/0` (allow all) for Render
   - Create database user with read/write permissions

3. **Get Connection String**:
   - Click "Connect" → "Connect your application"
   - Copy the connection string
   - Replace `<password>` with your database user password

## 🔍 Health Check

Your deployment includes a health check endpoint:
- **URL**: `https://your-app.onrender.com/health`
- **Response**: JSON with service status

## 📊 Performance Optimization

### Render Plan Recommendations:

- **Starter Plan**: Good for development/testing
- **Standard Plan**: Recommended for production
- **Pro Plan**: For high-traffic applications

### Resource Allocation:
- **CPU**: 0.5-1 CPU cores
- **Memory**: 512MB-1GB RAM
- **Storage**: 1GB for reports and cache

## 🔧 Troubleshooting

### Common Issues:

1. **Build Failures**:
   - Check Python version compatibility
   - Verify all dependencies in requirements.txt
   - Review build logs for specific errors

2. **Runtime Errors**:
   - Check environment variables are set correctly
   - Verify MongoDB connection string
   - Review application logs

3. **Performance Issues**:
   - Increase worker count for high traffic
   - Consider upgrading Render plan
   - Optimize database queries

### Debug Commands:
```bash
# Check health endpoint
curl https://your-app.onrender.com/health

# View logs in Render dashboard
# Go to your service → Logs tab
```

## 🚀 Post-Deployment

1. **Test All Features**:
   - Portfolio analysis
   - Chart generation
   - PDF export
   - Data fetching

2. **Monitor Performance**:
   - Use Render metrics dashboard
   - Set up alerts for downtime
   - Monitor response times

3. **Scale as Needed**:
   - Upgrade plan for more resources
   - Add horizontal scaling if required

## 🔐 Security Considerations

1. **Environment Variables**: Never commit secrets to repository
2. **HTTPS**: Render provides SSL certificates automatically
3. **Database Security**: Use MongoDB Atlas security features
4. **API Keys**: Rotate keys regularly

## 📈 Monitoring & Maintenance

1. **Health Checks**: Automatic via `/health` endpoint
2. **Logs**: Available in Render dashboard
3. **Metrics**: CPU, memory, and response time monitoring
4. **Updates**: Auto-deploy on git push (if enabled)

## 🆘 Support

- **Render Docs**: [render.com/docs](https://render.com/docs)
- **MongoDB Atlas**: [docs.atlas.mongodb.com](https://docs.atlas.mongodb.com)
- **Application Logs**: Check Render dashboard for detailed error messages
