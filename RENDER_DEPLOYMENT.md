# 🚀 Render Deployment Guide

Complete guide to deploy Portfolio Risk & Return Analyzer on Render platform.

## 📋 **Pre-Deployment Checklist**

### ✅ **Required Files (Already Included)**
- `render.yaml` - Service configuration
- `requirements.txt` - Python dependencies
- `runtime.txt` - Python version (3.11.5)
- `Procfile` - Process commands
- `render-build.sh` - Build script
- `render-start.sh` - Start script
- `.gitignore` - Git ignore rules

### ✅ **Required Accounts & API Keys**
- **Render Account**: [render.com](https://render.com)
- **GitHub Account**: Repository connected
- **MongoDB Atlas**: Database cluster
- **Finnhub API**: Market data
- **Twelve Data API**: Historical data
- **Gmail Account**: Email reports (optional)

## 🔧 **Step-by-Step Deployment**

### **Step 1: Connect GitHub Repository**
1. Login to [Render Dashboard](https://dashboard.render.com)
2. Click **"New +"** → **"Web Service"**
3. Connect your GitHub account
4. Select repository: `Mayur8182/Portfolio-Risk-Return-Analyzer-`
5. Click **"Connect"**

### **Step 2: Configure Service Settings**
```yaml
Name: portfolio-risk-analyzer
Environment: Python
Region: Oregon (recommended)
Branch: main
Build Command: ./render-build.sh
Start Command: ./render-start.sh
```

### **Step 3: Set Environment Variables**
Add these environment variables in Render dashboard:

#### **🔑 Required Variables**
```env
MONGODB_URI=mongodb+srv://mkbharvad8080:Mkb%408080@cluster0.a82h2.mongodb.net/
FINNHUB_API_KEY=d1hodf1r01qsvr2be4r0d1hodf1r01qsvr2be4rg
TWELVEDATA_API_KEY=b57491954ba64f1d9c851fb4c157bf51
FLASK_ENV=production
PORT=5000
```

#### **📧 Email Configuration (Optional)**
```env
SMTP_USERNAME=mkbharvad534@gmail.com
SMTP_PASSWORD=your_gmail_app_password
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
ENABLE_EMAIL_REPORTS=true
```

#### **⚙️ Performance Settings**
```env
ENHANCED_MODE=true
INSTITUTIONAL_GRADE_THRESHOLD=0.995
MIN_DATA_QUALITY_SCORE=0.85
CACHE_TIMEOUT=3600
API_RATE_LIMIT=100
WORKERS=4
TIMEOUT=120
```

### **Step 4: Advanced Configuration**
```env
PYTHON_VERSION=3.11.5
FLASK_DEBUG=false
LOG_LEVEL=INFO
ENABLE_ENHANCED_ANALYTICS=true
ENABLE_REAL_TIME_DATA=true
ENABLE_PDF_EXPORT=true
WORKER_CONNECTIONS=1000
```

## 🎯 **Render Service Configuration**

### **Auto-Deploy Settings**
- ✅ **Auto-Deploy**: Enabled
- ✅ **Health Check**: `/api/health`
- ✅ **Build Command**: `./render-build.sh`
- ✅ **Start Command**: `./render-start.sh`

### **Resource Allocation**
- **Plan**: Starter ($7/month) or Free
- **Region**: Oregon (fastest for US)
- **Disk**: 1GB for reports storage
- **Memory**: 512MB (Starter) / 1GB (Pro)

## 📊 **Deployment Process**

### **Build Phase** (render-build.sh)
```bash
#!/bin/bash
echo "🔧 Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "🗂️ Creating necessary directories..."
mkdir -p data/cache
mkdir -p reports
mkdir -p backend/reports

echo "✅ Build completed successfully!"
```

### **Start Phase** (render-start.sh)
```bash
#!/bin/bash
echo "🚀 Starting Portfolio Risk Analyzer..."
cd backend
python app.py
```

## 🔍 **Post-Deployment Verification**

### **Health Check Endpoints**
- `GET /api/health` - Service health
- `GET /` - Frontend homepage
- `POST /api/analyze` - Portfolio analysis

### **Expected Response Times**
- **Homepage Load**: <2 seconds
- **Portfolio Analysis**: 15-30 seconds
- **PDF Generation**: 5-10 seconds
- **Data Fetching**: 10-20 seconds

## 🛠️ **Troubleshooting**

### **Common Issues & Solutions**

#### **Build Failures**
```bash
# Check Python version
python --version  # Should be 3.11.5

# Check dependencies
pip list | grep -E "(flask|pandas|numpy)"

# Clear cache and rebuild
rm -rf __pycache__
pip install --no-cache-dir -r requirements.txt
```

#### **Runtime Errors**
```bash
# Check environment variables
echo $MONGODB_URI
echo $FINNHUB_API_KEY

# Check MongoDB connection
python -c "import pymongo; print('MongoDB OK')"

# Check API keys
curl -H "X-Finnhub-Token: $FINNHUB_API_KEY" \
     "https://finnhub.io/api/v1/quote?symbol=AAPL"
```

#### **Performance Issues**
```env
# Increase timeout
TIMEOUT=180

# Reduce concurrent requests
API_RATE_LIMIT=50

# Enable caching
CACHE_TIMEOUT=7200
```

## 📈 **Monitoring & Logs**

### **Render Dashboard Monitoring**
- **CPU Usage**: Monitor for spikes
- **Memory Usage**: Should stay under 80%
- **Response Times**: Track API performance
- **Error Rates**: Monitor 4xx/5xx errors

### **Application Logs**
```bash
# View recent logs
render logs --service portfolio-risk-analyzer --tail

# Filter error logs
render logs --service portfolio-risk-analyzer | grep ERROR

# Monitor real-time
render logs --service portfolio-risk-analyzer --follow
```

## 🔒 **Security Best Practices**

### **Environment Variables**
- ✅ Never commit API keys to Git
- ✅ Use Render's encrypted environment variables
- ✅ Rotate API keys regularly
- ✅ Use strong MongoDB passwords

### **API Security**
- ✅ Rate limiting enabled
- ✅ Input validation on all endpoints
- ✅ CORS properly configured
- ✅ HTTPS enforced by Render

## 💰 **Cost Optimization**

### **Free Tier Limitations**
- **Sleep after 15 minutes** of inactivity
- **750 hours/month** usage limit
- **Limited CPU/Memory**

### **Starter Plan Benefits** ($7/month)
- **No sleep** - always available
- **Unlimited hours**
- **Better performance**
- **Custom domains**

## 🚀 **Go Live Checklist**

- [ ] Repository pushed to GitHub
- [ ] Render service created and configured
- [ ] All environment variables set
- [ ] MongoDB Atlas accessible
- [ ] API keys valid and working
- [ ] Health check passing
- [ ] Frontend loading correctly
- [ ] Portfolio analysis working
- [ ] PDF export functional
- [ ] Email reports configured (optional)

## 📞 **Support**

### **Render Support**
- [Render Documentation](https://render.com/docs)
- [Render Community](https://community.render.com)
- [Render Status](https://status.render.com)

### **Application Support**
- **GitHub Issues**: Create issue for bugs
- **Email**: mkbharvad534@gmail.com
- **Documentation**: Check API_DOCUMENTATION.md

---

🎉 **Your Portfolio Risk Analyzer is now live on Render!**
