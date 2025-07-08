# ⚡ Quick Render Deployment

## 🚀 **5-Minute Deployment Guide**

### **Step 1: Create Render Service**
1. Go to [render.com](https://render.com) → Sign up/Login
2. Click **"New +"** → **"Web Service"**
3. Connect GitHub → Select `Mayur8182/Portfolio-Risk-Return-Analyzer-`
4. Click **"Connect"**

### **Step 2: Basic Configuration**
```
Name: portfolio-risk-analyzer
Environment: Python
Region: Oregon
Branch: main
Build Command: ./render-build.sh
Start Command: ./render-start.sh
```

### **Step 3: Add Environment Variables**
Copy-paste these in Render dashboard:

```env
MONGODB_URI=mongodb+srv://mkbharvad8080:Mkb%408080@cluster0.a82h2.mongodb.net/
FINNHUB_API_KEY=d1hodf1r01qsvr2be4r0d1hodf1r01qsvr2be4rg
TWELVEDATA_API_KEY=b57491954ba64f1d9c851fb4c157bf51
FLASK_ENV=production
PORT=5000
ENHANCED_MODE=true
PYTHON_VERSION=3.11.5
FLASK_DEBUG=false
LOG_LEVEL=INFO
ENABLE_ENHANCED_ANALYTICS=true
ENABLE_REAL_TIME_DATA=true
ENABLE_PDF_EXPORT=true
INSTITUTIONAL_GRADE_THRESHOLD=0.995
MIN_DATA_QUALITY_SCORE=0.85
CACHE_TIMEOUT=3600
API_RATE_LIMIT=100
WORKERS=4
WORKER_CONNECTIONS=1000
TIMEOUT=120
SMTP_USERNAME=mkbharvad534@gmail.com
SMTP_PASSWORD=dwtp fmiq miyl ccvq
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
ENABLE_EMAIL_REPORTS=true
```

### **Step 4: Deploy**
1. Click **"Create Web Service"**
2. Wait 5-10 minutes for build
3. Check logs for any errors
4. Test the deployed URL

## ✅ **Verification Steps**

1. **Homepage loads**: `https://your-app.onrender.com`
2. **Health check**: `https://your-app.onrender.com/api/health`
3. **Test analysis**: Try analyzing a portfolio
4. **PDF export**: Test PDF generation

## 🔧 **If Something Goes Wrong**

### **Build Fails**
- Check Python version in logs
- Verify requirements.txt exists
- Check render-build.sh permissions

### **App Won't Start**
- Check environment variables are set
- Verify MongoDB connection string
- Check API keys are valid

### **Slow Performance**
- Upgrade to Starter plan ($7/month)
- Check API rate limits
- Monitor resource usage

## 📞 **Need Help?**
- Check RENDER_DEPLOYMENT.md for detailed guide
- Create GitHub issue for bugs
- Email: mkbharvad534@gmail.com

---

🎉 **Your app should be live in 5-10 minutes!**
