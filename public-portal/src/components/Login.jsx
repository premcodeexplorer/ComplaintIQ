import React, { useState } from 'react';
import { LogIn, Key, Loader2 } from 'lucide-react';
import toast from 'react-hot-toast';
import { Turnstile } from '@marsidev/react-turnstile';

export default function Login({ onLogin }) {
  const [step, setStep] = useState(1);
  const [accountNumber, setAccountNumber] = useState('');
  const [otp, setOtp] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [captchaToken, setCaptchaToken] = useState(null);

  const turnstileSiteKey = import.meta.env.VITE_TURNSTILE_SITE_KEY || '1x00000000000000000000AA';
  const handleAccountSubmit = (e) => {
    e.preventDefault();
    if (!accountNumber.trim()) {
      toast.error('Please enter your Account Number');
      return;
    }
    // Simulate OTP generation and move to step 2
    toast.success('OTP sent to your registered mobile number.');
    setStep(2);
  };

  const handleOtpSubmit = async (e) => {
    e.preventDefault();
    if (otp !== '1234') {
      toast.error('Invalid OTP. Please enter 1234 for demo.');
      return;
    }
    
    if (!captchaToken) {
      toast.error('Please complete the bot verification.');
      return;
    }

    setIsLoading(true);
    try {
      // For development, assuming FastAPI runs on localhost:8000
      const apiUrl = import.meta.env.VITE_API_URL || 'http://localhost:8000';
      const response = await fetch(`${apiUrl}/verify-account`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ account_number: accountNumber, captcha_token: captchaToken })
      });
      
      const data = await response.json();
      
      if (!response.ok) {
        throw new Error(data.detail || 'Verification failed');
      }

      toast.success(`Welcome, ${data.customer_name}`);
      onLogin({ accountNumber, customerName: data.customer_name });
      
    } catch (err) {
      toast.error(err.message || 'Login failed. Please try again.');
      setStep(1); // Go back if verification totally fails
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="glass-card" style={{ maxWidth: '400px', margin: '0 auto' }}>
      <div style={{ textAlign: 'center', marginBottom: '2rem' }}>
        <h2 className="title" style={{ fontSize: '1.8rem', marginBottom: '0.5rem' }}>Secure Login</h2>
        <p className="subtitle" style={{ fontSize: '0.9rem' }}>
          {step === 1 ? 'Enter your banking details to proceed' : 'Enter the OTP sent to your mobile'}
        </p>
      </div>

      {step === 1 ? (
        <form onSubmit={handleAccountSubmit}>
          <div className="form-group">
            <label className="form-label" htmlFor="accountNumber">Account Number (Try 1234567890)</label>
            <div style={{ position: 'relative' }}>
              <input
                type="text"
                id="accountNumber"
                className="form-input"
                placeholder="e.g. 1234567890"
                value={accountNumber}
                onChange={(e) => setAccountNumber(e.target.value)}
                disabled={isLoading}
                autoFocus
              />
            </div>
          </div>

          <button type="submit" className="btn-primary" disabled={isLoading} style={{ width: '100%' }}>
            Next
          </button>
        </form>
      ) : (
        <form onSubmit={handleOtpSubmit}>
          <div className="form-group">
            <label className="form-label" htmlFor="otp">OTP Code</label>
            <div style={{ position: 'relative' }}>
              <input
                type="password"
                id="otp"
                className="form-input"
                placeholder="Enter 1234"
                value={otp}
                onChange={(e) => setOtp(e.target.value)}
                disabled={isLoading}
                autoFocus
              />
            </div>
          </div>

          <div style={{ display: 'flex', justifyContent: 'center', marginBottom: '1.5rem', minHeight: '65px' }}>
            <Turnstile 
              siteKey={turnstileSiteKey} 
              onSuccess={(token) => setCaptchaToken(token)} 
              onError={() => toast.error('Bot verification failed. Please try again.')}
              onExpire={() => setCaptchaToken(null)}
            />
          </div>

          <button type="submit" className="btn-primary" disabled={isLoading || !captchaToken} style={{ width: '100%' }}>
            {isLoading ? (
              <><div className="spinner"></div> Verifying...</>
            ) : (
              <><LogIn size={18} /> Secure Login</>
            )}
          </button>
          
          <button 
            type="button" 
            onClick={() => setStep(1)} 
            disabled={isLoading}
            style={{ 
              width: '100%', 
              background: 'transparent', 
              border: 'none', 
              color: '#94a3b8', 
              marginTop: '1rem',
              cursor: 'pointer',
              textDecoration: 'underline'
            }}
          >
            Back to Account Number
          </button>
        </form>
      )}
    </div>
  );
}

