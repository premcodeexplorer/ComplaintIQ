import React, { useState } from 'react';
import { LogIn, Key, Loader2 } from 'lucide-react';
import toast from 'react-hot-toast';

export default function Login({ onLogin }) {
  const [accountNumber, setAccountNumber] = useState('');
  const [otp, setOtp] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!accountNumber.trim()) {
      toast.error('Please enter your Account Number');
      return;
    }
    if (otp !== '1234') {
      toast.error('Invalid OTP. Please enter 1234 for demo.');
      return;
    }

    setIsLoading(true);
    try {
      // For development, assuming FastAPI runs on localhost:8000
      const apiUrl = import.meta.env.VITE_API_URL || 'http://localhost:8000';
      const response = await fetch(`${apiUrl}/verify-account`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ account_number: accountNumber })
      });
      
      const data = await response.json();
      
      if (!response.ok) {
        throw new Error(data.detail || 'Verification failed');
      }

      toast.success(`Welcome, ${data.customer_name}`);
      onLogin({ accountNumber, customerName: data.customer_name });
      
    } catch (err) {
      toast.error(err.message || 'Login failed. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="glass-card" style={{ maxWidth: '400px', margin: '0 auto' }}>
      <div style={{ textAlign: 'center', marginBottom: '2rem' }}>
        <h2 className="title" style={{ fontSize: '1.8rem', marginBottom: '0.5rem' }}>Secure Login</h2>
        <p className="subtitle" style={{ fontSize: '0.9rem' }}>Enter your banking details to proceed</p>
      </div>

      <form onSubmit={handleSubmit}>
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
            />
          </div>
        </div>

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
            />
          </div>
        </div>

        <button type="submit" className="btn-primary" disabled={isLoading} style={{ width: '100%' }}>
          {isLoading ? (
            <><div className="spinner"></div> Verifying...</>
          ) : (
            <><LogIn size={18} /> Secure Login</>
          )}
        </button>
      </form>
    </div>
  );
}
