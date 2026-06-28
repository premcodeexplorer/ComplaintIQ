import React, { useState } from 'react';
import { checkRateLimit, recordSubmission } from '../utils/rateLimit';
import { Send, AlertCircle, CheckCircle2, Loader2 } from 'lucide-react';
import toast from 'react-hot-toast';

export default function ComplaintForm() {
  const [formData, setFormData] = useState({
    customer_name: '',
    complaint_text: '',
    language: 'English',
    location: '',
    account_type: '',
    amount_involved: '',
  });

  const [isSubmitting, setIsSubmitting] = useState(false);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!formData.customer_name.trim() || !formData.complaint_text.trim()) {
      toast.error('Please fill in your name and the complaint details.');
      return;
    }

    // Client-side rate limiting (optional fallback layer)
    const rateLimit = checkRateLimit();
    if (!rateLimit.allowed) {
      toast.error(rateLimit.message, { duration: 5000 });
      return;
    }

    setIsSubmitting(true);
    
    try {
      // Send the data to our secure Netlify Serverless Function
      const response = await fetch('/api/submit-complaint', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(formData)
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || 'Failed to submit complaint.');
      }

      // Record successful submission in local storage as a fallback tracker
      recordSubmission();
      toast.success(data.message || 'Your complaint has been successfully registered.', { duration: 5000 });
      
      // Reset form
      setFormData({
        customer_name: '',
        complaint_text: '',
        language: 'English',
        location: '',
        account_type: '',
        amount_involved: '',
      });
      
    } catch (err) {
      toast.error(err.message || 'Failed to submit complaint. Please try again later.');
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="glass-card">
      <h2 className="title" style={{ fontSize: '1.8rem' }}>Register a Complaint</h2>
      <p className="subtitle" style={{ marginBottom: '1.5rem', fontSize: '0.95rem' }}>
        We are here to help. Please provide the details of your issue below.
      </p>

      <form onSubmit={handleSubmit}>
        <div className="form-group">
          <label className="form-label" htmlFor="customer_name">Full Name *</label>
          <input
            type="text"
            id="customer_name"
            name="customer_name"
            className="form-input"
            placeholder="John Doe"
            value={formData.customer_name}
            onChange={handleChange}
            required
            disabled={isSubmitting}
          />
        </div>

        <div className="form-group">
          <label className="form-label" htmlFor="complaint_text">Complaint Details *</label>
          <textarea
            id="complaint_text"
            name="complaint_text"
            className="form-textarea"
            placeholder="Please describe your issue in detail..."
            value={formData.complaint_text}
            onChange={handleChange}
            required
            disabled={isSubmitting}
          ></textarea>
        </div>

        <div className="form-row">
          <div className="form-group">
            <label className="form-label" htmlFor="location">City / Branch</label>
            <input
              type="text"
              id="location"
              name="location"
              className="form-input"
              placeholder="e.g. Mumbai Main Branch"
              value={formData.location}
              onChange={handleChange}
              disabled={isSubmitting}
            />
          </div>
          
          <div className="form-group">
            <label className="form-label" htmlFor="language">Preferred Language</label>
            <select
              id="language"
              name="language"
              className="form-select"
              value={formData.language}
              onChange={handleChange}
              disabled={isSubmitting}
            >
              <option value="English">English</option>
              <option value="Hindi">Hindi</option>
              <option value="Spanish">Spanish</option>
              <option value="Other">Other</option>
            </select>
          </div>
        </div>

        <div className="form-row">
          <div className="form-group">
            <label className="form-label" htmlFor="account_type">Account Type (Optional)</label>
            <select
              id="account_type"
              name="account_type"
              className="form-select"
              value={formData.account_type}
              onChange={handleChange}
              disabled={isSubmitting}
            >
              <option value="">Select an option</option>
              <option value="Savings">Savings Account</option>
              <option value="Current">Current Account</option>
              <option value="Credit Card">Credit Card</option>
              <option value="Loan">Loan</option>
            </select>
          </div>
          
          <div className="form-group">
            <label className="form-label" htmlFor="amount_involved">Amount Involved (if any)</label>
            <input
              type="number"
              id="amount_involved"
              name="amount_involved"
              className="form-input"
              placeholder="0.00"
              min="0"
              step="0.01"
              value={formData.amount_involved}
              onChange={handleChange}
              disabled={isSubmitting}
            />
          </div>
        </div>

        <button type="submit" className="btn-primary" disabled={isSubmitting}>
          {isSubmitting ? (
            <>
              <div className="spinner"></div>
              Submitting...
            </>
          ) : (
            <>
              <Send size={18} />
              Submit Complaint
            </>
          )}
        </button>
      </form>
    </div>
  );
}
