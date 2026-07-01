import React, { useState, useRef } from 'react';
import { checkRateLimit, recordSubmission } from '../utils/rateLimit';
import { Send, AlertCircle, CheckCircle2, Loader2, Mic, Square } from 'lucide-react';
import toast from 'react-hot-toast';

export default function ComplaintForm({ user }) {
  const [formData, setFormData] = useState({
    account_no: user?.accountNumber || '',
    email: '',
    customer_name: user?.customerName || '',
    complaint_text: '',
    language: 'English',
    location: '',
    account_type: '',
    amount_involved: '',
  });

  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [audioBlob, setAudioBlob] = useState(null);
  const [recordingTime, setRecordingTime] = useState(0);
  const [transcription, setTranscription] = useState('');
  const mediaRecorderRef = useRef(null);

  const timerRef = useRef(null);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));
  };

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;
      const chunks = [];

      mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunks.push(e.data);
      };

      mediaRecorder.onstop = () => {
        const blob = new Blob(chunks, { type: 'audio/webm' });
        setAudioBlob(blob);
        stream.getTracks().forEach(track => track.stop());
      };

      mediaRecorder.start();
      setIsRecording(true);
      setRecordingTime(0);
      setAudioBlob(null);
      setFormData(prev => ({ ...prev, complaint_text: '' })); // clear text if recording

      timerRef.current = setInterval(() => {
        setRecordingTime(prev => prev + 1);
      }, 1000);

    } catch (err) {
      toast.error('Microphone access denied or not available.');
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
      clearInterval(timerRef.current);
    }
  };

  const removeAudio = () => {
    setAudioBlob(null);
    setRecordingTime(0);
  };

  const formatTime = (seconds) => {
    const m = Math.floor(seconds / 60).toString().padStart(2, '0');
    const s = (seconds % 60).toString().padStart(2, '0');
    return `${m}:${s}`;
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!audioBlob && !formData.complaint_text.trim()) {
      toast.error('Please either record an audio or type your complaint details.');
      return;
    }

    const rateLimit = checkRateLimit();
    if (!rateLimit.allowed) {
      toast.error(rateLimit.message, { duration: 5000 });
      return;
    }

    setIsSubmitting(true);
    
    try {
      const apiUrl = import.meta.env.VITE_API_URL || 'http://localhost:8000';
      let endpoint = '';
      let fetchOptions = {};

      if (audioBlob) {
        // Submit via Voice
        endpoint = `${apiUrl}/voice-complaint`;
        const payload = new FormData();
        payload.append('audio', audioBlob, 'voice_complaint.webm');
        payload.append('account_number', user.accountNumber);
        payload.append('customer_name', user.customerName);
        if (formData.location) payload.append('location', formData.location);
        if (formData.account_type) payload.append('account_type', formData.account_type);
        if (formData.amount_involved) payload.append('amount_involved', formData.amount_involved);

        fetchOptions = {
          method: 'POST',
          body: payload, // no content-type header for FormData
        };
      } else {
        // Submit via Text to Netlify/FastAPI
        endpoint = `${apiUrl}/complaint`;
        fetchOptions = {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            ...formData,
            customer_name: user.customerName,
            account_number: user.accountNumber, // Passed to text endpoint too
          })
        };
      }

      const response = await fetch(endpoint, fetchOptions);
      const data = await response.json();

      if (!response.ok) {
        if (data.detail && typeof data.detail === 'object') {
          if (data.detail.transcribed_text) setTranscription(data.detail.transcribed_text);
          throw new Error(data.detail.message || 'Failed to submit complaint.');
        }
        throw new Error(data.detail || data.error || 'Failed to submit complaint.');
      }

      if (data.transcribed_text) {
        setTranscription(data.transcribed_text);
      }

      recordSubmission();

      toast.success(data.message || 'Your complaint has been successfully registered.', { duration: 5000 });
      
      setFormData({
        account_no: user?.accountNumber || '',
        email: '',
        customer_name: user?.customerName || '',
        complaint_text: '',
        language: 'English',
        location: '',
        account_type: '',
        amount_involved: '',
      });
      removeAudio();
      
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
          <div className="form-group">
            <label className="form-label" htmlFor="email">Email Address *</label>
            <input
              type="email"
              id="email"
              name="email"
              className="form-input"
              placeholder="john@example.com"
              value={formData.email}
              onChange={handleChange}
              required
              disabled={isSubmitting}
            />
          </div>
        </div>

        <div className="form-group">
          <label className="form-label" htmlFor="complaint_text">Complaint Details *</label>
          
          <div style={{ display: 'flex', gap: '1rem', marginBottom: '1rem' }}>
            {!isRecording && !audioBlob && (
              <button type="button" onClick={startRecording} className="btn-primary" style={{ backgroundColor: '#ef4444', flex: 1 }}>
                <Mic size={18} /> Record Voice Complaint
              </button>
            )}
            
            {isRecording && (
              <button type="button" onClick={stopRecording} className="btn-primary" style={{ backgroundColor: '#10b981', flex: 1, animation: 'pulse 2s infinite' }}>
                <Square size={18} /> Stop Recording ({formatTime(recordingTime)})
              </button>
            )}

            {audioBlob && (
              <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: '0.5rem', background: '#f1f5f9', padding: '0.5rem 1rem', borderRadius: '8px' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
                  <span style={{ color: '#10b981', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                    <CheckCircle2 size={18} /> Voice Recorded ({formatTime(recordingTime)})
                  </span>
                  <button type="button" onClick={removeAudio} style={{ background: 'transparent', border: 'none', color: '#ef4444', cursor: 'pointer', textDecoration: 'underline' }}>
                    Remove
                  </button>
                </div>
                {/* Audio playback to help diagnose silent microphone issues */}
                <audio controls src={URL.createObjectURL(audioBlob)} style={{ height: '30px', width: '100%' }} />
              </div>
            )}

          </div>

          <textarea
            id="complaint_text"
            name="complaint_text"
            className="form-textarea"
            placeholder="Or describe your issue in detail by typing here..."
            value={formData.complaint_text}
            onChange={handleChange}
            disabled={isSubmitting || audioBlob || isRecording}
          ></textarea>
        </div>
        
        {transcription && (
          <div style={{ marginBottom: '1.5rem', padding: '1rem', backgroundColor: '#f1f5f9', borderRadius: '8px', borderLeft: '4px solid #3b82f6' }}>
            <h3 style={{ fontSize: '0.9rem', marginBottom: '0.5rem', color: 'var(--text-secondary)' }}>What we heard (Transcription):</h3>
            <p style={{ fontStyle: 'italic', color: 'var(--text-primary)', fontSize: '0.95rem' }}>"{transcription}"</p>
          </div>
        )}

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
