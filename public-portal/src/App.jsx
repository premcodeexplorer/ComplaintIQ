import React, { useState } from 'react';
import { Toaster } from 'react-hot-toast';
import ComplaintForm from './components/ComplaintForm';
import Login from './components/Login';

function App() {
  const [user, setUser] = useState(null);

  return (

    <>
      <div className="bg-blobs">
        <div className="blob blob-1"></div>
        <div className="blob blob-2"></div>
      </div>
      
      <div className="app-container">
        <div style={{ textAlign: 'center', marginBottom: '3rem' }}>
          <h1 className="title">ComplaintIQ Portal</h1>
          <p className="subtitle">Secure, Anonymous, and Direct Communication Channel</p>
          {user && (
            <p style={{ color: '#94a3b8', fontSize: '0.9rem', marginTop: '0.5rem' }}>
              Logged in as {user.customerName} ({user.accountNumber})
            </p>
          )}
        </div>
        
        {!user ? (
          <Login onLogin={setUser} />
        ) : (
          <ComplaintForm user={user} />
        )}
      </div>


      <Toaster 
        position="top-center"
        toastOptions={{
          style: {
            background: '#1e293b',
            color: '#f8fafc',
            border: '1px solid rgba(255, 255, 255, 0.1)',
          },
          success: {
            iconTheme: {
              primary: '#10b981',
              secondary: '#1e293b',
            },
          },
          error: {
            iconTheme: {
              primary: '#ef4444',
              secondary: '#1e293b',
            },
          },
        }}
      />
    </>
  );
}

export default App;
