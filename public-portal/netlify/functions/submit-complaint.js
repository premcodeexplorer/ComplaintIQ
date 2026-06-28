import { createClient } from '@supabase/supabase-js';
import { v4 as uuidv4 } from 'uuid';

// Simple in-memory rate limiter (Note: resets on serverless cold starts)
// For enterprise rate-limiting, use Upstash Redis or a Supabase rate_limits table.
const rateLimitCache = new Map();
const MAX_REQUESTS = 3; 
const WINDOW_MS = 24 * 60 * 60 * 1000; // 24 hours

export const handler = async (event, context) => {
  // Only allow POST requests
  if (event.httpMethod !== 'POST') {
    return { statusCode: 405, body: 'Method Not Allowed' };
  }

  // IP-based Rate Limiting
  const clientIp = event.headers['x-forwarded-for'] || event.headers['client-ip'] || 'unknown';
  const now = Date.now();
  
  if (clientIp !== 'unknown') {
    const record = rateLimitCache.get(clientIp);
    if (record) {
      if (now - record.firstRequestTime < WINDOW_MS) {
        if (record.count >= MAX_REQUESTS) {
          const hoursLeft = Math.ceil((WINDOW_MS - (now - record.firstRequestTime)) / (1000 * 60 * 60));
          return {
            statusCode: 429,
            body: JSON.stringify({ 
              error: `Rate limit exceeded. Try again in ${hoursLeft} hours.` 
            })
          };
        }
        record.count++;
      } else {
        rateLimitCache.set(clientIp, { count: 1, firstRequestTime: now });
      }
    } else {
      rateLimitCache.set(clientIp, { count: 1, firstRequestTime: now });
    }
  }

  try {
    const body = JSON.parse(event.body);
    
    // Validate required fields
    if (!body.customer_name || !body.complaint_text) {
      return {
        statusCode: 400,
        body: JSON.stringify({ error: 'Name and complaint details are required.' })
      };
    }

    // Initialize Supabase Client
    // We use environment variables that you'll configure in Netlify Dashboard
    const supabaseUrl = process.env.VITE_SUPABASE_URL || process.env.SUPABASE_URL;
    const supabaseAnonKey = process.env.VITE_SUPABASE_ANON_KEY || process.env.SUPABASE_ANON_KEY;

    if (!supabaseUrl || !supabaseAnonKey) {
      console.error("Missing Supabase credentials in Netlify env vars.");
      return { statusCode: 500, body: JSON.stringify({ error: 'Internal server configuration error.' }) };
    }

    const supabase = createClient(supabaseUrl, supabaseAnonKey);

    // Prepare payload
    const complaintId = `PORTAL-${uuidv4().slice(0, 8).toUpperCase()}`;
    const amount = body.amount_involved ? parseFloat(body.amount_involved) : null;

    const { error } = await supabase
      .from('complaints')
      .insert([
        {
          id: complaintId,
          customer_name: body.customer_name,
          channel: 'portal',
          complaint_text: body.complaint_text,
          language: body.language || 'English',
          date: new Date().toISOString(),
          location: body.location || null,
          account_type: body.account_type || null,
          amount_involved: isNaN(amount) ? null : amount,
          status: 'open'
        }
      ]);

    if (error) {
      console.error('Supabase Insertion Error:', error);
      throw error;
    }

    return {
      statusCode: 200,
      body: JSON.stringify({ success: true, message: 'Complaint registered successfully.', id: complaintId })
    };

  } catch (error) {
    console.error('Function Error:', error);
    return {
      statusCode: 500,
      body: JSON.stringify({ error: error.message || error.toString() || 'Unknown error' })
    };
  }
};
