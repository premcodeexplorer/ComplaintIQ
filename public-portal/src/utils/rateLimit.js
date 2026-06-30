const RATE_LIMIT_KEY = 'complaintiq_rate_limit';
const MAX_COMPLAINTS_PER_DAY = 3;
const ONE_DAY_MS = 24 * 60 * 60 * 1000;

export const checkRateLimit = () => {
  try {
    const data = localStorage.getItem(RATE_LIMIT_KEY);
    const now = Date.now();
    
    if (!data) {
      return { allowed: true };
    }

    const { count, firstTimestamp } = JSON.parse(data);

    // If 24 hours have passed, reset the limit
    if (now - firstTimestamp > ONE_DAY_MS) {
      return { allowed: true };
    }

    if (count >= MAX_COMPLAINTS_PER_DAY) {
      const timeRemainingMs = ONE_DAY_MS - (now - firstTimestamp);
      const hoursRemaining = Math.ceil(timeRemainingMs / (1000 * 60 * 60));
      return { 
        allowed: false, 
        message: `Rate limit exceeded. You can submit another complaint in ${hoursRemaining} hours.`
      };
    }

    return { allowed: true };
  } catch (error) {
    console.error("Error checking rate limit:", error);
    // Fail open to avoid blocking legit users if localStorage is acting up
    return { allowed: true };
  }
};

export const recordSubmission = () => {
  try {
    const data = localStorage.getItem(RATE_LIMIT_KEY);
    const now = Date.now();

    if (!data) {
      localStorage.setItem(RATE_LIMIT_KEY, JSON.stringify({ count: 1, firstTimestamp: now }));
      return;
    }

    const { count, firstTimestamp } = JSON.parse(data);

    // If 24 hours have passed, start a new window
    if (now - firstTimestamp > ONE_DAY_MS) {
      localStorage.setItem(RATE_LIMIT_KEY, JSON.stringify({ count: 1, firstTimestamp: now }));
    } else {
      localStorage.setItem(RATE_LIMIT_KEY, JSON.stringify({ count: count + 1, firstTimestamp }));
    }
  } catch (error) {
    console.error("Error recording submission:", error);
  }
};
