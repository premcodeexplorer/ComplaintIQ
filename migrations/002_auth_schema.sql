-- =============================================================
--  ComplaintIQ — Auth Schema Migration (002)
--  Target: Supabase / PostgreSQL 15+
--
--  Creates the user_profiles table for bank admin accounts.
--  Supabase Auth manages the actual credentials (auth.users).
--  This table stores the app-level role + display name.
-- =============================================================

CREATE TABLE IF NOT EXISTS user_profiles (
    id          UUID        PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
    email       TEXT        NOT NULL UNIQUE,
    full_name   TEXT,
    role        TEXT        NOT NULL DEFAULT 'admin'
                            CHECK (role IN ('admin')),          -- only admins for now
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_login  TIMESTAMPTZ
);

-- Index for email look-ups during login
CREATE INDEX IF NOT EXISTS idx_user_profiles_email ON user_profiles(email);

-- Comment
COMMENT ON TABLE user_profiles IS
    'Bank admin accounts. Identity is managed by Supabase Auth (auth.users); '
    'this table stores the app role and display name.';
