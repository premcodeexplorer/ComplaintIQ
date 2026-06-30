-- Migration 003: Portal Authentication and Spam Filtering

CREATE TABLE IF NOT EXISTS bank_customers (
    account_number TEXT PRIMARY KEY,
    customer_name  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS spam_penalties (
    id             SERIAL PRIMARY KEY,
    account_number TEXT NOT NULL,
    created_at     TEXT NOT NULL,
    FOREIGN KEY(account_number) REFERENCES bank_customers(account_number)
);

-- Insert dummy customer for testing authentication
INSERT INTO bank_customers (account_number, customer_name) 
VALUES ('1234567890', 'John Doe')
ON CONFLICT (account_number) DO NOTHING;
